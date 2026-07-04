//! DRM syncobj protocol
//!
//! This module implement the `linux-drm-syncobj-v1` protocol, used to support
//! explicit sync.
//!
//! Acquire fences can delay the surface transaction through [`DrmSyncPointBlocker`]
//! when the compositor installs the accompanying [`DrmSyncPointSource`] from
//! [`DrmSyncobjHandler::drm_syncobj_install_acquire_point_source`]. Compositors that
//! do not install the source may only accept already-signalled acquire points; unsignalled
//! acquire commits fail closed before the buffer becomes current.
//!
//! The server should only expose the protocol if [`supports_syncobj_eventfd`] returns
//! `true` and the compositor installs acquire sources through
//! [`DrmSyncobjHandler::drm_syncobj_install_acquire_point_source`]. Otherwise normal
//! unsignalled acquire commits cannot be delayed safely and will fail closed before the
//! buffer becomes current.
//!
//! The committed release fence is signalled when all references to a
//! [`Buffer`][crate::backend::renderer::utils::Buffer] are dropped. Pending
//! release points that are discarded before becoming current, such as invalid
//! commits or syncobj-surface teardown, are signalled as part of the discard.
//!
//! ```no_run
//! # use smithay::wayland::drm_syncobj::*;
//!
//! pub struct State {
//!     syncobj_state: Option<DrmSyncobjState>,
//! }
//!
//! impl DrmSyncobjHandler for State {
//!     fn drm_syncobj_state(&mut self) -> Option<&mut DrmSyncobjState> {
//!         self.syncobj_state.as_mut()
//!     }
//! }
//!
//! # let mut display = wayland_server::Display::<State>::new().unwrap();
//! # let display_handle = display.handle();
//! # let import_device = todo!();
//! # let can_install_acquire_sources = false;
//! let syncobj_state = if supports_syncobj_eventfd(&import_device) && can_install_acquire_sources {
//!     Some(DrmSyncobjState::new::<State>(&display_handle, import_device))
//! } else {
//!     None
//! };
//!
//! smithay::delegate_dispatch2!(State);
//! ```

use std::{
    cell::RefCell,
    os::unix::io::AsFd,
    sync::{Arc, Weak},
};
use tracing::warn;
use wayland_protocols::wp::linux_drm_syncobj::v1::server::{
    wp_linux_drm_syncobj_manager_v1::{self, WpLinuxDrmSyncobjManagerV1},
    wp_linux_drm_syncobj_surface_v1::{self, WpLinuxDrmSyncobjSurfaceV1},
    wp_linux_drm_syncobj_timeline_v1::{self, WpLinuxDrmSyncobjTimelineV1},
};
use wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource, Weak as WlWeak,
    backend::GlobalId, protocol::wl_surface::WlSurface,
};

use super::{
    compositor::{self, BufferAssignment, Cacheable, HookId, SurfaceAttributes, with_states},
    dmabuf::get_dmabuf,
};
use crate::{
    backend::{drm::DrmDeviceFd, renderer::sync::Fence},
    wayland::{Dispatch2, GlobalData, GlobalDispatch2},
};

mod sync_point;
pub use sync_point::*;

/// Test if DRM device supports `syncobj_eventfd`.
// Similar to test used in Mutter
pub fn supports_syncobj_eventfd(device: &DrmDeviceFd) -> bool {
    // Pass device as placeholder for eventfd as well, since `drm_ffi` requires
    // a valid fd.
    match drm_ffi::syncobj::eventfd(device.as_fd(), 0, 0, device.as_fd(), false) {
        Ok(_) => unreachable!(),
        Err(err) => err.kind() == std::io::ErrorKind::NotFound,
    }
}

/// Handler trait for DRM syncobj protocol.
pub trait DrmSyncobjHandler {
    /// Returns a mutable reference to the [`DrmSyncobjState`] delegate type
    fn drm_syncobj_state(&mut self) -> Option<&mut DrmSyncobjState>;

    /// Install the event source for a pending acquire sync point.
    ///
    /// Returning `true` tells the protocol implementation that the source has been registered and will
    /// arrange for [`compositor::CompositorClientState::blocker_cleared`] to be called when it fires, so
    /// the current commit may be delayed with the matching [`DrmSyncPointBlocker`]. Returning `false`
    /// means no source was installed; the commit will then proceed only if the acquire point is already
    /// signalled, otherwise the pending commit is discarded instead of exposing an unsignalled buffer.
    fn drm_syncobj_install_acquire_point_source(
        &mut self,
        _dh: &DisplayHandle,
        _surface: &WlSurface,
        _acquire_point: &DrmSyncPoint,
        _source: DrmSyncPointSource,
    ) -> bool {
        false
    }
}

/// Data associated with a drm syncobj global
#[allow(missing_debug_implementations)]
pub struct DrmSyncobjGlobalData {
    filter: Box<dyn for<'c> Fn(&'c Client) -> bool + Send + Sync>,
}

/// Pending DRM syncobj sync point state
#[derive(Debug, Default)]
pub struct DrmSyncobjCachedState {
    /// Timeline point signaled when buffer is ready to read
    pub acquire_point: Option<DrmSyncPoint>,
    /// Timeline point to be signaled when server is done with buffer
    pub release_point: Option<DrmSyncPoint>,
}

fn discard_invalid_pending_sync_points(pending: &mut DrmSyncobjCachedState) {
    pending.acquire_point = None;
    if let Some(release_point) = pending.release_point.take() {
        if let Err(err) = release_point.signal() {
            tracing::error!("Failed to signal syncobj release point: {}", err);
        }
    }
}

fn discard_invalid_pending_commit(
    surface_cached: &mut compositor::CachedState<SurfaceAttributes>,
    syncobj_pending: &mut DrmSyncobjCachedState,
    discard_buffer: bool,
) {
    let discarded_buffer = discard_buffer
        .then(|| surface_cached.pending().buffer.take())
        .flatten();
    discard_invalid_pending_sync_points(syncobj_pending);
    if let Some(BufferAssignment::NewBuffer(buffer)) = discarded_buffer {
        let buffer_is_retained = surface_cached.current_ref().references_buffer(&buffer)
            || surface_cached
                .cached()
                .any(|state| state.references_buffer(&buffer));
        if !buffer_is_retained {
            buffer.release();
        }
    }
}

fn discard_unwaitable_acquire_commit(surface: &WlSurface) {
    compositor::with_states(surface, |states| {
        let mut surface_cached = states.cached_state.get::<SurfaceAttributes>();
        let mut cached = states.cached_state.get::<DrmSyncobjCachedState>();
        discard_invalid_pending_commit(&mut surface_cached, cached.pending(), true);
    });
}

fn same_sync_point(a: &DrmSyncPoint, b: &DrmSyncPoint) -> bool {
    a.timeline == b.timeline && a.point == b.point
}

fn destroy_syncobj_surface_state(surface: &WlSurface) {
    with_states(surface, |states| {
        *states
            .data_map
            .get::<RefCell<Option<WpLinuxDrmSyncobjSurfaceV1>>>()
            .unwrap()
            .borrow_mut() = None;
        // Committed sync points should still be used, but pending points can be cleared.
        let mut cached = states.cached_state.get::<DrmSyncobjCachedState>();
        cached.pending().acquire_point = None;
        if let Some(release_point) = cached.pending().release_point.take() {
            if let Err(err) = release_point.signal() {
                tracing::error!("Failed to signal syncobj release point: {}", err);
            }
        }
    });
}

struct DrmSyncobjSurfaceHooks {
    _commit_hook_id: HookId,
    _destruction_hook_id: HookId,
}

fn ensure_surface_hooks<D>(surface: &WlSurface)
where
    D: DrmSyncobjHandler + 'static,
{
    let needs_install = with_states(surface, |states| {
        states.data_map.get::<DrmSyncobjSurfaceHooks>().is_none()
    });
    if needs_install {
        let commit_hook_id = compositor::add_pre_commit_hook::<D, _>(surface, commit_hook);
        let destruction_hook_id = compositor::add_destruction_hook::<D, _>(surface, destruction_hook);
        with_states(surface, |states| {
            states.data_map.get_or_insert(|| DrmSyncobjSurfaceHooks {
                _commit_hook_id: commit_hook_id,
                _destruction_hook_id: destruction_hook_id,
            });
        });
    }
}

impl Cacheable for DrmSyncobjCachedState {
    const APPLY_PRIORITY: i32 = 100;
    const DISCARD_PRIORITY: i32 = 100;

    fn commit(&mut self, _dh: &DisplayHandle) -> Self {
        Self {
            acquire_point: self.acquire_point.take(),
            release_point: self.release_point.take(),
        }
    }

    fn merge_into(self, into: &mut Self, _dh: &DisplayHandle) {
        match (self.acquire_point, self.release_point) {
            (Some(acquire_point), Some(release_point)) => {
                if let Some(previous_release_point) = &into.release_point {
                    if let Err(err) = previous_release_point.signal() {
                        tracing::error!("Failed to signal syncobj release point: {}", err);
                    }
                }
                into.acquire_point = Some(acquire_point);
                into.release_point = Some(release_point);
            }
            // An acquire point without a release point is an internal apply-only marker used
            // when an accepted implicit buffer removal/replacement retires the current explicit
            // sync state after the syncobj surface object is gone. Signal the release point that
            // is current at apply time, not the one that was current when the commit was staged.
            (Some(_clear_marker), None) => {
                if let Some(release_point) = &into.release_point {
                    if let Err(err) = release_point.signal() {
                        tracing::error!("Failed to signal syncobj release point: {}", err);
                    }
                }
                into.acquire_point = None;
                into.release_point = None;
            }
            // A release point without an acquire point is an internal conditional marker for
            // same-buffer reattach after a queued explicit-sync attach. If the queued attach was
            // discarded, the current acquire point differs from this marker and the reattach now
            // retires the old current buffer. If the queued attach applied, the same buffer remains
            // current and its sync state must be kept.
            (None, Some(clear_if_current_differs_from)) => {
                let should_clear = into
                    .acquire_point
                    .as_ref()
                    .is_some_and(|current| !same_sync_point(current, &clear_if_current_differs_from));
                if should_clear {
                    if let Some(release_point) = &into.release_point {
                        if let Err(err) = release_point.signal() {
                            tracing::error!("Failed to signal syncobj release point: {}", err);
                        }
                    }
                    into.acquire_point = None;
                    into.release_point = None;
                }
            }
            _ => {}
        }
    }

    fn discard(self, _current: &mut Self) {
        if let (Some(_), Some(release_point)) = (self.acquire_point, self.release_point) {
            if let Err(err) = release_point.signal() {
                tracing::error!("Failed to signal discarded syncobj release point: {}", err);
            }
        }
    }
}

/// Delegate type for a `wp_linux_drm_syncobj_manager_v1` global
#[derive(Debug)]
pub struct DrmSyncobjState {
    global: GlobalId,
    import_device: Option<DrmDeviceFd>,
    known_timelines: Vec<Weak<DrmTimelineInner>>,
}

impl DrmSyncobjState {
    /// Create a new `wp_linux_drm_syncobj_manager_v1` global
    ///
    /// The `import_device` will be used to import the syncobj fds, and wait on them.
    pub fn new<D>(display: &DisplayHandle, import_device: DrmDeviceFd) -> Self
    where
        D: GlobalDispatch<WpLinuxDrmSyncobjManagerV1, DrmSyncobjGlobalData>,
        D: 'static,
    {
        Self::new_with_filter::<D, _>(display, import_device, |_| true)
    }

    /// Create a new `wp_linuxdrm_syncobj_manager_v1` global with a client filter
    ///
    /// The `import_device` will be used to import the syncobj fds, and wait on them.
    pub fn new_with_filter<D, F>(display: &DisplayHandle, import_device: DrmDeviceFd, filter: F) -> Self
    where
        D: GlobalDispatch<WpLinuxDrmSyncobjManagerV1, DrmSyncobjGlobalData>,
        D: 'static,
        F: for<'c> Fn(&'c Client) -> bool + Send + Sync + 'static,
    {
        let global = display.create_global::<D, WpLinuxDrmSyncobjManagerV1, DrmSyncobjGlobalData>(
            1,
            DrmSyncobjGlobalData {
                filter: Box::new(filter),
            },
        );

        Self {
            global,
            import_device: Some(import_device),
            known_timelines: Vec::new(),
        }
    }

    #[cfg(test)]
    fn new_without_import_device_for_tests<D>(display: &DisplayHandle) -> Self
    where
        D: GlobalDispatch<WpLinuxDrmSyncobjManagerV1, DrmSyncobjGlobalData>,
        D: 'static,
    {
        let global = display.create_global::<D, WpLinuxDrmSyncobjManagerV1, DrmSyncobjGlobalData>(
            1,
            DrmSyncobjGlobalData {
                filter: Box::new(|_| true),
            },
        );

        Self {
            global,
            import_device: None,
            known_timelines: Vec::new(),
        }
    }

    /// Closes the current `import_device`, allowing compositors to acquire a new fd.
    pub fn close_device<'a>(&'a mut self) -> CloseGuard<'a> {
        self.import_device.take();
        CloseGuard { state: self }
    }

    /// Sets a new `import_device` to import the syncobj fds and wait on them.
    pub fn update_device(&mut self, import_device: DrmDeviceFd) {
        for timeline in self.known_timelines.iter().filter_map(Weak::upgrade) {
            if let Err(err) = timeline.update_device(&import_device) {
                warn!(?err, "Failed to update existing timeline");
            }
        }
        self.import_device = Some(import_device);
    }

    /// Destroys the state and returns the `GlobalId` for compositors to disable/destroy.
    ///
    /// Note: This will cause any future timeline import to raise a protocol error for
    /// clients that have bound this protocol until [`DrmSyncobjHandler::drm_syncobj_state`] returns `Some` again.
    pub fn into_global(self) -> GlobalId {
        for timeline in self.known_timelines.iter().filter_map(Weak::upgrade) {
            timeline.invalidate();
        }
        self.global
    }
}

/// Guard returned by [`DrmSyncobjState::close_device`] to allow temporarily closing the import device.
#[derive(Debug)]
#[must_use = "You need to assign a new import device to not break the drm_syncobj global."]
pub struct CloseGuard<'a> {
    state: &'a mut DrmSyncobjState,
}

impl<'a> CloseGuard<'a> {
    /// Sets a new `import_device` to import the syncobj fds and wait on them.
    pub fn update_device(self, import_device: DrmDeviceFd) {
        self.state.update_device(import_device);
    }
}

impl<D> GlobalDispatch2<WpLinuxDrmSyncobjManagerV1, D> for DrmSyncobjGlobalData
where
    D: Dispatch<WpLinuxDrmSyncobjManagerV1, GlobalData>,
{
    fn bind(
        &self,
        _state: &mut D,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WpLinuxDrmSyncobjManagerV1>,
        data_init: &mut DataInit<'_, D>,
    ) {
        data_init.init::<_, _>(resource, GlobalData);
    }

    fn can_view(&self, client: &Client) -> bool {
        (self.filter)(client)
    }
}

fn commit_hook<D: DrmSyncobjHandler>(data: &mut D, dh: &DisplayHandle, surface: &WlSurface) {
    enum ClearCurrentMode {
        Unconditional,
        IfQueuedSyncDiscarded,
    }

    let acquire_blocker_point = compositor::with_states(surface, |states| {
        let mut surface_cached = states.cached_state.get::<SurfaceAttributes>();
        let (has_new_buffer, new_buffer_is_unsupported) = {
            let surface_pending = surface_cached.pending();
            let new_buffer = surface_pending.buffer.as_ref().and_then(|buffer| match buffer {
                BufferAssignment::NewBuffer(buffer) => Some(buffer),
                _ => None,
            });
            (
                new_buffer.is_some(),
                new_buffer
                    .map(|buffer| get_dmabuf(buffer).is_err())
                    .unwrap_or(false),
            )
        };
        if let Some(data) = states
            .data_map
            .get::<RefCell<Option<WpLinuxDrmSyncobjSurfaceV1>>>()
        {
            if let Some(syncobj_surface) = data.borrow().as_ref() {
                let mut cached = states.cached_state.get::<DrmSyncobjCachedState>();
                let pending = cached.pending();
                let has_acquire_point = pending.acquire_point.is_some();
                let has_release_point = pending.release_point.is_some();
                if (has_acquire_point || has_release_point) && !has_new_buffer {
                    syncobj_surface.post_error(
                        wp_linux_drm_syncobj_surface_v1::Error::NoBuffer,
                        "sync point without buffer".to_string(),
                    );
                    discard_invalid_pending_commit(&mut surface_cached, pending, true);
                } else if has_new_buffer && !has_acquire_point {
                    syncobj_surface.post_error(
                        wp_linux_drm_syncobj_surface_v1::Error::NoAcquirePoint,
                        "buffer without acquire point".to_string(),
                    );
                    discard_invalid_pending_commit(&mut surface_cached, pending, true);
                } else if has_new_buffer && !has_release_point {
                    syncobj_surface.post_error(
                        wp_linux_drm_syncobj_surface_v1::Error::NoReleasePoint,
                        "buffer without release point".to_string(),
                    );
                    discard_invalid_pending_commit(&mut surface_cached, pending, true);
                } else if let (Some(acquire), Some(release)) =
                    (pending.acquire_point.as_ref(), pending.release_point.as_ref())
                {
                    if acquire.timeline == release.timeline && release.point <= acquire.point {
                        syncobj_surface.post_error(
                            wp_linux_drm_syncobj_surface_v1::Error::ConflictingPoints,
                            format!(
                                "release point {} is not greater than acquire point {}",
                                release.point, acquire.point
                            ),
                        );
                        discard_invalid_pending_commit(&mut surface_cached, pending, true);
                    } else if new_buffer_is_unsupported {
                        syncobj_surface.post_error(
                            wp_linux_drm_syncobj_surface_v1::Error::UnsupportedBuffer,
                            "sync points with non-dmabuf buffer".to_string(),
                        );
                        discard_invalid_pending_commit(&mut surface_cached, pending, true);
                    } else {
                        return pending.acquire_point.clone();
                    }
                }
            }
        }
        let pending_buffer_after_validation = match surface_cached.pending().buffer.as_ref() {
            Some(BufferAssignment::Removed) => Some(None),
            Some(BufferAssignment::NewBuffer(buffer)) => Some(Some(buffer.clone())),
            None => None,
        };
        let latest_queued_buffer_assignment = surface_cached
            .cached()
            .filter_map(|state| state.buffer.as_ref())
            .last();
        let clear_current_mode = match pending_buffer_after_validation {
            Some(None) => Some(ClearCurrentMode::Unconditional),
            Some(Some(buffer)) => match latest_queued_buffer_assignment {
                Some(BufferAssignment::NewBuffer(latest_buffer)) if latest_buffer == &buffer => {
                    Some(ClearCurrentMode::IfQueuedSyncDiscarded)
                }
                Some(BufferAssignment::NewBuffer(_)) | Some(BufferAssignment::Removed) => {
                    Some(ClearCurrentMode::Unconditional)
                }
                None if surface_cached.current_ref().references_buffer(&buffer) => None,
                None => Some(ClearCurrentMode::Unconditional),
            },
            None => None,
        };
        if let Some(clear_current_mode) = clear_current_mode {
            let mut cached = states.cached_state.get::<DrmSyncobjCachedState>();
            let should_clear_current = {
                let pending = cached.pending();
                pending.acquire_point.is_none() && pending.release_point.is_none()
            };
            if should_clear_current {
                let latest_effective_sync = cached
                    .cached()
                    .filter(|state| state.acquire_point.is_some() || state.release_point.is_some())
                    .last();
                let unconditional_marker = latest_effective_sync
                    .and_then(|state| state.acquire_point.clone())
                    .or_else(|| cached.current_ref().acquire_point.clone());
                match clear_current_mode {
                    ClearCurrentMode::Unconditional => {
                        cached.pending().acquire_point = unconditional_marker;
                    }
                    ClearCurrentMode::IfQueuedSyncDiscarded => {
                        if let Some(latest_effective_sync) = latest_effective_sync {
                            if latest_effective_sync.acquire_point.is_some()
                                && latest_effective_sync.release_point.is_some()
                            {
                                cached.pending().release_point = latest_effective_sync.acquire_point.clone();
                            } else if latest_effective_sync.acquire_point.is_none()
                                && latest_effective_sync.release_point.is_some()
                            {
                                cached.pending().release_point = latest_effective_sync.release_point.clone();
                            } else {
                                cached.pending().acquire_point = unconditional_marker;
                            }
                        }
                    }
                }
            }
        }
        None
    });

    let Some(acquire_point) = acquire_blocker_point else {
        return;
    };

    match acquire_point.generate_blocker() {
        Ok((blocker, source)) => {
            if data.drm_syncobj_install_acquire_point_source(dh, surface, &acquire_point, source) {
                compositor::add_blocker(surface, blocker);
            } else if acquire_point.is_signaled() {
                warn!(
                    "DRM syncobj acquire point source was not installed, but acquire point is already signalled"
                );
            } else {
                warn!(
                    "DRM syncobj acquire point source was not installed; discarding unsignalled acquire buffer/sync state"
                );
                discard_unwaitable_acquire_commit(surface);
            }
        }
        Err(err) => {
            if acquire_point.is_signaled() {
                warn!(
                    ?err,
                    "failed to create DRM syncobj acquire point blocker for already-signalled point"
                );
            } else {
                warn!(
                    ?err,
                    "failed to create DRM syncobj acquire point blocker; discarding unsignalled acquire buffer/sync state"
                );
                discard_unwaitable_acquire_commit(surface);
            }
        }
    }
}

fn destruction_hook<D: DrmSyncobjHandler>(_data: &mut D, surface: &WlSurface) {
    compositor::with_states(surface, |states| {
        let mut cached = states.cached_state.get::<DrmSyncobjCachedState>();
        if let Some(release_point) = &cached.pending().release_point {
            if let Err(err) = release_point.signal() {
                tracing::error!("Failed to signal syncobj release point: {}", err);
            }
        }
        if let Some(release_point) = &cached.current().release_point {
            if let Err(err) = release_point.signal() {
                tracing::error!("Failed to signal syncobj release point: {}", err);
            }
        }
    });
}

#[derive(Debug, Clone, Copy)]
enum PendingSyncPointKind {
    Acquire,
    Release,
}

fn set_pending_sync_point_from_timeline_resource(
    surface: &WlSurface,
    timeline: &WpLinuxDrmSyncobjTimelineV1,
    point_hi: u32,
    point_lo: u32,
    kind: PendingSyncPointKind,
) {
    let sync_point = DrmSyncPoint {
        timeline: timeline
            .data::<DrmSyncobjTimelineData>()
            .unwrap()
            .timeline
            .clone(),
        point: ((point_hi as u64) << 32) + (point_lo as u64),
    };
    with_states(surface, |states| {
        let mut cached = states.cached_state.get::<DrmSyncobjCachedState>();
        let cached_state = cached.pending();
        match kind {
            PendingSyncPointKind::Acquire => cached_state.acquire_point = Some(sync_point),
            PendingSyncPointKind::Release => cached_state.release_point = Some(sync_point),
        }
    });
}

impl<D> Dispatch2<WpLinuxDrmSyncobjManagerV1, D> for GlobalData
where
    D: Dispatch<WpLinuxDrmSyncobjSurfaceV1, DrmSyncobjSurfaceData>,
    D: Dispatch<WpLinuxDrmSyncobjTimelineV1, DrmSyncobjTimelineData>,
    D: DrmSyncobjHandler,
{
    fn request(
        &self,
        state: &mut D,
        _client: &Client,
        resource: &WpLinuxDrmSyncobjManagerV1,
        request: wp_linux_drm_syncobj_manager_v1::Request,
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, D>,
    ) {
        match request {
            wp_linux_drm_syncobj_manager_v1::Request::GetSurface { id, surface } => {
                let already_exists = with_states(&surface, |states| {
                    states
                        .data_map
                        .get::<RefCell<Option<WpLinuxDrmSyncobjSurfaceV1>>>()
                        .map(|v| v.borrow().is_some())
                        .unwrap_or(false)
                });
                if already_exists {
                    resource.post_error(
                        wp_linux_drm_syncobj_manager_v1::Error::SurfaceExists,
                        "the surface already has a syncobj_surface object associated".to_string(),
                    );
                    return;
                }
                ensure_surface_hooks::<D>(&surface);
                let syncobj_surface = data_init.init::<_, _>(
                    id,
                    DrmSyncobjSurfaceData {
                        surface: surface.downgrade(),
                    },
                );
                with_states(&surface, |states| {
                    let syncobj_surface_cell = states
                        .data_map
                        .get_or_insert(|| RefCell::new(None::<WpLinuxDrmSyncobjSurfaceV1>));
                    *syncobj_surface_cell.borrow_mut() = Some(syncobj_surface);
                });
            }
            wp_linux_drm_syncobj_manager_v1::Request::ImportTimeline { id, fd } => {
                if let Some(state) = state.drm_syncobj_state() {
                    match state.import_device.as_ref().map(|dev| DrmTimeline::new(dev, fd)) {
                        Some(Ok(timeline)) => {
                            state.known_timelines.push(Arc::downgrade(&timeline.0));
                            data_init.init::<_, _>(id, DrmSyncobjTimelineData { timeline });
                        }
                        Some(Err(err)) => {
                            resource.post_error(
                                wp_linux_drm_syncobj_manager_v1::Error::InvalidTimeline,
                                format!("failed to import syncobj timeline: {err}"),
                            );
                        }
                        None => {
                            resource.post_error(
                                wp_linux_drm_syncobj_manager_v1::Error::InvalidTimeline,
                                "failed to import syncobj timeline: No device".to_string(),
                            );
                        }
                    }
                } else {
                    resource.post_error(
                        wp_linux_drm_syncobj_manager_v1::Error::InvalidTimeline,
                        "global orphaned",
                    )
                }
            }
            wp_linux_drm_syncobj_manager_v1::Request::Destroy => {}
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
pub(crate) mod test_utils {
    use std::{
        cell::RefCell,
        io,
        os::fd::{AsFd, OwnedFd},
        os::unix::net::UnixStream,
        sync::{Arc, Weak},
        time::Duration,
    };

    use wayland_client::{
        Connection, Dispatch as ClientDispatch, QueueHandle, delegate_noop,
        protocol::{wl_buffer, wl_compositor, wl_registry, wl_surface},
    };
    use wayland_protocols::wp::linux_dmabuf::zv1::client::{zwp_linux_buffer_params_v1, zwp_linux_dmabuf_v1};
    use wayland_protocols::wp::linux_drm_syncobj::v1::client::{
        wp_linux_drm_syncobj_manager_v1, wp_linux_drm_syncobj_surface_v1, wp_linux_drm_syncobj_timeline_v1,
    };
    use wayland_protocols::wp::linux_drm_syncobj::v1::server::{
        wp_linux_drm_syncobj_surface_v1::WpLinuxDrmSyncobjSurfaceV1,
        wp_linux_drm_syncobj_timeline_v1::WpLinuxDrmSyncobjTimelineV1,
    };
    use wayland_server::{
        Client, Dispatch as ServerDispatch, Display, DisplayHandle, Resource,
        backend::{ClientData, ClientId, DisconnectReason, InitError},
        protocol::wl_buffer as server_wl_buffer,
    };

    use super::{
        DrmSyncPoint, DrmSyncPointSource, DrmSyncobjCachedState, DrmSyncobjHandler, DrmSyncobjState,
        DrmSyncobjSurfaceData, DrmSyncobjTimelineData, PendingSyncPointKind, destroy_syncobj_surface_state,
        ensure_surface_hooks, set_pending_sync_point_from_timeline_resource,
    };
    use crate::backend::allocator::{
        Buffer as AllocatorBuffer,
        dmabuf::{Dmabuf, DmabufSyncFlags},
    };
    use crate::backend::drm::DrmDeviceFd;
    use crate::backend::renderer::utils as renderer_utils;
    use crate::wayland::buffer::BufferHandler;
    use crate::wayland::compositor::{self, BufferAssignment, SurfaceAttributes, with_states};
    use crate::wayland::dmabuf::{DmabufGlobal, DmabufHandler, DmabufState, ImportNotifier, get_dmabuf};

    #[derive(Debug)]
    #[allow(dead_code)]
    pub(crate) struct ClientTimelineImportEvidence {
        pub(crate) known_timeline_count: usize,
    }

    #[derive(Debug)]
    #[allow(dead_code)]
    pub(crate) struct ClientSurfacePointEvidence {
        pub(crate) known_timeline_count: usize,
        pub(crate) acquire_point: u64,
        pub(crate) release_point: u64,
        pub(crate) acquire_release_same_timeline: bool,
    }

    #[derive(Debug)]
    #[allow(dead_code)]
    pub(crate) struct ClientDmabufCommitEvidence {
        pub(crate) known_timeline_count: usize,
        pub(crate) imported_dmabuf_syncable: bool,
        pub(crate) imported_dmabuf_matches_expected: bool,
        pub(crate) current_has_dmabuf: bool,
        pub(crate) acquire_point: Option<u64>,
        pub(crate) release_point: Option<u64>,
        pub(crate) acquire_release_same_timeline: bool,
        pub(crate) transaction_acquire_source_installed: Option<bool>,
        pub(crate) transaction_pending_before_acquire_signal: Option<bool>,
        pub(crate) transaction_released_after_acquire_signal: Option<bool>,
        pub(crate) removal_release_point_signaled_before_commit: Option<bool>,
        pub(crate) removal_release_point_signaled_after_commit: Option<bool>,
        pub(crate) removal_release_point_signaled_at_buffer_release_event: Option<bool>,
        pub(crate) removal_buffer_release_events: Option<usize>,
        pub(crate) current_has_dmabuf_after_removal: Option<bool>,
    }

    #[allow(dead_code)]
    pub(crate) struct ClientRendererDmabufCommitHarness {
        evidence: ClientDmabufCommitEvidence,
        server_surface: wayland_server::protocol::wl_surface::WlSurface,
        renderer_buffer: renderer_utils::Buffer,
        display: Display<SurfacePointProtocolServerState>,
        server_state: SurfacePointProtocolServerState,
        client_connection: Connection,
        event_queue: wayland_client::EventQueue<SurfacePointProtocolClientState>,
        client_state: SurfacePointProtocolClientState,
    }

    impl ClientRendererDmabufCommitHarness {
        pub(crate) fn evidence(&self) -> &ClientDmabufCommitEvidence {
            &self.evidence
        }

        pub(crate) fn surface(&self) -> &wayland_server::protocol::wl_surface::WlSurface {
            &self.server_surface
        }

        pub(crate) fn renderer_buffer(&self) -> &renderer_utils::Buffer {
            &self.renderer_buffer
        }

        pub(crate) fn renderer_buffer_acquire_point(&self) -> Option<u64> {
            self.renderer_buffer.acquire_point().map(|point| point.point)
        }

        pub(crate) fn renderer_buffer_release_point(&self) -> Option<u64> {
            self.renderer_buffer.release_point().map(|point| point.point)
        }

        pub(crate) fn renderer_buffer_has_dmabuf(&self) -> bool {
            get_dmabuf(self.renderer_buffer()).is_ok()
        }

        #[allow(dead_code)]
        pub(crate) fn commit_replacement_dmabuf(
            &mut self,
            timeline_fd: OwnedFd,
            source_dmabuf: Dmabuf,
            acquire_point: u64,
            release_point: u64,
        ) -> renderer_utils::Buffer {
            self.server_state.expected_dmabuf = Some(source_dmabuf.clone());
            self.server_state.last_imported_dmabuf_syncable = false;
            self.server_state.last_imported_dmabuf_matches_expected = false;

            let qh = self.event_queue.handle();
            let dmabuf_global = self
                .client_state
                .dmabuf
                .as_ref()
                .expect("test dmabuf global should remain bound");
            let syncobj_surface = self
                .client_state
                .syncobj_surface
                .as_ref()
                .expect("test syncobj surface should remain live");
            let syncobj_manager = self
                .client_state
                .syncobj_manager
                .as_ref()
                .expect("test syncobj manager should remain bound");
            let surface = self
                .client_state
                .surface
                .as_ref()
                .expect("test surface should remain live");
            let timeline = syncobj_manager.import_timeline(timeline_fd.as_fd(), &qh, ());

            let dmabuf_format = source_dmabuf.format();
            let dmabuf_size = source_dmabuf.size();
            let params = dmabuf_global.create_params(&qh, ());
            let modifier: u64 = dmabuf_format.modifier.into();
            for plane in &source_dmabuf.0.planes {
                params.add(
                    plane.fd.as_fd(),
                    plane.plane_idx,
                    plane.offset,
                    plane.stride,
                    (modifier >> 32) as u32,
                    modifier as u32,
                );
            }
            let buffer = params.create_immed(
                dmabuf_size.w,
                dmabuf_size.h,
                dmabuf_format.code as u32,
                zwp_linux_buffer_params_v1::Flags::empty(),
                &qh,
                (),
            );
            let (acquire_hi, acquire_lo) = (((acquire_point >> 32) as u32), acquire_point as u32);
            let (release_hi, release_lo) = (((release_point >> 32) as u32), release_point as u32);
            syncobj_surface.set_acquire_point(&timeline, acquire_hi, acquire_lo);
            syncobj_surface.set_release_point(&timeline, release_hi, release_lo);
            self.client_connection
                .flush()
                .expect("flush replacement dmabuf setup and syncobj surface point requests");
            pump_surface_point_protocol_server(&mut self.display, &mut self.server_state);
            let pending_acquire = with_states(&self.server_surface, |states| {
                let mut cached = states.cached_state.get::<DrmSyncobjCachedState>();
                cached
                    .pending()
                    .acquire_point
                    .as_ref()
                    .expect("replacement set_acquire_point should stage a pending acquire point")
                    .clone()
            });
            pending_acquire
                .signal()
                .expect("signal replacement acquire point for non-blocking explicit-sync commit helper");

            surface.attach(Some(&buffer), 0, 0);
            surface.commit();
            if let Some(previous_timeline) = self.client_state.timeline.take() {
                self.client_state.retained_timelines.push(previous_timeline);
            }
            if let Some(previous_buffer) = self.client_state.buffer.take() {
                self.client_state.retained_buffers.push(previous_buffer);
            }
            self.client_state.timeline = Some(timeline);
            self.client_state.params = Some(params);
            self.client_state.buffer = Some(buffer);
            self.client_connection
                .flush()
                .expect("flush replacement dmabuf surface commit request");
            pump_surface_point_protocol_server(&mut self.display, &mut self.server_state);

            let (
                current_has_dmabuf,
                staged_acquire_point,
                staged_release_point,
                acquire_release_same_timeline,
            ) = with_states(&self.server_surface, |states| {
                let mut attributes = states.cached_state.get::<SurfaceAttributes>();
                let current_buffer =
                    attributes
                        .current()
                        .buffer
                        .as_ref()
                        .and_then(|assignment| match assignment {
                            BufferAssignment::NewBuffer(buffer) => Some(buffer),
                            BufferAssignment::Removed => None,
                        });
                let current_has_dmabuf = current_buffer
                    .map(|buffer| get_dmabuf(buffer).is_ok())
                    .unwrap_or(false);

                let mut cached = states.cached_state.get::<DrmSyncobjCachedState>();
                let current = cached.current();
                let acquire = current.acquire_point.as_ref();
                let release = current.release_point.as_ref();
                (
                    current_has_dmabuf,
                    acquire.map(|point| point.point),
                    release.map(|point| point.point),
                    acquire
                        .zip(release)
                        .map(|(acquire, release)| acquire.timeline == release.timeline)
                        .unwrap_or(false),
                )
            });
            let known_timeline_count = self
                .server_state
                .syncobj_state
                .as_ref()
                .expect("test syncobj state should remain installed")
                .known_timelines
                .iter()
                .filter_map(Weak::upgrade)
                .count();
            self.evidence = ClientDmabufCommitEvidence {
                known_timeline_count,
                imported_dmabuf_syncable: self.server_state.last_imported_dmabuf_syncable,
                imported_dmabuf_matches_expected: self.server_state.last_imported_dmabuf_matches_expected,
                current_has_dmabuf,
                acquire_point: staged_acquire_point,
                release_point: staged_release_point,
                acquire_release_same_timeline,
                transaction_acquire_source_installed: None,
                transaction_pending_before_acquire_signal: None,
                transaction_released_after_acquire_signal: None,
                removal_release_point_signaled_before_commit: None,
                removal_release_point_signaled_after_commit: None,
                removal_release_point_signaled_at_buffer_release_event: None,
                removal_buffer_release_events: None,
                current_has_dmabuf_after_removal: None,
            };
            self.renderer_buffer =
                renderer_utils::with_renderer_surface_state(&self.server_surface, |state| {
                    state.buffer().cloned()
                })
                .flatten()
                .expect(
                    "replacement renderer commit handler should install a renderer-managed dmabuf buffer",
                );

            self.renderer_buffer.clone()
        }
    }

    struct ClientDmabufCommitArtifacts {
        evidence: ClientDmabufCommitEvidence,
        display: Display<SurfacePointProtocolServerState>,
        server_state: SurfacePointProtocolServerState,
        client_connection: Connection,
        event_queue: wayland_client::EventQueue<SurfacePointProtocolClientState>,
        client_state: SurfacePointProtocolClientState,
        server_surface: wayland_server::protocol::wl_surface::WlSurface,
    }

    #[derive(Default)]
    struct ClientDmabufCommitOptions {
        install_transaction_acquire_source: bool,
        remove_after_commit: bool,
        run_renderer_commit_handler: bool,
        acquire_sync_file: Option<OwnedFd>,
    }

    #[derive(Debug)]
    #[allow(dead_code)]
    pub(crate) struct ClientCommitProtocolErrorEvidence {
        pub(crate) object_interface: String,
        pub(crate) code: u32,
        pub(crate) current_acquire_point_present: bool,
        pub(crate) current_release_point_present: bool,
        pub(crate) release_point_signaled_before_discard: Option<bool>,
        pub(crate) release_point_signaled_after_discard: Option<bool>,
    }

    #[derive(Default)]
    struct TimelineProtocolClientData;

    impl ClientData for TimelineProtocolClientData {
        fn initialized(&self, _client_id: ClientId) {}

        fn disconnected(&self, _client_id: ClientId, _reason: DisconnectReason) {}
    }

    struct TimelineProtocolServerState {
        syncobj_state: Option<DrmSyncobjState>,
    }

    impl DrmSyncobjHandler for TimelineProtocolServerState {
        fn drm_syncobj_state(&mut self) -> Option<&mut DrmSyncobjState> {
            self.syncobj_state.as_mut()
        }
    }

    crate::delegate_dispatch2!(TimelineProtocolServerState);

    #[derive(Default)]
    struct TimelineProtocolClientState {
        syncobj_manager: Option<wp_linux_drm_syncobj_manager_v1::WpLinuxDrmSyncobjManagerV1>,
        timeline: Option<wp_linux_drm_syncobj_timeline_v1::WpLinuxDrmSyncobjTimelineV1>,
    }

    impl ClientDispatch<wl_registry::WlRegistry, ()> for TimelineProtocolClientState {
        fn event(
            state: &mut Self,
            registry: &wl_registry::WlRegistry,
            event: wl_registry::Event,
            _: &(),
            _: &Connection,
            qh: &QueueHandle<Self>,
        ) {
            if let wl_registry::Event::Global { name, interface, .. } = event {
                if interface.as_str() == "wp_linux_drm_syncobj_manager_v1" {
                    state.syncobj_manager = Some(
                        registry.bind::<wp_linux_drm_syncobj_manager_v1::WpLinuxDrmSyncobjManagerV1, _, _>(
                            name,
                            1,
                            qh,
                            (),
                        ),
                    );
                }
            }
        }
    }

    delegate_noop!(TimelineProtocolClientState: ignore wp_linux_drm_syncobj_manager_v1::WpLinuxDrmSyncobjManagerV1);
    delegate_noop!(TimelineProtocolClientState: ignore wp_linux_drm_syncobj_timeline_v1::WpLinuxDrmSyncobjTimelineV1);

    #[derive(Default)]
    struct SurfacePointProtocolClientData {
        compositor_state: compositor::CompositorClientState,
    }

    impl ClientData for SurfacePointProtocolClientData {
        fn initialized(&self, _client_id: ClientId) {}

        fn disconnected(&self, _client_id: ClientId, _reason: DisconnectReason) {}
    }

    struct SurfacePointProtocolServerState {
        compositor_state: compositor::CompositorState,
        syncobj_state: Option<DrmSyncobjState>,
        dmabuf_state: DmabufState,
        expected_dmabuf: Option<Dmabuf>,
        last_imported_dmabuf_syncable: bool,
        last_imported_dmabuf_matches_expected: bool,
        install_acquire_sources: bool,
        installed_acquire_points: Vec<DrmSyncPoint>,
        acquire_sources: Vec<DrmSyncPointSource>,
        committed_surfaces: usize,
        surfaces: Vec<wayland_server::protocol::wl_surface::WlSurface>,
        run_renderer_commit_handler: bool,
    }

    impl compositor::CompositorHandler for SurfacePointProtocolServerState {
        fn compositor_state(&mut self) -> &mut compositor::CompositorState {
            &mut self.compositor_state
        }

        fn client_compositor_state<'a>(
            &self,
            client: &'a wayland_server::Client,
        ) -> &'a compositor::CompositorClientState {
            &client
                .get_data::<SurfacePointProtocolClientData>()
                .expect("test client should carry compositor state")
                .compositor_state
        }

        fn new_surface(&mut self, surface: &wayland_server::protocol::wl_surface::WlSurface) {
            self.surfaces.push(surface.clone());
        }

        fn commit(&mut self, surface: &wayland_server::protocol::wl_surface::WlSurface) {
            if self.run_renderer_commit_handler {
                renderer_utils::on_commit_buffer_handler::<SurfacePointProtocolServerState>(surface);
            }
            self.committed_surfaces += 1;
        }
    }

    impl DrmSyncobjHandler for SurfacePointProtocolServerState {
        fn drm_syncobj_state(&mut self) -> Option<&mut DrmSyncobjState> {
            self.syncobj_state.as_mut()
        }

        fn drm_syncobj_install_acquire_point_source(
            &mut self,
            _dh: &DisplayHandle,
            _surface: &wayland_server::protocol::wl_surface::WlSurface,
            acquire_point: &DrmSyncPoint,
            source: DrmSyncPointSource,
        ) -> bool {
            if self.install_acquire_sources {
                self.installed_acquire_points.push(acquire_point.clone());
                self.acquire_sources.push(source);
                true
            } else {
                false
            }
        }
    }

    impl BufferHandler for SurfacePointProtocolServerState {
        fn buffer_destroyed(&mut self, _buffer: &server_wl_buffer::WlBuffer) {}
    }

    impl DmabufHandler for SurfacePointProtocolServerState {
        fn dmabuf_state(&mut self) -> &mut DmabufState {
            &mut self.dmabuf_state
        }

        fn dmabuf_imported(&mut self, _global: &DmabufGlobal, dmabuf: Dmabuf, notifier: ImportNotifier) {
            let syncable = dmabuf_is_kernel_syncable_for_tests(&dmabuf);
            let matches_expected = self
                .expected_dmabuf
                .as_ref()
                .map(|expected| dmabuf_matches_expected_metadata_for_tests(expected, &dmabuf))
                .unwrap_or(true);
            self.last_imported_dmabuf_syncable = syncable;
            self.last_imported_dmabuf_matches_expected = matches_expected;

            if syncable && matches_expected {
                let _ = notifier.successful::<SurfacePointProtocolServerState>();
            } else {
                notifier.failed();
            }
        }
    }

    fn dmabuf_matches_expected_metadata_for_tests(expected: &Dmabuf, actual: &Dmabuf) -> bool {
        expected.size() == actual.size()
            && expected.format() == actual.format()
            && expected.0.flags == actual.0.flags
            && expected.num_planes() == actual.num_planes()
            && expected.offsets().eq(actual.offsets())
            && expected.strides().eq(actual.strides())
    }

    fn dmabuf_is_kernel_syncable_for_tests(dmabuf: &Dmabuf) -> bool {
        dmabuf.num_planes() > 0
            && (0..dmabuf.num_planes()).all(|idx| {
                dmabuf
                    .sync_plane(idx, DmabufSyncFlags::READ | DmabufSyncFlags::START)
                    .is_ok()
                    && dmabuf
                        .sync_plane(idx, DmabufSyncFlags::READ | DmabufSyncFlags::END)
                        .is_ok()
            })
    }

    pub(crate) fn sync_point_signaled_for_tests(point: &DrmSyncPoint) -> bool {
        point
            .timeline
            .query_signalled_point()
            .expect("query test DRM syncobj timeline point")
            >= point.point
    }

    impl AsMut<compositor::CompositorState> for SurfacePointProtocolServerState {
        fn as_mut(&mut self) -> &mut compositor::CompositorState {
            &mut self.compositor_state
        }
    }

    crate::delegate_dispatch2!(SurfacePointProtocolServerState);

    #[derive(Default)]
    struct SurfacePointProtocolClientState {
        compositor: Option<wl_compositor::WlCompositor>,
        syncobj_manager: Option<wp_linux_drm_syncobj_manager_v1::WpLinuxDrmSyncobjManagerV1>,
        dmabuf: Option<zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1>,
        surface: Option<wl_surface::WlSurface>,
        buffer: Option<wl_buffer::WlBuffer>,
        buffer_release_events: usize,
        expected_release_point_for_release_event: Option<DrmSyncPoint>,
        release_point_signaled_at_buffer_release_event: Option<bool>,
        params: Option<zwp_linux_buffer_params_v1::ZwpLinuxBufferParamsV1>,
        syncobj_surface: Option<wp_linux_drm_syncobj_surface_v1::WpLinuxDrmSyncobjSurfaceV1>,
        timeline: Option<wp_linux_drm_syncobj_timeline_v1::WpLinuxDrmSyncobjTimelineV1>,
        retained_timelines: Vec<wp_linux_drm_syncobj_timeline_v1::WpLinuxDrmSyncobjTimelineV1>,
        retained_buffers: Vec<wl_buffer::WlBuffer>,
    }

    impl ClientDispatch<wl_registry::WlRegistry, ()> for SurfacePointProtocolClientState {
        fn event(
            state: &mut Self,
            registry: &wl_registry::WlRegistry,
            event: wl_registry::Event,
            _: &(),
            _: &Connection,
            qh: &QueueHandle<Self>,
        ) {
            if let wl_registry::Event::Global { name, interface, .. } = event {
                match interface.as_str() {
                    "wl_compositor" => {
                        state.compositor =
                            Some(registry.bind::<wl_compositor::WlCompositor, _, _>(name, 1, qh, ()))
                    }
                    "wp_linux_drm_syncobj_manager_v1" => {
                        state.syncobj_manager = Some(
                            registry
                                .bind::<wp_linux_drm_syncobj_manager_v1::WpLinuxDrmSyncobjManagerV1, _, _>(
                                    name,
                                    1,
                                    qh,
                                    (),
                                ),
                        );
                    }
                    "zwp_linux_dmabuf_v1" => {
                        state.dmabuf = Some(registry.bind::<zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1, _, _>(
                            name,
                            3,
                            qh,
                            (),
                        ));
                    }
                    _ => {}
                }
            }
        }
    }

    delegate_noop!(SurfacePointProtocolClientState: ignore wl_compositor::WlCompositor);
    delegate_noop!(SurfacePointProtocolClientState: ignore wl_surface::WlSurface);
    delegate_noop!(SurfacePointProtocolClientState: ignore zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1);
    delegate_noop!(SurfacePointProtocolClientState: ignore zwp_linux_buffer_params_v1::ZwpLinuxBufferParamsV1);
    delegate_noop!(SurfacePointProtocolClientState: ignore wp_linux_drm_syncobj_manager_v1::WpLinuxDrmSyncobjManagerV1);
    delegate_noop!(SurfacePointProtocolClientState: ignore wp_linux_drm_syncobj_surface_v1::WpLinuxDrmSyncobjSurfaceV1);
    delegate_noop!(SurfacePointProtocolClientState: ignore wp_linux_drm_syncobj_timeline_v1::WpLinuxDrmSyncobjTimelineV1);

    impl ClientDispatch<wl_buffer::WlBuffer, ()> for SurfacePointProtocolClientState {
        fn event(
            state: &mut Self,
            _proxy: &wl_buffer::WlBuffer,
            event: wl_buffer::Event,
            _: &(),
            _: &Connection,
            _: &QueueHandle<Self>,
        ) {
            if let wl_buffer::Event::Release = event {
                state.buffer_release_events += 1;
                if let Some(release_point) = state.expected_release_point_for_release_event.as_ref() {
                    state.release_point_signaled_at_buffer_release_event =
                        Some(sync_point_signaled_for_tests(release_point));
                }
            }
        }
    }

    fn pump_timeline_protocol_server(
        display: &mut Display<TimelineProtocolServerState>,
        state: &mut TimelineProtocolServerState,
    ) {
        display
            .dispatch_clients(state)
            .expect("dispatch test client requests");
        display.flush_clients().expect("flush test server events");
    }

    #[allow(dead_code)]
    pub(crate) fn import_timeline_through_client_for_tests(
        import_device: DrmDeviceFd,
        timeline_fd: OwnedFd,
    ) -> Option<ClientTimelineImportEvidence> {
        let mut display = match Display::<TimelineProtocolServerState>::new() {
            Ok(display) => display,
            Err(InitError::NoWaylandLib) => return None,
            Err(err) => panic!("failed to create test Wayland display: {err}"),
        };
        let mut display_handle = display.handle();
        let syncobj_state =
            DrmSyncobjState::new::<TimelineProtocolServerState>(&display_handle, import_device);
        let mut server_state = TimelineProtocolServerState {
            syncobj_state: Some(syncobj_state),
        };
        let (client_side, server_side) = UnixStream::pair().unwrap();
        let _server_client = display_handle
            .insert_client(server_side, Arc::new(TimelineProtocolClientData))
            .expect("insert test client");

        let client_connection = Connection::from_socket(client_side).expect("connect test client socket");
        let mut event_queue = client_connection.new_event_queue();
        let qh = event_queue.handle();
        let mut client_state = TimelineProtocolClientState::default();

        client_connection.display().get_registry(&qh, ());
        client_connection.flush().expect("flush get_registry");
        pump_timeline_protocol_server(&mut display, &mut server_state);
        event_queue
            .blocking_dispatch(&mut client_state)
            .expect("dispatch registry globals");

        let syncobj_manager = client_state
            .syncobj_manager
            .as_ref()
            .expect("test syncobj global should be advertised");
        let timeline = syncobj_manager.import_timeline(timeline_fd.as_fd(), &qh, ());
        client_state.timeline = Some(timeline);
        client_connection.flush().expect("flush import_timeline request");
        pump_timeline_protocol_server(&mut display, &mut server_state);

        let known_timeline_count = server_state
            .syncobj_state
            .as_ref()
            .expect("test syncobj state should remain installed")
            .known_timelines
            .iter()
            .filter_map(Weak::upgrade)
            .count();
        Some(ClientTimelineImportEvidence { known_timeline_count })
    }

    fn pump_surface_point_protocol_server(
        display: &mut Display<SurfacePointProtocolServerState>,
        state: &mut SurfacePointProtocolServerState,
    ) {
        display
            .dispatch_clients(state)
            .expect("dispatch test client requests");
        display.flush_clients().expect("flush test server events");
    }

    fn read_surface_point_protocol_error(
        event_queue: &mut wayland_client::EventQueue<SurfacePointProtocolClientState>,
        client_connection: &Connection,
        client_state: &mut SurfacePointProtocolClientState,
        current_acquire_point_present: bool,
        current_release_point_present: bool,
    ) -> ClientCommitProtocolErrorEvidence {
        let protocol_error = if let Some(guard) = event_queue.prepare_read() {
            let _ = guard.read();
            let _ = event_queue.dispatch_pending(client_state);
            client_connection.protocol_error()
        } else {
            let _ = event_queue.dispatch_pending(client_state);
            client_connection.protocol_error()
        }
        .expect("client request should disconnect with a protocol error");
        ClientCommitProtocolErrorEvidence {
            object_interface: protocol_error.object_interface,
            code: protocol_error.code,
            current_acquire_point_present,
            current_release_point_present,
            release_point_signaled_before_discard: None,
            release_point_signaled_after_discard: None,
        }
    }

    fn dispatch_surface_point_client_events(
        event_queue: &mut wayland_client::EventQueue<SurfacePointProtocolClientState>,
        client_connection: &Connection,
        client_state: &mut SurfacePointProtocolClientState,
    ) {
        if let Some(guard) = event_queue.prepare_read() {
            let _ = guard.read();
        }
        let _ = event_queue.dispatch_pending(client_state);
        if client_connection.protocol_error().is_some() {
            panic!("test client should not receive a protocol error while dispatching release events");
        }
    }

    fn read_surface_point_protocol_error_with_release_signal_evidence(
        event_queue: &mut wayland_client::EventQueue<SurfacePointProtocolClientState>,
        client_connection: &Connection,
        client_state: &mut SurfacePointProtocolClientState,
        current_acquire_point_present: bool,
        current_release_point_present: bool,
        release_point_signaled_before_discard: bool,
        release_point_signaled_after_discard: bool,
    ) -> ClientCommitProtocolErrorEvidence {
        let mut evidence = read_surface_point_protocol_error(
            event_queue,
            client_connection,
            client_state,
            current_acquire_point_present,
            current_release_point_present,
        );
        evidence.release_point_signaled_before_discard = Some(release_point_signaled_before_discard);
        evidence.release_point_signaled_after_discard = Some(release_point_signaled_after_discard);
        evidence
    }

    #[allow(dead_code)]
    pub(crate) fn import_timeline_and_set_surface_points_through_client_for_tests(
        import_device: DrmDeviceFd,
        timeline_fd: OwnedFd,
        acquire_point: u64,
        release_point: u64,
    ) -> Option<ClientSurfacePointEvidence> {
        let mut display = match Display::<SurfacePointProtocolServerState>::new() {
            Ok(display) => display,
            Err(InitError::NoWaylandLib) => return None,
            Err(err) => panic!("failed to create test Wayland display: {err}"),
        };
        let mut display_handle = display.handle();
        let compositor_state =
            compositor::CompositorState::new::<SurfacePointProtocolServerState>(&display_handle);
        let syncobj_state =
            DrmSyncobjState::new::<SurfacePointProtocolServerState>(&display_handle, import_device);
        let mut server_state = SurfacePointProtocolServerState {
            compositor_state,
            syncobj_state: Some(syncobj_state),
            dmabuf_state: DmabufState::new(),
            expected_dmabuf: None,
            last_imported_dmabuf_syncable: false,
            last_imported_dmabuf_matches_expected: false,
            install_acquire_sources: false,
            installed_acquire_points: Vec::new(),
            acquire_sources: Vec::new(),
            committed_surfaces: 0,
            surfaces: Vec::new(),
            run_renderer_commit_handler: false,
        };
        let (client_side, server_side) = UnixStream::pair().unwrap();
        let _server_client = display_handle
            .insert_client(server_side, Arc::new(SurfacePointProtocolClientData::default()))
            .expect("insert test client");

        let client_connection = Connection::from_socket(client_side).expect("connect test client socket");
        let mut event_queue = client_connection.new_event_queue();
        let qh = event_queue.handle();
        let mut client_state = SurfacePointProtocolClientState::default();

        client_connection.display().get_registry(&qh, ());
        client_connection.flush().expect("flush get_registry");
        pump_surface_point_protocol_server(&mut display, &mut server_state);
        event_queue
            .blocking_dispatch(&mut client_state)
            .expect("dispatch registry globals");

        let compositor = client_state
            .compositor
            .as_ref()
            .expect("test compositor global should be advertised");
        let syncobj_manager = client_state
            .syncobj_manager
            .as_ref()
            .expect("test syncobj global should be advertised");
        let surface = compositor.create_surface(&qh, ());
        let syncobj_surface = syncobj_manager.get_surface(&surface, &qh, ());
        let timeline = syncobj_manager.import_timeline(timeline_fd.as_fd(), &qh, ());
        let (acquire_hi, acquire_lo) = (((acquire_point >> 32) as u32), acquire_point as u32);
        let (release_hi, release_lo) = (((release_point >> 32) as u32), release_point as u32);
        syncobj_surface.set_acquire_point(&timeline, acquire_hi, acquire_lo);
        syncobj_surface.set_release_point(&timeline, release_hi, release_lo);
        client_state.surface = Some(surface);
        client_state.syncobj_surface = Some(syncobj_surface);
        client_state.timeline = Some(timeline);
        client_connection
            .flush()
            .expect("flush syncobj surface point requests");
        pump_surface_point_protocol_server(&mut display, &mut server_state);

        let server_surface = server_state
            .surfaces
            .first()
            .cloned()
            .expect("client create_surface should reach server state");
        let (staged_acquire_point, staged_release_point, acquire_release_same_timeline) =
            with_states(&server_surface, |states| {
                let mut cached = states.cached_state.get::<DrmSyncobjCachedState>();
                let pending = cached.pending();
                let acquire = pending
                    .acquire_point
                    .as_ref()
                    .expect("set_acquire_point should stage a pending acquire point");
                let release = pending
                    .release_point
                    .as_ref()
                    .expect("set_release_point should stage a pending release point");
                (acquire.point, release.point, acquire.timeline == release.timeline)
            });
        let known_timeline_count = server_state
            .syncobj_state
            .as_ref()
            .expect("test syncobj state should remain installed")
            .known_timelines
            .iter()
            .filter_map(Weak::upgrade)
            .count();

        Some(ClientSurfacePointEvidence {
            known_timeline_count,
            acquire_point: staged_acquire_point,
            release_point: staged_release_point,
            acquire_release_same_timeline,
        })
    }

    #[allow(dead_code)]
    pub(crate) fn commit_surface_points_without_buffer_through_client_for_tests(
        import_device: DrmDeviceFd,
        timeline_fd: OwnedFd,
        acquire_point: u64,
        release_point: u64,
    ) -> Option<ClientCommitProtocolErrorEvidence> {
        let mut display = match Display::<SurfacePointProtocolServerState>::new() {
            Ok(display) => display,
            Err(InitError::NoWaylandLib) => return None,
            Err(err) => panic!("failed to create test Wayland display: {err}"),
        };
        let mut display_handle = display.handle();
        let compositor_state =
            compositor::CompositorState::new::<SurfacePointProtocolServerState>(&display_handle);
        let syncobj_state =
            DrmSyncobjState::new::<SurfacePointProtocolServerState>(&display_handle, import_device);
        let mut server_state = SurfacePointProtocolServerState {
            compositor_state,
            syncobj_state: Some(syncobj_state),
            dmabuf_state: DmabufState::new(),
            expected_dmabuf: None,
            last_imported_dmabuf_syncable: false,
            last_imported_dmabuf_matches_expected: false,
            install_acquire_sources: false,
            installed_acquire_points: Vec::new(),
            acquire_sources: Vec::new(),
            committed_surfaces: 0,
            surfaces: Vec::new(),
            run_renderer_commit_handler: false,
        };
        let (client_side, server_side) = UnixStream::pair().unwrap();
        let _server_client = display_handle
            .insert_client(server_side, Arc::new(SurfacePointProtocolClientData::default()))
            .expect("insert test client");

        let client_connection = Connection::from_socket(client_side).expect("connect test client socket");
        let mut event_queue = client_connection.new_event_queue();
        let qh = event_queue.handle();
        let mut client_state = SurfacePointProtocolClientState::default();

        client_connection.display().get_registry(&qh, ());
        client_connection.flush().expect("flush get_registry");
        pump_surface_point_protocol_server(&mut display, &mut server_state);
        event_queue
            .blocking_dispatch(&mut client_state)
            .expect("dispatch registry globals");

        let compositor = client_state
            .compositor
            .as_ref()
            .expect("test compositor global should be advertised");
        let syncobj_manager = client_state
            .syncobj_manager
            .as_ref()
            .expect("test syncobj global should be advertised");
        let surface = compositor.create_surface(&qh, ());
        let syncobj_surface = syncobj_manager.get_surface(&surface, &qh, ());
        let timeline = syncobj_manager.import_timeline(timeline_fd.as_fd(), &qh, ());
        let (acquire_hi, acquire_lo) = (((acquire_point >> 32) as u32), acquire_point as u32);
        let (release_hi, release_lo) = (((release_point >> 32) as u32), release_point as u32);
        syncobj_surface.set_acquire_point(&timeline, acquire_hi, acquire_lo);
        syncobj_surface.set_release_point(&timeline, release_hi, release_lo);
        surface.commit();
        client_state.surface = Some(surface);
        client_state.syncobj_surface = Some(syncobj_surface);
        client_state.timeline = Some(timeline);
        client_connection
            .flush()
            .expect("flush syncobj surface commit request");
        pump_surface_point_protocol_server(&mut display, &mut server_state);

        let server_surface = server_state
            .surfaces
            .first()
            .cloned()
            .expect("client create_surface should reach server state");
        let (current_acquire_point_present, current_release_point_present) =
            with_states(&server_surface, |states| {
                let mut cached = states.cached_state.get::<DrmSyncobjCachedState>();
                let current = cached.current();
                (current.acquire_point.is_some(), current.release_point.is_some())
            });

        Some(read_surface_point_protocol_error(
            &mut event_queue,
            &client_connection,
            &mut client_state,
            current_acquire_point_present,
            current_release_point_present,
        ))
    }

    #[allow(dead_code)]
    pub(crate) fn commit_surface_points_without_buffer_and_probe_release_signal_through_client_for_tests(
        import_device: DrmDeviceFd,
        timeline_fd: OwnedFd,
        acquire_point: u64,
        release_point: u64,
    ) -> Option<ClientCommitProtocolErrorEvidence> {
        let mut display = match Display::<SurfacePointProtocolServerState>::new() {
            Ok(display) => display,
            Err(InitError::NoWaylandLib) => return None,
            Err(err) => panic!("failed to create test Wayland display: {err}"),
        };
        let mut display_handle = display.handle();
        let compositor_state =
            compositor::CompositorState::new::<SurfacePointProtocolServerState>(&display_handle);
        let syncobj_state =
            DrmSyncobjState::new::<SurfacePointProtocolServerState>(&display_handle, import_device);
        let mut server_state = SurfacePointProtocolServerState {
            compositor_state,
            syncobj_state: Some(syncobj_state),
            dmabuf_state: DmabufState::new(),
            expected_dmabuf: None,
            last_imported_dmabuf_syncable: false,
            last_imported_dmabuf_matches_expected: false,
            install_acquire_sources: false,
            installed_acquire_points: Vec::new(),
            acquire_sources: Vec::new(),
            committed_surfaces: 0,
            surfaces: Vec::new(),
            run_renderer_commit_handler: false,
        };
        let (client_side, server_side) = UnixStream::pair().unwrap();
        let _server_client = display_handle
            .insert_client(server_side, Arc::new(SurfacePointProtocolClientData::default()))
            .expect("insert test client");

        let client_connection = Connection::from_socket(client_side).expect("connect test client socket");
        let mut event_queue = client_connection.new_event_queue();
        let qh = event_queue.handle();
        let mut client_state = SurfacePointProtocolClientState::default();

        client_connection.display().get_registry(&qh, ());
        client_connection.flush().expect("flush get_registry");
        pump_surface_point_protocol_server(&mut display, &mut server_state);
        event_queue
            .blocking_dispatch(&mut client_state)
            .expect("dispatch registry globals");

        let compositor = client_state
            .compositor
            .as_ref()
            .expect("test compositor global should be advertised");
        let syncobj_manager = client_state
            .syncobj_manager
            .as_ref()
            .expect("test syncobj global should be advertised");
        let surface = compositor.create_surface(&qh, ());
        let syncobj_surface = syncobj_manager.get_surface(&surface, &qh, ());
        let timeline = syncobj_manager.import_timeline(timeline_fd.as_fd(), &qh, ());
        let (acquire_hi, acquire_lo) = (((acquire_point >> 32) as u32), acquire_point as u32);
        let (release_hi, release_lo) = (((release_point >> 32) as u32), release_point as u32);
        syncobj_surface.set_acquire_point(&timeline, acquire_hi, acquire_lo);
        syncobj_surface.set_release_point(&timeline, release_hi, release_lo);
        client_state.surface = Some(surface);
        client_state.syncobj_surface = Some(syncobj_surface);
        client_state.timeline = Some(timeline);
        client_connection
            .flush()
            .expect("flush syncobj surface point requests");
        pump_surface_point_protocol_server(&mut display, &mut server_state);

        let server_surface = server_state
            .surfaces
            .first()
            .cloned()
            .expect("client create_surface should reach server state");
        let release_sync_point = with_states(&server_surface, |states| {
            let mut cached = states.cached_state.get::<DrmSyncobjCachedState>();
            cached
                .pending()
                .release_point
                .as_ref()
                .expect("set_release_point should stage a pending release point")
                .clone()
        });
        let release_point_signaled_before_discard = sync_point_signaled_for_tests(&release_sync_point);

        let surface = client_state
            .surface
            .as_ref()
            .expect("test surface should remain live");
        surface.commit();
        client_connection
            .flush()
            .expect("flush syncobj surface commit request");
        pump_surface_point_protocol_server(&mut display, &mut server_state);

        let (current_acquire_point_present, current_release_point_present) =
            with_states(&server_surface, |states| {
                let mut cached = states.cached_state.get::<DrmSyncobjCachedState>();
                let current = cached.current();
                (current.acquire_point.is_some(), current.release_point.is_some())
            });
        let release_point_signaled_after_discard = sync_point_signaled_for_tests(&release_sync_point);

        Some(read_surface_point_protocol_error_with_release_signal_evidence(
            &mut event_queue,
            &client_connection,
            &mut client_state,
            current_acquire_point_present,
            current_release_point_present,
            release_point_signaled_before_discard,
            release_point_signaled_after_discard,
        ))
    }

    #[allow(dead_code)]
    pub(crate) fn commit_dmabuf_surface_with_sync_points_through_client_for_tests(
        import_device: DrmDeviceFd,
        timeline_fd: OwnedFd,
        source_dmabuf: Dmabuf,
        acquire_point: u64,
        release_point: u64,
    ) -> Option<ClientDmabufCommitEvidence> {
        commit_dmabuf_surface_with_sync_points_through_client_for_tests_impl(
            import_device,
            timeline_fd,
            source_dmabuf,
            acquire_point,
            release_point,
            ClientDmabufCommitOptions::default(),
            None,
        )
        .map(|artifacts| artifacts.evidence)
    }

    #[allow(dead_code)]
    pub(crate) fn commit_dmabuf_surface_with_renderer_buffer_through_client_for_tests(
        import_device: DrmDeviceFd,
        timeline_fd: OwnedFd,
        source_dmabuf: Dmabuf,
        acquire_point: u64,
        release_point: u64,
    ) -> Option<ClientRendererDmabufCommitHarness> {
        let artifacts = commit_dmabuf_surface_with_sync_points_through_client_for_tests_impl(
            import_device,
            timeline_fd,
            source_dmabuf,
            acquire_point,
            release_point,
            ClientDmabufCommitOptions {
                run_renderer_commit_handler: true,
                ..ClientDmabufCommitOptions::default()
            },
            None,
        )?;

        let renderer_buffer =
            renderer_utils::with_renderer_surface_state(&artifacts.server_surface, |state| {
                state.buffer().cloned()
            })
            .flatten()
            .expect("renderer commit handler should install a renderer-managed dmabuf buffer");

        Some(ClientRendererDmabufCommitHarness {
            evidence: artifacts.evidence,
            server_surface: artifacts.server_surface,
            renderer_buffer,
            display: artifacts.display,
            server_state: artifacts.server_state,
            client_connection: artifacts.client_connection,
            event_queue: artifacts.event_queue,
            client_state: artifacts.client_state,
        })
    }

    #[allow(dead_code)]
    pub(crate) fn commit_dmabuf_surface_with_renderer_buffer_and_acquire_sync_file_through_client_for_tests(
        import_device: DrmDeviceFd,
        timeline_fd: OwnedFd,
        acquire_sync_file: OwnedFd,
        source_dmabuf: Dmabuf,
        acquire_point: u64,
        release_point: u64,
    ) -> Option<ClientRendererDmabufCommitHarness> {
        let artifacts = commit_dmabuf_surface_with_sync_points_through_client_for_tests_impl(
            import_device,
            timeline_fd,
            source_dmabuf,
            acquire_point,
            release_point,
            ClientDmabufCommitOptions {
                run_renderer_commit_handler: true,
                install_transaction_acquire_source: true,
                acquire_sync_file: Some(acquire_sync_file),
                ..ClientDmabufCommitOptions::default()
            },
            None,
        )?;

        let renderer_buffer =
            renderer_utils::with_renderer_surface_state(&artifacts.server_surface, |state| {
                state.buffer().cloned()
            })
            .flatten()
            .expect("renderer commit handler should install a renderer-managed dmabuf buffer");

        Some(ClientRendererDmabufCommitHarness {
            evidence: artifacts.evidence,
            server_surface: artifacts.server_surface,
            renderer_buffer,
            display: artifacts.display,
            server_state: artifacts.server_state,
            client_connection: artifacts.client_connection,
            event_queue: artifacts.event_queue,
            client_state: artifacts.client_state,
        })
    }

    #[allow(dead_code)]
    pub(crate) fn commit_dmabuf_surface_with_renderer_buffer_and_acquire_satisfier_through_client_for_tests<
        F,
    >(
        import_device: DrmDeviceFd,
        timeline_fd: OwnedFd,
        source_dmabuf: Dmabuf,
        acquire_point: u64,
        release_point: u64,
        mut satisfy_acquire: F,
    ) -> Option<ClientRendererDmabufCommitHarness>
    where
        F: FnMut(&DrmSyncPoint) -> io::Result<()>,
    {
        let artifacts = commit_dmabuf_surface_with_sync_points_through_client_for_tests_impl(
            import_device,
            timeline_fd,
            source_dmabuf,
            acquire_point,
            release_point,
            ClientDmabufCommitOptions {
                run_renderer_commit_handler: true,
                install_transaction_acquire_source: true,
                ..ClientDmabufCommitOptions::default()
            },
            Some(&mut satisfy_acquire),
        )?;

        let renderer_buffer =
            renderer_utils::with_renderer_surface_state(&artifacts.server_surface, |state| {
                state.buffer().cloned()
            })
            .flatten()
            .expect("renderer commit handler should install a renderer-managed dmabuf buffer");

        Some(ClientRendererDmabufCommitHarness {
            evidence: artifacts.evidence,
            server_surface: artifacts.server_surface,
            renderer_buffer,
            display: artifacts.display,
            server_state: artifacts.server_state,
            client_connection: artifacts.client_connection,
            event_queue: artifacts.event_queue,
            client_state: artifacts.client_state,
        })
    }

    #[allow(dead_code)]
    pub(crate) fn commit_dmabuf_surface_remove_and_probe_release_through_client_for_tests(
        import_device: DrmDeviceFd,
        timeline_fd: OwnedFd,
        source_dmabuf: Dmabuf,
        acquire_point: u64,
        release_point: u64,
    ) -> Option<ClientDmabufCommitEvidence> {
        commit_dmabuf_surface_with_sync_points_through_client_for_tests_impl(
            import_device,
            timeline_fd,
            source_dmabuf,
            acquire_point,
            release_point,
            ClientDmabufCommitOptions {
                remove_after_commit: true,
                ..ClientDmabufCommitOptions::default()
            },
            None,
        )
        .map(|artifacts| artifacts.evidence)
    }

    #[allow(dead_code)]
    pub(crate) fn commit_dmabuf_surface_and_probe_transaction_acquire_blocker_through_client_for_tests(
        import_device: DrmDeviceFd,
        timeline_fd: OwnedFd,
        source_dmabuf: Dmabuf,
        acquire_point: u64,
        release_point: u64,
    ) -> Option<ClientDmabufCommitEvidence> {
        commit_dmabuf_surface_with_sync_points_through_client_for_tests_impl(
            import_device,
            timeline_fd,
            source_dmabuf,
            acquire_point,
            release_point,
            ClientDmabufCommitOptions {
                install_transaction_acquire_source: true,
                ..ClientDmabufCommitOptions::default()
            },
            None,
        )
        .map(|artifacts| artifacts.evidence)
    }

    fn commit_dmabuf_surface_with_sync_points_through_client_for_tests_impl(
        import_device: DrmDeviceFd,
        timeline_fd: OwnedFd,
        source_dmabuf: Dmabuf,
        acquire_point: u64,
        release_point: u64,
        options: ClientDmabufCommitOptions,
        mut acquire_point_satisfier: Option<&mut dyn FnMut(&DrmSyncPoint) -> io::Result<()>>,
    ) -> Option<ClientDmabufCommitArtifacts> {
        let mut display = match Display::<SurfacePointProtocolServerState>::new() {
            Ok(display) => display,
            Err(InitError::NoWaylandLib) => return None,
            Err(err) => panic!("failed to create test Wayland display: {err}"),
        };
        let mut display_handle = display.handle();
        let compositor_state =
            compositor::CompositorState::new::<SurfacePointProtocolServerState>(&display_handle);
        let syncobj_state =
            DrmSyncobjState::new::<SurfacePointProtocolServerState>(&display_handle, import_device);
        let dmabuf_format = source_dmabuf.format();
        let dmabuf_size = source_dmabuf.size();
        let mut dmabuf_state = DmabufState::new();
        dmabuf_state.create_global::<SurfacePointProtocolServerState>(&display_handle, [dmabuf_format]);
        let mut server_state = SurfacePointProtocolServerState {
            compositor_state,
            syncobj_state: Some(syncobj_state),
            dmabuf_state,
            expected_dmabuf: Some(source_dmabuf.clone()),
            last_imported_dmabuf_syncable: false,
            last_imported_dmabuf_matches_expected: false,
            install_acquire_sources: options.install_transaction_acquire_source,
            installed_acquire_points: Vec::new(),
            acquire_sources: Vec::new(),
            committed_surfaces: 0,
            surfaces: Vec::new(),
            run_renderer_commit_handler: options.run_renderer_commit_handler,
        };
        let (client_side, server_side) = UnixStream::pair().unwrap();
        let _server_client = display_handle
            .insert_client(server_side, Arc::new(SurfacePointProtocolClientData::default()))
            .expect("insert test client");

        let client_connection = Connection::from_socket(client_side).expect("connect test client socket");
        let mut event_queue = client_connection.new_event_queue();
        let qh = event_queue.handle();
        let mut client_state = SurfacePointProtocolClientState::default();

        client_connection.display().get_registry(&qh, ());
        client_connection.flush().expect("flush get_registry");
        pump_surface_point_protocol_server(&mut display, &mut server_state);
        event_queue
            .blocking_dispatch(&mut client_state)
            .expect("dispatch registry globals");

        let compositor = client_state
            .compositor
            .as_ref()
            .expect("test compositor global should be advertised");
        let syncobj_manager = client_state
            .syncobj_manager
            .as_ref()
            .expect("test syncobj global should be advertised");
        let dmabuf_global = client_state
            .dmabuf
            .as_ref()
            .expect("test dmabuf global should be advertised");
        let surface = compositor.create_surface(&qh, ());
        let syncobj_surface = syncobj_manager.get_surface(&surface, &qh, ());
        let timeline = syncobj_manager.import_timeline(timeline_fd.as_fd(), &qh, ());
        let params = dmabuf_global.create_params(&qh, ());
        let modifier: u64 = dmabuf_format.modifier.into();
        for plane in &source_dmabuf.0.planes {
            params.add(
                plane.fd.as_fd(),
                plane.plane_idx,
                plane.offset,
                plane.stride,
                (modifier >> 32) as u32,
                modifier as u32,
            );
        }
        let buffer = params.create_immed(
            dmabuf_size.w,
            dmabuf_size.h,
            dmabuf_format.code as u32,
            zwp_linux_buffer_params_v1::Flags::empty(),
            &qh,
            (),
        );
        let (acquire_hi, acquire_lo) = (((acquire_point >> 32) as u32), acquire_point as u32);
        let (release_hi, release_lo) = (((release_point >> 32) as u32), release_point as u32);
        syncobj_surface.set_acquire_point(&timeline, acquire_hi, acquire_lo);
        syncobj_surface.set_release_point(&timeline, release_hi, release_lo);
        if !options.install_transaction_acquire_source {
            client_connection
                .flush()
                .expect("flush dmabuf setup and syncobj surface point requests");
            pump_surface_point_protocol_server(&mut display, &mut server_state);
            let server_surface = server_state
                .surfaces
                .first()
                .expect("client create_surface should reach server state");
            let pending_acquire = with_states(server_surface, |states| {
                let mut cached = states.cached_state.get::<DrmSyncobjCachedState>();
                cached
                    .pending()
                    .acquire_point
                    .as_ref()
                    .expect("set_acquire_point should stage a pending acquire point")
                    .clone()
            });
            if let Some(acquire_sync_file) = options.acquire_sync_file.as_ref() {
                pending_acquire
                    .import_sync_file(acquire_sync_file.as_fd())
                    .expect("import acquire sync-file into pending explicit-sync commit point");
            } else {
                pending_acquire
                    .signal()
                    .expect("signal acquire point for non-blocking explicit-sync commit helper");
            }
        }
        surface.attach(Some(&buffer), 0, 0);
        surface.commit();
        client_state.surface = Some(surface);
        client_state.syncobj_surface = Some(syncobj_surface);
        client_state.timeline = Some(timeline);
        client_state.params = Some(params);
        client_state.buffer = Some(buffer);
        client_connection
            .flush()
            .expect("flush dmabuf surface commit request");
        pump_surface_point_protocol_server(&mut display, &mut server_state);

        let (transaction_acquire_source_installed, transaction_pending_before_acquire_signal) =
            if options.install_transaction_acquire_source {
                assert_eq!(
                    server_state.installed_acquire_points.len(),
                    1,
                    "valid explicit-sync commit should install exactly one acquire point source"
                );
                assert_eq!(
                    server_state.acquire_sources.len(),
                    1,
                    "valid explicit-sync commit should store exactly one acquire event source"
                );
                let server_surface = server_state
                    .surfaces
                    .first()
                    .expect("client create_surface should reach server state");
                let current_has_dmabuf_before_signal = with_states(server_surface, |states| {
                    let mut attributes = states.cached_state.get::<SurfaceAttributes>();
                    attributes
                        .current()
                        .buffer
                        .as_ref()
                        .and_then(|assignment| match assignment {
                            BufferAssignment::NewBuffer(buffer) => get_dmabuf(buffer).ok(),
                            BufferAssignment::Removed => None,
                        })
                        .is_some()
                });
                (
                    Some(true),
                    Some(server_state.committed_surfaces == 0 && !current_has_dmabuf_before_signal),
                )
            } else {
                (None, None)
            };

        if options.install_transaction_acquire_source {
            let acquire = server_state
                .installed_acquire_points
                .first()
                .expect("valid explicit-sync commit should install an acquire point")
                .clone();
            let source = server_state
                .acquire_sources
                .pop()
                .expect("valid explicit-sync commit should install an acquire event source");
            let server_surface = server_state
                .surfaces
                .first()
                .cloned()
                .expect("client create_surface should reach server state");
            let server_client = server_surface
                .client()
                .expect("test surface should remain attached to a client");
            let display_handle = display.handle();
            let mut event_loop = calloop::EventLoop::<SurfacePointProtocolServerState>::try_new()
                .expect("create acquire blocker test event loop");
            event_loop
                .handle()
                .insert_source(source, move |(), (), state| {
                    server_client
                        .get_data::<SurfacePointProtocolClientData>()
                        .expect("test client should carry compositor state")
                        .compositor_state
                        .blocker_cleared(state, &display_handle);
                    Ok(())
                })
                .expect("insert acquire blocker event source");
            if let Some(satisfy_acquire) = acquire_point_satisfier.as_mut() {
                satisfy_acquire(&acquire).expect("satisfy transaction explicit-sync acquire point");
            } else if let Some(acquire_sync_file) = options.acquire_sync_file.as_ref() {
                acquire
                    .import_sync_file(acquire_sync_file.as_fd())
                    .expect("import acquire sync-file into transaction explicit-sync commit point");
            } else {
                acquire.signal().expect("signal transaction acquire point");
            }
            event_loop
                .dispatch(Duration::from_millis(100), &mut server_state)
                .expect("dispatch transaction acquire event source");
        }

        let server_surface = server_state
            .surfaces
            .first()
            .cloned()
            .expect("client create_surface should reach server state");
        let (
            current_has_dmabuf,
            staged_acquire_point,
            staged_release_point,
            acquire_release_same_timeline,
            current_release_sync_point,
        ) = with_states(&server_surface, |states| {
            let mut attributes = states.cached_state.get::<SurfaceAttributes>();
            let current_buffer =
                attributes
                    .current()
                    .buffer
                    .as_ref()
                    .and_then(|assignment| match assignment {
                        BufferAssignment::NewBuffer(buffer) => Some(buffer),
                        BufferAssignment::Removed => None,
                    });
            let current_has_dmabuf = current_buffer
                .map(|buffer| get_dmabuf(buffer).is_ok())
                .unwrap_or(false);

            let mut cached = states.cached_state.get::<DrmSyncobjCachedState>();
            let current = cached.current();
            let acquire = current.acquire_point.as_ref();
            let release = current.release_point.as_ref();
            (
                current_has_dmabuf,
                acquire.map(|point| point.point),
                release.map(|point| point.point),
                acquire
                    .zip(release)
                    .map(|(acquire, release)| acquire.timeline == release.timeline)
                    .unwrap_or(false),
                release.cloned(),
            )
        });
        let (
            removal_release_point_signaled_before_commit,
            removal_release_point_signaled_after_commit,
            removal_release_point_signaled_at_buffer_release_event,
            removal_buffer_release_events,
            current_has_dmabuf_after_removal,
        ) = if options.remove_after_commit {
            let release_sync_point = current_release_sync_point
                .as_ref()
                .expect("valid explicit-sync dmabuf commit should promote a release point")
                .clone();
            let signaled_before = sync_point_signaled_for_tests(&release_sync_point);
            let surface = client_state
                .surface
                .as_ref()
                .expect("test surface should remain live");
            surface.attach(None, 0, 0);
            surface.commit();
            client_connection
                .flush()
                .expect("flush dmabuf surface removal commit request");
            pump_surface_point_protocol_server(&mut display, &mut server_state);
            client_state.expected_release_point_for_release_event = Some(release_sync_point.clone());
            dispatch_surface_point_client_events(&mut event_queue, &client_connection, &mut client_state);
            let signaled_after = sync_point_signaled_for_tests(&release_sync_point);
            let current_has_dmabuf_after_removal = with_states(&server_surface, |states| {
                let mut attributes = states.cached_state.get::<SurfaceAttributes>();
                attributes
                    .current()
                    .buffer
                    .as_ref()
                    .and_then(|assignment| match assignment {
                        BufferAssignment::NewBuffer(buffer) => get_dmabuf(buffer).ok(),
                        BufferAssignment::Removed => None,
                    })
                    .is_some()
            });
            (
                Some(signaled_before),
                Some(signaled_after),
                client_state.release_point_signaled_at_buffer_release_event,
                Some(client_state.buffer_release_events),
                Some(current_has_dmabuf_after_removal),
            )
        } else {
            (None, None, None, None, None)
        };
        let known_timeline_count = server_state
            .syncobj_state
            .as_ref()
            .expect("test syncobj state should remain installed")
            .known_timelines
            .iter()
            .filter_map(Weak::upgrade)
            .count();
        let renderer_current_has_dmabuf_after_acquire = options.run_renderer_commit_handler
            && renderer_utils::with_renderer_surface_state(&server_surface, |state| {
                state
                    .buffer()
                    .map(|buffer| get_dmabuf(buffer).is_ok())
                    .unwrap_or(false)
            })
            .unwrap_or(false);
        let transaction_released_after_acquire_signal = options.install_transaction_acquire_source.then_some(
            server_state.committed_surfaces == 1
                && (current_has_dmabuf || renderer_current_has_dmabuf_after_acquire),
        );

        let evidence = ClientDmabufCommitEvidence {
            known_timeline_count,
            imported_dmabuf_syncable: server_state.last_imported_dmabuf_syncable,
            imported_dmabuf_matches_expected: server_state.last_imported_dmabuf_matches_expected,
            current_has_dmabuf,
            acquire_point: staged_acquire_point,
            release_point: staged_release_point,
            acquire_release_same_timeline,
            transaction_acquire_source_installed,
            transaction_pending_before_acquire_signal,
            transaction_released_after_acquire_signal,
            removal_release_point_signaled_before_commit,
            removal_release_point_signaled_after_commit,
            removal_release_point_signaled_at_buffer_release_event,
            removal_buffer_release_events,
            current_has_dmabuf_after_removal,
        };

        Some(ClientDmabufCommitArtifacts {
            evidence,
            display,
            server_state,
            client_connection,
            event_queue,
            client_state,
            server_surface,
        })
    }

    /// Install a server-side DRM syncobj surface object for a focused test surface.
    ///
    /// This mirrors the server-side setup performed by `wp_linux_drm_syncobj_manager_v1.get_surface`:
    /// it creates the protocol resource, installs the commit/destruction hooks, and stores the resource
    /// in the surface data map. It intentionally still bypasses client socket dispatch; tests using it
    /// must stage pending sync points themselves or drive the request handlers separately.
    #[allow(dead_code)]
    pub(crate) fn install_surface_for_tests<D>(
        handle: &DisplayHandle,
        surface: &wayland_server::protocol::wl_surface::WlSurface,
    ) -> WpLinuxDrmSyncobjSurfaceV1
    where
        D: ServerDispatch<WpLinuxDrmSyncobjSurfaceV1, DrmSyncobjSurfaceData> + DrmSyncobjHandler + 'static,
    {
        let already_exists = with_states(surface, |states| {
            states
                .data_map
                .get::<RefCell<Option<WpLinuxDrmSyncobjSurfaceV1>>>()
                .map(|v| v.borrow().is_some())
                .unwrap_or(false)
        });
        assert!(
            !already_exists,
            "test surface already has a DRM syncobj surface object"
        );

        ensure_surface_hooks::<D>(surface);
        let client = surface
            .client()
            .expect("test WlSurface should still be attached to a live client");
        let syncobj_surface = client
            .create_resource::<WpLinuxDrmSyncobjSurfaceV1, DrmSyncobjSurfaceData, D>(
                handle,
                1,
                DrmSyncobjSurfaceData {
                    surface: surface.downgrade(),
                },
            )
            .expect("create test DRM syncobj surface resource");
        with_states(surface, |states| {
            let syncobj_surface_cell = states
                .data_map
                .get_or_insert(|| RefCell::new(None::<WpLinuxDrmSyncobjSurfaceV1>));
            *syncobj_surface_cell.borrow_mut() = Some(syncobj_surface.clone());
        });

        syncobj_surface
    }

    pub(crate) fn destroy_surface_for_tests(surface: &wayland_server::protocol::wl_surface::WlSurface) {
        destroy_syncobj_surface_state(surface);
    }

    fn timeline_resource_for_tests<D>(
        client: &Client,
        handle: &DisplayHandle,
        point: &DrmSyncPoint,
    ) -> WpLinuxDrmSyncobjTimelineV1
    where
        D: ServerDispatch<WpLinuxDrmSyncobjTimelineV1, DrmSyncobjTimelineData> + 'static,
    {
        client
            .create_resource::<WpLinuxDrmSyncobjTimelineV1, DrmSyncobjTimelineData, D>(
                handle,
                1,
                DrmSyncobjTimelineData {
                    timeline: point.timeline.clone(),
                },
            )
            .expect("create test DRM syncobj timeline resource")
    }

    fn point_halves(point: &DrmSyncPoint) -> (u32, u32) {
        ((point.point >> 32) as u32, point.point as u32)
    }

    /// Stage acquire/release points through DRM syncobj timeline resources for a focused test surface.
    ///
    /// This uses the same pending-point construction helper as the protocol request handlers after
    /// creating server-side timeline resources for the provided [`DrmSyncPoint`]s. It still bypasses
    /// client socket dispatch and `wp_linux_drm_syncobj_manager_v1.import_timeline`; callers provide
    /// already-created timeline points.
    #[allow(dead_code)]
    pub(crate) fn set_surface_points_for_tests<D>(
        handle: &DisplayHandle,
        surface: &wayland_server::protocol::wl_surface::WlSurface,
        acquire_point: &DrmSyncPoint,
        release_point: &DrmSyncPoint,
    ) where
        D: ServerDispatch<WpLinuxDrmSyncobjTimelineV1, DrmSyncobjTimelineData> + 'static,
    {
        let has_syncobj_surface = with_states(surface, |states| {
            states
                .data_map
                .get::<RefCell<Option<WpLinuxDrmSyncobjSurfaceV1>>>()
                .map(|v| v.borrow().is_some())
                .unwrap_or(false)
        });
        assert!(
            has_syncobj_surface,
            "test surface should have a DRM syncobj surface object before staging points"
        );
        let client = surface
            .client()
            .expect("test WlSurface should still be attached to a live client");
        let acquire_timeline = timeline_resource_for_tests::<D>(&client, handle, acquire_point);
        let release_timeline = timeline_resource_for_tests::<D>(&client, handle, release_point);
        let (acquire_hi, acquire_lo) = point_halves(acquire_point);
        let (release_hi, release_lo) = point_halves(release_point);

        set_pending_sync_point_from_timeline_resource(
            surface,
            &acquire_timeline,
            acquire_hi,
            acquire_lo,
            PendingSyncPointKind::Acquire,
        );
        set_pending_sync_point_from_timeline_resource(
            surface,
            &release_timeline,
            release_hi,
            release_lo,
            PendingSyncPointKind::Release,
        );
    }
}

/// Data attached to wp_linux_drm_syncobj_surface_v1 objects
#[derive(Debug)]
pub struct DrmSyncobjSurfaceData {
    surface: WlWeak<WlSurface>,
}

impl<D> Dispatch2<WpLinuxDrmSyncobjSurfaceV1, D> for DrmSyncobjSurfaceData
where
    D: DrmSyncobjHandler,
{
    fn request(
        &self,
        _state: &mut D,
        _client: &Client,
        resource: &WpLinuxDrmSyncobjSurfaceV1,
        request: wp_linux_drm_syncobj_surface_v1::Request,
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, D>,
    ) {
        match request {
            wp_linux_drm_syncobj_surface_v1::Request::Destroy => {
                if let Ok(surface) = self.surface.upgrade() {
                    destroy_syncobj_surface_state(&surface);
                }
            }
            wp_linux_drm_syncobj_surface_v1::Request::SetAcquirePoint {
                timeline,
                point_hi,
                point_lo,
            } => {
                let Ok(surface) = self.surface.upgrade() else {
                    resource.post_error(
                        wp_linux_drm_syncobj_surface_v1::Error::NoSurface,
                        "Set acquire point for destroyed surface.",
                    );
                    return;
                };

                set_pending_sync_point_from_timeline_resource(
                    &surface,
                    &timeline,
                    point_hi,
                    point_lo,
                    PendingSyncPointKind::Acquire,
                );
            }
            wp_linux_drm_syncobj_surface_v1::Request::SetReleasePoint {
                timeline,
                point_hi,
                point_lo,
            } => {
                let Ok(surface) = self.surface.upgrade() else {
                    resource.post_error(
                        wp_linux_drm_syncobj_surface_v1::Error::NoSurface,
                        "Set release point for destroyed surface.",
                    );
                    return;
                };

                set_pending_sync_point_from_timeline_resource(
                    &surface,
                    &timeline,
                    point_hi,
                    point_lo,
                    PendingSyncPointKind::Release,
                );
            }
            _ => unreachable!(),
        }
    }
}

/// Data attached to wp_linux_drm_syncobj_timeline_v1 objects
#[derive(Debug)]
pub struct DrmSyncobjTimelineData {
    timeline: DrmTimeline,
}

impl<D: DrmSyncobjHandler> Dispatch2<WpLinuxDrmSyncobjTimelineV1, D> for DrmSyncobjTimelineData {
    fn request(
        &self,
        _state: &mut D,
        _client: &Client,
        _resource: &WpLinuxDrmSyncobjTimelineV1,
        request: wp_linux_drm_syncobj_timeline_v1::Request,
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, D>,
    ) {
        match request {
            wp_linux_drm_syncobj_timeline_v1::Request::Destroy => {}
            _ => unreachable!(),
        }
    }

    fn destroyed(
        &self,
        state: &mut D,
        _client: wayland_server::backend::ClientId,
        _resource: &WpLinuxDrmSyncobjTimelineV1,
    ) {
        if let Some(state) = state.drm_syncobj_state() {
            state
                .known_timelines
                .retain(|t| t.upgrade().is_some_and(|t| !Arc::ptr_eq(&t, &self.timeline.0)))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{os::fd::AsFd, os::unix::net::UnixStream, sync::Arc};

    use wayland_client::{
        Connection, Dispatch, QueueHandle, delegate_noop,
        protocol::{wl_compositor, wl_registry, wl_surface},
    };
    use wayland_protocols::wp::linux_drm_syncobj::v1::client::{
        wp_linux_drm_syncobj_manager_v1, wp_linux_drm_syncobj_surface_v1, wp_linux_drm_syncobj_timeline_v1,
    };
    use wayland_server::{
        Display, DisplayHandle, Resource,
        backend::{ClientData, ClientId, DisconnectReason, InitError},
        protocol::wl_buffer as server_wl_buffer,
    };

    use super::*;
    use crate::backend::allocator::{
        Fourcc, Modifier,
        dmabuf::{Dmabuf, DmabufFlags},
    };
    use crate::utils::Serial;
    use crate::wayland::buffer::BufferHandler;
    use crate::wayland::compositor;

    #[derive(Default)]
    struct ProtocolClientData {
        compositor_state: compositor::CompositorClientState,
    }

    impl ClientData for ProtocolClientData {
        fn initialized(&self, _client_id: ClientId) {}

        fn disconnected(&self, _client_id: ClientId, _reason: DisconnectReason) {}
    }

    struct ProtocolServerState {
        compositor_state: compositor::CompositorState,
        syncobj_state: Option<DrmSyncobjState>,
        surfaces: Vec<wayland_server::protocol::wl_surface::WlSurface>,
    }

    impl compositor::CompositorHandler for ProtocolServerState {
        fn compositor_state(&mut self) -> &mut compositor::CompositorState {
            &mut self.compositor_state
        }

        fn client_compositor_state<'a>(
            &self,
            client: &'a wayland_server::Client,
        ) -> &'a compositor::CompositorClientState {
            &client
                .get_data::<ProtocolClientData>()
                .expect("test client should carry compositor state")
                .compositor_state
        }

        fn new_surface(&mut self, surface: &wayland_server::protocol::wl_surface::WlSurface) {
            self.surfaces.push(surface.clone());
        }

        fn commit(&mut self, _surface: &wayland_server::protocol::wl_surface::WlSurface) {}
    }

    impl DrmSyncobjHandler for ProtocolServerState {
        fn drm_syncobj_state(&mut self) -> Option<&mut DrmSyncobjState> {
            self.syncobj_state.as_mut()
        }
    }

    impl BufferHandler for ProtocolServerState {
        fn buffer_destroyed(&mut self, _buffer: &server_wl_buffer::WlBuffer) {}
    }

    struct ProtocolBufferData;

    impl crate::wayland::Dispatch2<server_wl_buffer::WlBuffer, ProtocolServerState> for ProtocolBufferData {
        fn request(
            &self,
            _state: &mut ProtocolServerState,
            _client: &wayland_server::Client,
            _resource: &server_wl_buffer::WlBuffer,
            _request: server_wl_buffer::Request,
            _dhandle: &DisplayHandle,
            _data_init: &mut wayland_server::DataInit<'_, ProtocolServerState>,
        ) {
        }
    }

    impl AsMut<compositor::CompositorState> for ProtocolServerState {
        fn as_mut(&mut self) -> &mut compositor::CompositorState {
            &mut self.compositor_state
        }
    }

    crate::delegate_dispatch2!(ProtocolServerState);

    #[derive(Default)]
    struct ProtocolClientState {
        compositor: Option<wl_compositor::WlCompositor>,
        syncobj_manager: Option<wp_linux_drm_syncobj_manager_v1::WpLinuxDrmSyncobjManagerV1>,
        surface: Option<wl_surface::WlSurface>,
        syncobj_surface: Option<wp_linux_drm_syncobj_surface_v1::WpLinuxDrmSyncobjSurfaceV1>,
        timeline: Option<wp_linux_drm_syncobj_timeline_v1::WpLinuxDrmSyncobjTimelineV1>,
    }

    impl Dispatch<wl_registry::WlRegistry, ()> for ProtocolClientState {
        fn event(
            state: &mut Self,
            registry: &wl_registry::WlRegistry,
            event: wl_registry::Event,
            _: &(),
            _: &Connection,
            qh: &QueueHandle<Self>,
        ) {
            if let wl_registry::Event::Global { name, interface, .. } = event {
                match interface.as_str() {
                    "wl_compositor" => {
                        state.compositor =
                            Some(registry.bind::<wl_compositor::WlCompositor, _, _>(name, 1, qh, ()));
                    }
                    "wp_linux_drm_syncobj_manager_v1" => {
                        state.syncobj_manager = Some(
                            registry
                                .bind::<wp_linux_drm_syncobj_manager_v1::WpLinuxDrmSyncobjManagerV1, _, _>(
                                    name,
                                    1,
                                    qh,
                                    (),
                                ),
                        );
                    }
                    _ => {}
                }
            }
        }
    }

    delegate_noop!(ProtocolClientState: ignore wl_compositor::WlCompositor);
    delegate_noop!(ProtocolClientState: ignore wl_surface::WlSurface);
    delegate_noop!(ProtocolClientState: ignore wp_linux_drm_syncobj_manager_v1::WpLinuxDrmSyncobjManagerV1);
    delegate_noop!(ProtocolClientState: ignore wp_linux_drm_syncobj_surface_v1::WpLinuxDrmSyncobjSurfaceV1);
    delegate_noop!(ProtocolClientState: ignore wp_linux_drm_syncobj_timeline_v1::WpLinuxDrmSyncobjTimelineV1);

    fn pump_server(display: &mut Display<ProtocolServerState>, state: &mut ProtocolServerState) {
        display
            .dispatch_clients(state)
            .expect("dispatch test client requests");
        display.flush_clients().expect("flush test server events");
    }

    fn syncobj_surface_marker_is_some(surface: &wayland_server::protocol::wl_surface::WlSurface) -> bool {
        syncobj_surface_marker_protocol_id(surface).is_some()
    }

    fn syncobj_surface_marker_protocol_id(
        surface: &wayland_server::protocol::wl_surface::WlSurface,
    ) -> Option<u32> {
        compositor::with_states(surface, |states| {
            states
                .data_map
                .get::<RefCell<Option<WpLinuxDrmSyncobjSurfaceV1>>>()
                .and_then(|resource| {
                    resource
                        .borrow()
                        .as_ref()
                        .map(|resource| resource.id().protocol_id())
                })
        })
    }

    fn read_protocol_error_after_server_pump(
        event_queue: &mut wayland_client::EventQueue<ProtocolClientState>,
        client_connection: &Connection,
        client_state: &mut ProtocolClientState,
    ) -> (String, u32) {
        let protocol_error = if let Some(guard) = event_queue.prepare_read() {
            let _ = guard.read();
            let _ = event_queue.dispatch_pending(client_state);
            client_connection.protocol_error()
        } else {
            let _ = event_queue.dispatch_pending(client_state);
            client_connection.protocol_error()
        }
        .expect("client request should disconnect with a protocol error");
        (protocol_error.object_interface, protocol_error.code)
    }

    fn focused_commit_state() -> Option<(
        Display<ProtocolServerState>,
        DisplayHandle,
        ProtocolServerState,
        UnixStream,
        wayland_server::Client,
        wayland_server::protocol::wl_surface::WlSurface,
    )> {
        let display = match Display::<ProtocolServerState>::new() {
            Ok(display) => display,
            Err(InitError::NoWaylandLib) => return None,
            Err(err) => panic!("failed to create test Wayland display: {err}"),
        };
        let mut display_handle = display.handle();
        let compositor_state = compositor::CompositorState::new::<ProtocolServerState>(&display_handle);
        let syncobj_state =
            DrmSyncobjState::new_without_import_device_for_tests::<ProtocolServerState>(&display_handle);
        let server_state = ProtocolServerState {
            compositor_state,
            syncobj_state: Some(syncobj_state),
            surfaces: Vec::new(),
        };
        let (client_side, server_side) = UnixStream::pair().unwrap();
        let server_client = display_handle
            .insert_client(server_side, Arc::new(ProtocolClientData::default()))
            .expect("insert test client");
        let surface =
            compositor::test_utils::create_surface::<ProtocolServerState>(&server_client, &display_handle);
        Some((
            display,
            display_handle,
            server_state,
            client_side,
            server_client,
            surface,
        ))
    }

    fn dmabuf_for_invalid_commit_tests() -> Dmabuf {
        // This fd is intentionally inert: these tests only need `get_dmabuf()` to identify the
        // buffer as dmabuf-backed so the pre-commit guard reaches the targeted syncobj branch.
        // No renderer or kernel dmabuf importer consumes this buffer.
        let plane_fd = rustix::event::eventfd(0, rustix::event::EventfdFlags::CLOEXEC)
            .expect("create inert test dmabuf fd");
        let mut builder = Dmabuf::builder((1, 1), Fourcc::Abgr8888, Modifier::Invalid, DmabufFlags::empty());
        assert!(builder.add_plane(plane_fd, 0, 4));
        builder.build().expect("test dmabuf should have one plane")
    }

    fn protocol_dmabuf_buffer(
        client: &wayland_server::Client,
        display_handle: &DisplayHandle,
    ) -> server_wl_buffer::WlBuffer {
        client
            .create_resource::<server_wl_buffer::WlBuffer, Dmabuf, ProtocolServerState>(
                display_handle,
                1,
                dmabuf_for_invalid_commit_tests(),
            )
            .expect("create test dmabuf wl_buffer")
    }

    fn protocol_buffer(
        client: &wayland_server::Client,
        display_handle: &DisplayHandle,
    ) -> server_wl_buffer::WlBuffer {
        client
            .create_resource::<server_wl_buffer::WlBuffer, ProtocolBufferData, ProtocolServerState>(
                display_handle,
                1,
                ProtocolBufferData,
            )
            .expect("create test wl_buffer")
    }

    fn stage_pending_sync_points(
        surface: &wayland_server::protocol::wl_surface::WlSurface,
        acquire_point: Option<DrmSyncPoint>,
        release_point: Option<DrmSyncPoint>,
    ) {
        compositor::with_states(surface, |states| {
            let mut cached = states.cached_state.get::<DrmSyncobjCachedState>();
            let pending = cached.pending();
            pending.acquire_point = acquire_point;
            pending.release_point = release_point;
        });
    }

    fn current_sync_points_are_empty(surface: &wayland_server::protocol::wl_surface::WlSurface) -> bool {
        compositor::with_states(surface, |states| {
            let mut cached = states.cached_state.get::<DrmSyncobjCachedState>();
            let current = cached.current();
            current.acquire_point.is_none() && current.release_point.is_none()
        })
    }

    fn current_buffer_is_some(surface: &wayland_server::protocol::wl_surface::WlSurface) -> bool {
        compositor::with_states(surface, |states| {
            let mut attributes = states.cached_state.get::<SurfaceAttributes>();
            matches!(attributes.current().buffer, Some(BufferAssignment::NewBuffer(_)))
        })
    }

    fn current_buffer_is_none(surface: &wayland_server::protocol::wl_surface::WlSurface) -> bool {
        compositor::with_states(surface, |states| {
            let mut attributes = states.cached_state.get::<SurfaceAttributes>();
            attributes.current().buffer.is_none()
        })
    }

    fn current_buffer_has_no_new_buffer(surface: &wayland_server::protocol::wl_surface::WlSurface) -> bool {
        compositor::with_states(surface, |states| {
            let mut attributes = states.cached_state.get::<SurfaceAttributes>();
            !matches!(attributes.current().buffer, Some(BufferAssignment::NewBuffer(_)))
        })
    }

    fn queue_explicit_buffer_state(
        surface: &wayland_server::protocol::wl_surface::WlSurface,
        display_handle: &DisplayHandle,
        serial: u32,
        buffer: wayland_server::protocol::wl_buffer::WlBuffer,
        acquire_point: DrmSyncPoint,
        release_point: DrmSyncPoint,
    ) {
        compositor::with_states(surface, |states| {
            {
                let mut attributes = states.cached_state.get::<SurfaceAttributes>();
                attributes.pending().buffer = Some(BufferAssignment::NewBuffer(buffer));
            }
            {
                let mut syncobj = states.cached_state.get::<DrmSyncobjCachedState>();
                let pending = syncobj.pending();
                pending.acquire_point = Some(acquire_point);
                pending.release_point = Some(release_point);
            }
        });
        compositor::test_utils::commit_pending_state_with_serial(surface, display_handle, serial);
    }

    fn queue_buffer_assignment_after_pre_commit(
        server_state: &mut ProtocolServerState,
        surface: &wayland_server::protocol::wl_surface::WlSurface,
        display_handle: &DisplayHandle,
        serial: u32,
        buffer: Option<wayland_server::protocol::wl_buffer::WlBuffer>,
    ) {
        compositor::with_states(surface, |states| {
            let mut attributes = states.cached_state.get::<SurfaceAttributes>();
            attributes.pending().buffer = Some(match buffer {
                Some(buffer) => BufferAssignment::NewBuffer(buffer),
                None => BufferAssignment::Removed,
            });
        });
        commit_hook(server_state, display_handle, surface);
        compositor::test_utils::commit_pending_state_with_serial(surface, display_handle, serial);
    }

    #[test]
    fn invalid_commit_new_buffer_without_acquire_discards_buffer_and_sync_state() {
        let Some((_display, display_handle, mut server_state, _client_side, server_client, surface)) =
            focused_commit_state()
        else {
            return;
        };
        test_utils::install_surface_for_tests::<ProtocolServerState>(&display_handle, &surface);
        let buffer = protocol_dmabuf_buffer(&server_client, &display_handle);
        let release_point = DrmSyncPoint::invalid_for_tests(7).expect("create test release point");
        stage_pending_sync_points(&surface, None, Some(release_point));

        compositor::test_utils::commit_buffer_assignment(
            &mut server_state,
            &display_handle,
            &surface,
            Some(buffer),
        );

        assert!(current_sync_points_are_empty(&surface));
        assert!(current_buffer_is_none(&surface));
    }

    #[test]
    fn invalid_commit_new_buffer_without_release_discards_buffer_and_sync_state() {
        let Some((_display, display_handle, mut server_state, _client_side, server_client, surface)) =
            focused_commit_state()
        else {
            return;
        };
        test_utils::install_surface_for_tests::<ProtocolServerState>(&display_handle, &surface);
        let buffer = protocol_dmabuf_buffer(&server_client, &display_handle);
        let acquire_point = DrmSyncPoint::invalid_for_tests(7).expect("create test acquire point");
        stage_pending_sync_points(&surface, Some(acquire_point), None);

        compositor::test_utils::commit_buffer_assignment(
            &mut server_state,
            &display_handle,
            &surface,
            Some(buffer),
        );

        assert!(current_sync_points_are_empty(&surface));
        assert!(current_buffer_is_none(&surface));
    }

    #[test]
    fn invalid_commit_conflicting_points_discards_buffer_and_sync_state() {
        let Some((_display, display_handle, mut server_state, _client_side, server_client, surface)) =
            focused_commit_state()
        else {
            return;
        };
        test_utils::install_surface_for_tests::<ProtocolServerState>(&display_handle, &surface);
        let buffer = protocol_dmabuf_buffer(&server_client, &display_handle);
        let (acquire_point, release_point) =
            DrmSyncPoint::invalid_timeline_pair_for_tests(9, 9).expect("create conflicting points");
        stage_pending_sync_points(&surface, Some(acquire_point), Some(release_point));

        compositor::test_utils::commit_buffer_assignment(
            &mut server_state,
            &display_handle,
            &surface,
            Some(buffer),
        );

        assert!(current_sync_points_are_empty(&surface));
        assert!(current_buffer_is_none(&surface));
    }

    #[test]
    fn invalid_commit_unsupported_buffer_discards_buffer_and_sync_state() {
        let Some((_display, display_handle, mut server_state, _client_side, server_client, surface)) =
            focused_commit_state()
        else {
            return;
        };
        test_utils::install_surface_for_tests::<ProtocolServerState>(&display_handle, &surface);
        let buffer = protocol_buffer(&server_client, &display_handle);
        let (acquire_point, release_point) =
            DrmSyncPoint::invalid_timeline_pair_for_tests(9, 10).expect("create ordered points");
        stage_pending_sync_points(&surface, Some(acquire_point), Some(release_point));

        compositor::test_utils::commit_buffer_assignment(
            &mut server_state,
            &display_handle,
            &surface,
            Some(buffer),
        );

        assert!(current_sync_points_are_empty(&surface));
        assert!(current_buffer_is_none(&surface));
    }

    #[test]
    fn commit_unwaitable_acquire_without_source_discards_buffer_and_sync_state() {
        let Some((_display, display_handle, mut server_state, _client_side, server_client, surface)) =
            focused_commit_state()
        else {
            return;
        };
        test_utils::install_surface_for_tests::<ProtocolServerState>(&display_handle, &surface);
        let buffer = protocol_dmabuf_buffer(&server_client, &display_handle);
        let (acquire_point, release_point) =
            DrmSyncPoint::invalid_timeline_pair_for_tests(31, 32).expect("create invalid test sync points");
        stage_pending_sync_points(&surface, Some(acquire_point), Some(release_point));

        compositor::test_utils::commit_buffer_assignment(
            &mut server_state,
            &display_handle,
            &surface,
            Some(buffer),
        );

        assert!(current_sync_points_are_empty(&surface));
        assert!(current_buffer_is_none(&surface));
    }

    #[test]
    fn buffer_removal_clears_current_sync_points() {
        let Some((_display, display_handle, mut server_state, _client_side, server_client, surface)) =
            focused_commit_state()
        else {
            return;
        };
        test_utils::install_surface_for_tests::<ProtocolServerState>(&display_handle, &surface);
        let buffer = protocol_dmabuf_buffer(&server_client, &display_handle);
        let (acquire_point, release_point) =
            DrmSyncPoint::invalid_timeline_pair_for_tests(33, 34).expect("create current test sync points");
        compositor::with_states(&surface, |states| {
            let mut attributes = states.cached_state.get::<SurfaceAttributes>();
            attributes.current().buffer = Some(BufferAssignment::NewBuffer(buffer));
            let mut syncobj = states.cached_state.get::<DrmSyncobjCachedState>();
            let current = syncobj.current();
            current.acquire_point = Some(acquire_point);
            current.release_point = Some(release_point);
        });

        compositor::test_utils::commit_buffer_assignment(&mut server_state, &display_handle, &surface, None);

        assert!(current_sync_points_are_empty(&surface));
        assert!(current_buffer_has_no_new_buffer(&surface));
    }

    #[test]
    fn buffer_removal_after_syncobj_surface_destroy_clears_current_sync_points() {
        let Some((_display, display_handle, mut server_state, _client_side, server_client, surface)) =
            focused_commit_state()
        else {
            return;
        };
        test_utils::install_surface_for_tests::<ProtocolServerState>(&display_handle, &surface);
        let buffer = protocol_dmabuf_buffer(&server_client, &display_handle);
        let (acquire_point, release_point) =
            DrmSyncPoint::invalid_timeline_pair_for_tests(35, 36).expect("create current test sync points");
        compositor::with_states(&surface, |states| {
            let mut attributes = states.cached_state.get::<SurfaceAttributes>();
            attributes.current().buffer = Some(BufferAssignment::NewBuffer(buffer));
            let mut syncobj = states.cached_state.get::<DrmSyncobjCachedState>();
            let current = syncobj.current();
            current.acquire_point = Some(acquire_point);
            current.release_point = Some(release_point);
        });

        test_utils::destroy_surface_for_tests(&surface);
        compositor::test_utils::commit_buffer_assignment(&mut server_state, &display_handle, &surface, None);

        assert!(current_sync_points_are_empty(&surface));
        assert!(current_buffer_has_no_new_buffer(&surface));
    }

    #[test]
    fn buffer_replacement_after_syncobj_surface_destroy_clears_current_sync_points() {
        let Some((_display, display_handle, mut server_state, _client_side, server_client, surface)) =
            focused_commit_state()
        else {
            return;
        };
        test_utils::install_surface_for_tests::<ProtocolServerState>(&display_handle, &surface);
        let current_buffer = protocol_dmabuf_buffer(&server_client, &display_handle);
        let replacement_buffer = protocol_buffer(&server_client, &display_handle);
        let (acquire_point, release_point) =
            DrmSyncPoint::invalid_timeline_pair_for_tests(37, 38).expect("create current test sync points");
        compositor::with_states(&surface, |states| {
            let mut attributes = states.cached_state.get::<SurfaceAttributes>();
            attributes.current().buffer = Some(BufferAssignment::NewBuffer(current_buffer));
            let mut syncobj = states.cached_state.get::<DrmSyncobjCachedState>();
            let current = syncobj.current();
            current.acquire_point = Some(acquire_point);
            current.release_point = Some(release_point);
        });

        test_utils::destroy_surface_for_tests(&surface);
        compositor::test_utils::commit_buffer_assignment(
            &mut server_state,
            &display_handle,
            &surface,
            Some(replacement_buffer),
        );

        assert!(current_sync_points_are_empty(&surface));
        assert!(current_buffer_is_some(&surface));
    }

    #[test]
    fn same_buffer_reattach_after_syncobj_surface_destroy_keeps_current_sync_points() {
        let Some((_display, display_handle, mut server_state, _client_side, server_client, surface)) =
            focused_commit_state()
        else {
            return;
        };
        test_utils::install_surface_for_tests::<ProtocolServerState>(&display_handle, &surface);
        let current_buffer = protocol_dmabuf_buffer(&server_client, &display_handle);
        let (acquire_point, release_point) =
            DrmSyncPoint::invalid_timeline_pair_for_tests(39, 40).expect("create current test sync points");
        compositor::with_states(&surface, |states| {
            let mut attributes = states.cached_state.get::<SurfaceAttributes>();
            attributes.current().buffer = Some(BufferAssignment::NewBuffer(current_buffer.clone()));
            let mut syncobj = states.cached_state.get::<DrmSyncobjCachedState>();
            let current = syncobj.current();
            current.acquire_point = Some(acquire_point);
            current.release_point = Some(release_point);
        });

        test_utils::destroy_surface_for_tests(&surface);
        compositor::test_utils::commit_buffer_assignment(
            &mut server_state,
            &display_handle,
            &surface,
            Some(current_buffer),
        );

        assert!(!current_sync_points_are_empty(&surface));
        assert!(current_buffer_is_some(&surface));
    }

    #[test]
    fn acquire_only_clear_current_marker_discard_keeps_current_sync_points() {
        let Some((_display, display_handle, _server_state, _client_side, _server_client, _surface)) =
            focused_commit_state()
        else {
            return;
        };
        let (acquire_point, release_point) =
            DrmSyncPoint::invalid_timeline_pair_for_tests(41, 42).expect("create current test sync points");
        let clear_marker = acquire_point.clone();
        let mut current = DrmSyncobjCachedState {
            acquire_point: Some(acquire_point),
            release_point: Some(release_point.clone()),
        };
        let mut pending = DrmSyncobjCachedState {
            acquire_point: Some(clear_marker),
            release_point: None,
        };
        let committed = pending.commit(&display_handle);

        committed.discard(&mut current);

        assert!(current.acquire_point.is_some());
        assert!(current.release_point.is_some());
    }

    #[test]
    fn queued_removal_after_syncobj_surface_destroy_clears_latest_sync_points() {
        let Some((_display, display_handle, mut server_state, _client_side, server_client, surface)) =
            focused_commit_state()
        else {
            return;
        };
        test_utils::install_surface_for_tests::<ProtocolServerState>(&display_handle, &surface);
        let current_buffer = protocol_dmabuf_buffer(&server_client, &display_handle);
        let queued_buffer = protocol_dmabuf_buffer(&server_client, &display_handle);
        let (current_acquire, current_release) =
            DrmSyncPoint::invalid_timeline_pair_for_tests(43, 44).expect("create current test sync points");
        let (queued_acquire, queued_release) =
            DrmSyncPoint::invalid_timeline_pair_for_tests(45, 46).expect("create queued test sync points");
        compositor::with_states(&surface, |states| {
            let mut attributes = states.cached_state.get::<SurfaceAttributes>();
            attributes.current().buffer = Some(BufferAssignment::NewBuffer(current_buffer));
            let mut syncobj = states.cached_state.get::<DrmSyncobjCachedState>();
            let current = syncobj.current();
            current.acquire_point = Some(current_acquire);
            current.release_point = Some(current_release);
        });
        queue_explicit_buffer_state(
            &surface,
            &display_handle,
            1,
            queued_buffer,
            queued_acquire,
            queued_release,
        );

        test_utils::destroy_surface_for_tests(&surface);
        queue_buffer_assignment_after_pre_commit(&mut server_state, &surface, &display_handle, 2, None);
        compositor::with_states(&surface, |states| {
            states.cached_state.apply_state(Serial::from(2), &display_handle);
        });

        assert!(current_sync_points_are_empty(&surface));
        assert!(current_buffer_has_no_new_buffer(&surface));
    }

    #[test]
    fn queued_same_buffer_reattach_after_syncobj_surface_destroy_keeps_latest_sync_points() {
        let Some((_display, display_handle, mut server_state, _client_side, server_client, surface)) =
            focused_commit_state()
        else {
            return;
        };
        test_utils::install_surface_for_tests::<ProtocolServerState>(&display_handle, &surface);
        let current_buffer = protocol_dmabuf_buffer(&server_client, &display_handle);
        let queued_buffer = protocol_dmabuf_buffer(&server_client, &display_handle);
        let (current_acquire, current_release) =
            DrmSyncPoint::invalid_timeline_pair_for_tests(47, 48).expect("create current test sync points");
        let (queued_acquire, queued_release) =
            DrmSyncPoint::invalid_timeline_pair_for_tests(49, 50).expect("create queued test sync points");
        compositor::with_states(&surface, |states| {
            let mut attributes = states.cached_state.get::<SurfaceAttributes>();
            attributes.current().buffer = Some(BufferAssignment::NewBuffer(current_buffer));
            let mut syncobj = states.cached_state.get::<DrmSyncobjCachedState>();
            let current = syncobj.current();
            current.acquire_point = Some(current_acquire);
            current.release_point = Some(current_release);
        });
        queue_explicit_buffer_state(
            &surface,
            &display_handle,
            1,
            queued_buffer.clone(),
            queued_acquire,
            queued_release,
        );

        test_utils::destroy_surface_for_tests(&surface);
        queue_buffer_assignment_after_pre_commit(
            &mut server_state,
            &surface,
            &display_handle,
            2,
            Some(queued_buffer),
        );
        compositor::with_states(&surface, |states| {
            states.cached_state.apply_state(Serial::from(2), &display_handle);
        });

        assert!(!current_sync_points_are_empty(&surface));
        assert!(current_buffer_is_some(&surface));
    }

    #[test]
    fn chained_queued_same_buffer_reattach_keeps_latest_sync_points() {
        let Some((_display, display_handle, mut server_state, _client_side, server_client, surface)) =
            focused_commit_state()
        else {
            return;
        };
        test_utils::install_surface_for_tests::<ProtocolServerState>(&display_handle, &surface);
        let current_buffer = protocol_dmabuf_buffer(&server_client, &display_handle);
        let queued_buffer = protocol_dmabuf_buffer(&server_client, &display_handle);
        let (current_acquire, current_release) =
            DrmSyncPoint::invalid_timeline_pair_for_tests(65, 66).expect("create current test sync points");
        let (queued_acquire, queued_release) =
            DrmSyncPoint::invalid_timeline_pair_for_tests(67, 68).expect("create queued test sync points");
        compositor::with_states(&surface, |states| {
            let mut attributes = states.cached_state.get::<SurfaceAttributes>();
            attributes.current().buffer = Some(BufferAssignment::NewBuffer(current_buffer));
            let mut syncobj = states.cached_state.get::<DrmSyncobjCachedState>();
            let current = syncobj.current();
            current.acquire_point = Some(current_acquire);
            current.release_point = Some(current_release);
        });
        queue_explicit_buffer_state(
            &surface,
            &display_handle,
            1,
            queued_buffer.clone(),
            queued_acquire,
            queued_release,
        );

        test_utils::destroy_surface_for_tests(&surface);
        queue_buffer_assignment_after_pre_commit(
            &mut server_state,
            &surface,
            &display_handle,
            2,
            Some(queued_buffer.clone()),
        );
        queue_buffer_assignment_after_pre_commit(
            &mut server_state,
            &surface,
            &display_handle,
            3,
            Some(queued_buffer),
        );
        compositor::with_states(&surface, |states| {
            states.cached_state.apply_state(Serial::from(3), &display_handle);
        });

        assert!(!current_sync_points_are_empty(&surface));
        assert!(current_buffer_is_some(&surface));
    }

    #[test]
    fn queued_no_buffer_state_before_same_buffer_reattach_keeps_latest_sync_points() {
        let Some((_display, display_handle, mut server_state, _client_side, server_client, surface)) =
            focused_commit_state()
        else {
            return;
        };
        test_utils::install_surface_for_tests::<ProtocolServerState>(&display_handle, &surface);
        let current_buffer = protocol_dmabuf_buffer(&server_client, &display_handle);
        let queued_buffer = protocol_dmabuf_buffer(&server_client, &display_handle);
        let (current_acquire, current_release) =
            DrmSyncPoint::invalid_timeline_pair_for_tests(55, 56).expect("create current test sync points");
        let (queued_acquire, queued_release) =
            DrmSyncPoint::invalid_timeline_pair_for_tests(57, 58).expect("create queued test sync points");
        compositor::with_states(&surface, |states| {
            let mut attributes = states.cached_state.get::<SurfaceAttributes>();
            attributes.current().buffer = Some(BufferAssignment::NewBuffer(current_buffer));
            let mut syncobj = states.cached_state.get::<DrmSyncobjCachedState>();
            let current = syncobj.current();
            current.acquire_point = Some(current_acquire);
            current.release_point = Some(current_release);
        });
        queue_explicit_buffer_state(
            &surface,
            &display_handle,
            1,
            queued_buffer.clone(),
            queued_acquire,
            queued_release,
        );
        compositor::test_utils::commit_pending_state_with_serial(&surface, &display_handle, 2);

        test_utils::destroy_surface_for_tests(&surface);
        queue_buffer_assignment_after_pre_commit(
            &mut server_state,
            &surface,
            &display_handle,
            3,
            Some(queued_buffer),
        );
        compositor::with_states(&surface, |states| {
            states.cached_state.apply_state(Serial::from(3), &display_handle);
        });

        assert!(!current_sync_points_are_empty(&surface));
        assert!(current_buffer_is_some(&surface));
    }

    #[test]
    fn queued_clear_current_marker_discard_keeps_latest_sync_points() {
        let Some((_display, display_handle, mut server_state, _client_side, server_client, surface)) =
            focused_commit_state()
        else {
            return;
        };
        test_utils::install_surface_for_tests::<ProtocolServerState>(&display_handle, &surface);
        let current_buffer = protocol_dmabuf_buffer(&server_client, &display_handle);
        let queued_buffer = protocol_dmabuf_buffer(&server_client, &display_handle);
        let (current_acquire, current_release) =
            DrmSyncPoint::invalid_timeline_pair_for_tests(51, 52).expect("create current test sync points");
        let (queued_acquire, queued_release) =
            DrmSyncPoint::invalid_timeline_pair_for_tests(53, 54).expect("create queued test sync points");
        compositor::with_states(&surface, |states| {
            let mut attributes = states.cached_state.get::<SurfaceAttributes>();
            attributes.current().buffer = Some(BufferAssignment::NewBuffer(current_buffer));
            let mut syncobj = states.cached_state.get::<DrmSyncobjCachedState>();
            let current = syncobj.current();
            current.acquire_point = Some(current_acquire);
            current.release_point = Some(current_release);
        });
        queue_explicit_buffer_state(
            &surface,
            &display_handle,
            1,
            queued_buffer,
            queued_acquire,
            queued_release,
        );

        test_utils::destroy_surface_for_tests(&surface);
        queue_buffer_assignment_after_pre_commit(&mut server_state, &surface, &display_handle, 2, None);
        compositor::with_states(&surface, |states| {
            states
                .cached_state
                .discard_state_range(Serial::from(2), Serial::from(2));
            states.cached_state.apply_state(Serial::from(1), &display_handle);
        });

        assert!(!current_sync_points_are_empty(&surface));
        assert!(current_buffer_is_some(&surface));
    }

    #[test]
    fn queued_same_buffer_after_discarded_marker_clears_current_sync_points() {
        let Some((_display, display_handle, mut server_state, _client_side, server_client, surface)) =
            focused_commit_state()
        else {
            return;
        };
        test_utils::install_surface_for_tests::<ProtocolServerState>(&display_handle, &surface);
        let current_buffer = protocol_dmabuf_buffer(&server_client, &display_handle);
        let replacement_buffer = protocol_buffer(&server_client, &display_handle);
        let (current_acquire, current_release) =
            DrmSyncPoint::invalid_timeline_pair_for_tests(59, 60).expect("create current test sync points");
        compositor::with_states(&surface, |states| {
            let mut attributes = states.cached_state.get::<SurfaceAttributes>();
            attributes.current().buffer = Some(BufferAssignment::NewBuffer(current_buffer));
            let mut syncobj = states.cached_state.get::<DrmSyncobjCachedState>();
            let current = syncobj.current();
            current.acquire_point = Some(current_acquire);
            current.release_point = Some(current_release);
        });

        test_utils::destroy_surface_for_tests(&surface);
        queue_buffer_assignment_after_pre_commit(
            &mut server_state,
            &surface,
            &display_handle,
            1,
            Some(replacement_buffer.clone()),
        );
        queue_buffer_assignment_after_pre_commit(
            &mut server_state,
            &surface,
            &display_handle,
            2,
            Some(replacement_buffer),
        );
        compositor::with_states(&surface, |states| {
            states
                .cached_state
                .discard_state_range(Serial::from(1), Serial::from(1));
            states.cached_state.apply_state(Serial::from(2), &display_handle);
        });

        assert!(current_sync_points_are_empty(&surface));
        assert!(current_buffer_is_some(&surface));
    }

    #[test]
    fn queued_same_buffer_after_discarded_explicit_attach_clears_current_sync_points() {
        let Some((_display, display_handle, mut server_state, _client_side, server_client, surface)) =
            focused_commit_state()
        else {
            return;
        };
        test_utils::install_surface_for_tests::<ProtocolServerState>(&display_handle, &surface);
        let current_buffer = protocol_dmabuf_buffer(&server_client, &display_handle);
        let queued_buffer = protocol_dmabuf_buffer(&server_client, &display_handle);
        let (current_acquire, current_release) =
            DrmSyncPoint::invalid_timeline_pair_for_tests(61, 62).expect("create current test sync points");
        let (queued_acquire, queued_release) =
            DrmSyncPoint::invalid_timeline_pair_for_tests(63, 64).expect("create queued test sync points");
        compositor::with_states(&surface, |states| {
            let mut attributes = states.cached_state.get::<SurfaceAttributes>();
            attributes.current().buffer = Some(BufferAssignment::NewBuffer(current_buffer));
            let mut syncobj = states.cached_state.get::<DrmSyncobjCachedState>();
            let current = syncobj.current();
            current.acquire_point = Some(current_acquire);
            current.release_point = Some(current_release);
        });
        queue_explicit_buffer_state(
            &surface,
            &display_handle,
            1,
            queued_buffer.clone(),
            queued_acquire,
            queued_release,
        );

        test_utils::destroy_surface_for_tests(&surface);
        queue_buffer_assignment_after_pre_commit(
            &mut server_state,
            &surface,
            &display_handle,
            2,
            Some(queued_buffer),
        );
        compositor::with_states(&surface, |states| {
            states
                .cached_state
                .discard_state_range(Serial::from(1), Serial::from(1));
            states.cached_state.apply_state(Serial::from(2), &display_handle);
        });

        assert!(current_sync_points_are_empty(&surface));
        assert!(current_buffer_is_some(&surface));
    }

    #[test]
    fn invalid_commit_attach_null_with_sync_points_preserves_current_buffer() {
        let Some((_display, display_handle, mut server_state, _client_side, server_client, surface)) =
            focused_commit_state()
        else {
            return;
        };
        let current_buffer = protocol_buffer(&server_client, &display_handle);
        compositor::test_utils::commit_buffer_assignment(
            &mut server_state,
            &display_handle,
            &surface,
            Some(current_buffer),
        );
        assert!(current_buffer_is_some(&surface));

        test_utils::install_surface_for_tests::<ProtocolServerState>(&display_handle, &surface);
        let (acquire_point, release_point) =
            DrmSyncPoint::invalid_timeline_pair_for_tests(11, 12).expect("create ordered points");
        stage_pending_sync_points(&surface, Some(acquire_point), Some(release_point));
        compositor::test_utils::commit_buffer_assignment(&mut server_state, &display_handle, &surface, None);

        assert!(current_sync_points_are_empty(&surface));
        assert!(current_buffer_is_some(&surface));
    }

    #[test]
    fn get_surface_request_installs_syncobj_surface_state() {
        let mut display = match Display::<ProtocolServerState>::new() {
            Ok(display) => display,
            Err(InitError::NoWaylandLib) => return,
            Err(err) => panic!("failed to create test Wayland display: {err}"),
        };
        let mut display_handle: DisplayHandle = display.handle();
        let compositor_state = compositor::CompositorState::new::<ProtocolServerState>(&display_handle);
        let syncobj_state =
            DrmSyncobjState::new_without_import_device_for_tests::<ProtocolServerState>(&display_handle);
        let mut server_state = ProtocolServerState {
            compositor_state,
            syncobj_state: Some(syncobj_state),
            surfaces: Vec::new(),
        };
        let (client_side, server_side) = UnixStream::pair().unwrap();
        let _server_client = display_handle
            .insert_client(server_side, Arc::new(ProtocolClientData::default()))
            .expect("insert test client");

        let client_connection = Connection::from_socket(client_side).expect("connect test client socket");
        let mut event_queue = client_connection.new_event_queue();
        let qh = event_queue.handle();
        let mut client_state = ProtocolClientState::default();

        client_connection.display().get_registry(&qh, ());
        client_connection.flush().expect("flush get_registry");
        pump_server(&mut display, &mut server_state);
        event_queue
            .blocking_dispatch(&mut client_state)
            .expect("dispatch registry globals");

        let compositor = client_state
            .compositor
            .as_ref()
            .expect("test compositor global should be advertised");
        let syncobj_manager = client_state
            .syncobj_manager
            .as_ref()
            .expect("test syncobj global should be advertised");
        let surface = compositor.create_surface(&qh, ());
        let syncobj_surface = syncobj_manager.get_surface(&surface, &qh, ());
        client_state.surface = Some(surface);
        client_state.syncobj_surface = Some(syncobj_surface);
        client_connection.flush().expect("flush get_surface request");
        pump_server(&mut display, &mut server_state);

        let server_surface = server_state
            .surfaces
            .first()
            .expect("client create_surface should reach server state");
        assert!(
            syncobj_surface_marker_is_some(server_surface),
            "client get_surface request should install server syncobj surface marker"
        );
    }

    #[test]
    fn get_surface_destroy_allows_protocol_reinstall() {
        let mut display = match Display::<ProtocolServerState>::new() {
            Ok(display) => display,
            Err(InitError::NoWaylandLib) => return,
            Err(err) => panic!("failed to create test Wayland display: {err}"),
        };
        let mut display_handle: DisplayHandle = display.handle();
        let compositor_state = compositor::CompositorState::new::<ProtocolServerState>(&display_handle);
        let syncobj_state =
            DrmSyncobjState::new_without_import_device_for_tests::<ProtocolServerState>(&display_handle);
        let mut server_state = ProtocolServerState {
            compositor_state,
            syncobj_state: Some(syncobj_state),
            surfaces: Vec::new(),
        };
        let (client_side, server_side) = UnixStream::pair().unwrap();
        let _server_client = display_handle
            .insert_client(server_side, Arc::new(ProtocolClientData::default()))
            .expect("insert test client");

        let client_connection = Connection::from_socket(client_side).expect("connect test client socket");
        let mut event_queue = client_connection.new_event_queue();
        let qh = event_queue.handle();
        let mut client_state = ProtocolClientState::default();

        client_connection.display().get_registry(&qh, ());
        client_connection.flush().expect("flush get_registry");
        pump_server(&mut display, &mut server_state);
        event_queue
            .blocking_dispatch(&mut client_state)
            .expect("dispatch registry globals");

        let compositor = client_state
            .compositor
            .as_ref()
            .expect("test compositor global should be advertised");
        let syncobj_manager = client_state
            .syncobj_manager
            .as_ref()
            .expect("test syncobj global should be advertised");
        let surface = compositor.create_surface(&qh, ());
        let syncobj_surface = syncobj_manager.get_surface(&surface, &qh, ());
        client_state.surface = Some(surface);
        client_state.syncobj_surface = Some(syncobj_surface);
        client_connection
            .flush()
            .expect("flush initial get_surface request");
        pump_server(&mut display, &mut server_state);

        let server_surface = server_state
            .surfaces
            .first()
            .expect("client create_surface should reach server state")
            .clone();
        assert!(
            syncobj_surface_marker_is_some(&server_surface),
            "initial get_surface should install server syncobj surface marker"
        );
        client_state
            .syncobj_surface
            .take()
            .expect("test syncobj surface should exist")
            .destroy();
        client_connection.flush().expect("flush syncobj surface destroy");
        pump_server(&mut display, &mut server_state);
        assert!(
            !syncobj_surface_marker_is_some(&server_surface),
            "destroy request should clear server syncobj surface marker"
        );

        let surface = client_state
            .surface
            .as_ref()
            .expect("test wl_surface should remain live");
        let syncobj_surface = syncobj_manager.get_surface(surface, &qh, ());
        client_state.syncobj_surface = Some(syncobj_surface);
        client_connection
            .flush()
            .expect("flush reinstall get_surface request");
        pump_server(&mut display, &mut server_state);
        assert!(
            syncobj_surface_marker_is_some(&server_surface),
            "second get_surface should reinstall server syncobj surface marker"
        );
    }

    #[test]
    fn duplicate_get_surface_reports_protocol_error_without_replacing_marker() {
        let mut display = match Display::<ProtocolServerState>::new() {
            Ok(display) => display,
            Err(InitError::NoWaylandLib) => return,
            Err(err) => panic!("failed to create test Wayland display: {err}"),
        };
        let mut display_handle: DisplayHandle = display.handle();
        let compositor_state = compositor::CompositorState::new::<ProtocolServerState>(&display_handle);
        let syncobj_state =
            DrmSyncobjState::new_without_import_device_for_tests::<ProtocolServerState>(&display_handle);
        let mut server_state = ProtocolServerState {
            compositor_state,
            syncobj_state: Some(syncobj_state),
            surfaces: Vec::new(),
        };
        let (client_side, server_side) = UnixStream::pair().unwrap();
        let _server_client = display_handle
            .insert_client(server_side, Arc::new(ProtocolClientData::default()))
            .expect("insert test client");

        let client_connection = Connection::from_socket(client_side).expect("connect test client socket");
        let mut event_queue = client_connection.new_event_queue();
        let qh = event_queue.handle();
        let mut client_state = ProtocolClientState::default();

        client_connection.display().get_registry(&qh, ());
        client_connection.flush().expect("flush get_registry");
        pump_server(&mut display, &mut server_state);
        event_queue
            .blocking_dispatch(&mut client_state)
            .expect("dispatch registry globals");

        let compositor = client_state
            .compositor
            .as_ref()
            .expect("test compositor global should be advertised");
        let syncobj_manager = client_state
            .syncobj_manager
            .as_ref()
            .expect("test syncobj global should be advertised")
            .clone();
        let surface = compositor.create_surface(&qh, ());
        let syncobj_surface = syncobj_manager.get_surface(&surface, &qh, ());
        client_state.surface = Some(surface);
        client_state.syncobj_surface = Some(syncobj_surface);
        client_connection
            .flush()
            .expect("flush initial get_surface request");
        pump_server(&mut display, &mut server_state);

        let server_surface = server_state
            .surfaces
            .first()
            .expect("client create_surface should reach server state")
            .clone();
        assert!(
            syncobj_surface_marker_is_some(&server_surface),
            "initial get_surface should install server syncobj surface marker"
        );
        let original_marker_id = syncobj_surface_marker_protocol_id(&server_surface)
            .expect("initial get_surface should install server syncobj surface marker");

        let surface = client_state
            .surface
            .as_ref()
            .expect("test wl_surface should remain live");
        let duplicate_syncobj_surface = syncobj_manager.get_surface(surface, &qh, ());
        client_state.syncobj_surface = Some(duplicate_syncobj_surface);
        client_connection
            .flush()
            .expect("flush duplicate get_surface request");
        pump_server(&mut display, &mut server_state);

        let protocol_error =
            read_protocol_error_after_server_pump(&mut event_queue, &client_connection, &mut client_state);
        assert_eq!(protocol_error.0, "wp_linux_drm_syncobj_manager_v1");
        assert_eq!(
            protocol_error.1, 0,
            "surface_exists is error code 0 in linux-drm-syncobj-v1"
        );
        assert!(
            syncobj_surface_marker_is_some(&server_surface),
            "duplicate get_surface must not clear or replace the existing marker"
        );
        assert_eq!(
            syncobj_surface_marker_protocol_id(&server_surface),
            Some(original_marker_id),
            "duplicate get_surface must preserve the original marker resource"
        );
    }

    #[test]
    fn import_timeline_without_import_device_reports_invalid_timeline() {
        let mut display = match Display::<ProtocolServerState>::new() {
            Ok(display) => display,
            Err(InitError::NoWaylandLib) => return,
            Err(err) => panic!("failed to create test Wayland display: {err}"),
        };
        let mut display_handle: DisplayHandle = display.handle();
        let compositor_state = compositor::CompositorState::new::<ProtocolServerState>(&display_handle);
        let syncobj_state =
            DrmSyncobjState::new_without_import_device_for_tests::<ProtocolServerState>(&display_handle);
        let mut server_state = ProtocolServerState {
            compositor_state,
            syncobj_state: Some(syncobj_state),
            surfaces: Vec::new(),
        };
        let (client_side, server_side) = UnixStream::pair().unwrap();
        let _server_client = display_handle
            .insert_client(server_side, Arc::new(ProtocolClientData::default()))
            .expect("insert test client");

        let client_connection = Connection::from_socket(client_side).expect("connect test client socket");
        let mut event_queue = client_connection.new_event_queue();
        let qh = event_queue.handle();
        let mut client_state = ProtocolClientState::default();

        client_connection.display().get_registry(&qh, ());
        client_connection.flush().expect("flush get_registry");
        pump_server(&mut display, &mut server_state);
        event_queue
            .blocking_dispatch(&mut client_state)
            .expect("dispatch registry globals");

        let syncobj_manager = client_state
            .syncobj_manager
            .as_ref()
            .expect("test syncobj global should be advertised");
        let timeline_fd = rustix::event::eventfd(0, rustix::event::EventfdFlags::CLOEXEC)
            .expect("create test non-syncobj fd");
        let timeline = syncobj_manager.import_timeline(timeline_fd.as_fd(), &qh, ());
        client_state.timeline = Some(timeline);
        client_connection.flush().expect("flush import_timeline request");
        pump_server(&mut display, &mut server_state);

        let protocol_error =
            read_protocol_error_after_server_pump(&mut event_queue, &client_connection, &mut client_state);
        assert_eq!(protocol_error.0, "wp_linux_drm_syncobj_manager_v1");
        assert_eq!(
            protocol_error.1, 1,
            "invalid_timeline is error code 1 in linux-drm-syncobj-v1"
        );
        assert!(
            server_state
                .syncobj_state
                .as_ref()
                .expect("test syncobj state should remain installed")
                .known_timelines
                .is_empty(),
            "failed import_timeline must not install a server timeline"
        );
    }
}
