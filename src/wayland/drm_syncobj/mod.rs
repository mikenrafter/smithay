//! DRM syncobj protocol
//!
//! This module implement the `linux-drm-syncobj-v1` protocol, used to support
//! explicit sync.
//!
//! Currently, the implementation here assumes acquire fences are already signalled
//! when the surface transaction is ready. Use [`DrmSyncPointBlocker`].
//!
//! The server should only expose the protocol if [`supports_syncobj_eventfd`] returns
//! `true`. Or it won't be possible to create the blocker. This is similar to other
//! implementations.
//!
//! The release fence is signalled when all references to a
//! [`Buffer`][crate::backend::renderer::utils::Buffer] are dropped.
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
//! let syncobj_state = if supports_syncobj_eventfd(&import_device) {
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
    backend::drm::DrmDeviceFd,
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

impl Cacheable for DrmSyncobjCachedState {
    fn commit(&mut self, _dh: &DisplayHandle) -> Self {
        Self {
            acquire_point: self.acquire_point.take(),
            release_point: self.release_point.take(),
        }
    }

    fn merge_into(self, into: &mut Self, _dh: &DisplayHandle) {
        if self.acquire_point.is_some() && self.release_point.is_some() {
            if let Some(release_point) = &into.release_point {
                if let Err(err) = release_point.signal() {
                    tracing::error!("Failed to signal syncobj release point: {}", err);
                }
            }
            into.acquire_point = self.acquire_point;
            into.release_point = self.release_point;
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

fn commit_hook<D: DrmSyncobjHandler>(_data: &mut D, _dh: &DisplayHandle, surface: &WlSurface) {
    compositor::with_states(surface, |states| {
        let mut cached = states.cached_state.get::<SurfaceAttributes>();
        let pending = cached.pending();
        let new_buffer = pending.buffer.as_ref().and_then(|buffer| match buffer {
            BufferAssignment::NewBuffer(buffer) => Some(buffer),
            _ => None,
        });
        if let Some(data) = states
            .data_map
            .get::<RefCell<Option<WpLinuxDrmSyncobjSurfaceV1>>>()
        {
            if let Some(syncobj_surface) = data.borrow().as_ref() {
                let mut cached = states.cached_state.get::<DrmSyncobjCachedState>();
                let pending = cached.pending();
                if pending.acquire_point.is_some() && new_buffer.is_none() {
                    syncobj_surface.post_error(
                        wp_linux_drm_syncobj_surface_v1::Error::NoBuffer,
                        "acquire point without buffer".to_string(),
                    );
                } else if pending.acquire_point.is_some() && pending.release_point.is_none() {
                    syncobj_surface.post_error(
                        wp_linux_drm_syncobj_surface_v1::Error::NoReleasePoint,
                        "acquire point without release point".to_string(),
                    );
                } else if pending.acquire_point.is_none() && pending.release_point.is_some() {
                    syncobj_surface.post_error(
                        wp_linux_drm_syncobj_surface_v1::Error::NoAcquirePoint,
                        "release point without acquire point".to_string(),
                    );
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
                    }
                    if let Some(buffer) = new_buffer {
                        if get_dmabuf(buffer).is_err() {
                            syncobj_surface.post_error(
                                wp_linux_drm_syncobj_surface_v1::Error::UnsupportedBuffer,
                                "sync points with non-dmabuf buffer".to_string(),
                            );
                        }
                    }
                }
            }
        }
    });
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
                let commit_hook_id = compositor::add_pre_commit_hook::<D, _>(&surface, commit_hook);
                let destruction_hook_id =
                    compositor::add_destruction_hook::<D, _>(&surface, destruction_hook);
                let syncobj_surface = data_init.init::<_, _>(
                    id,
                    DrmSyncobjSurfaceData {
                        surface: surface.downgrade(),
                        commit_hook_id,
                        destruction_hook_id,
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
    use std::cell::RefCell;

    use wayland_protocols::wp::linux_drm_syncobj::v1::server::{
        wp_linux_drm_syncobj_surface_v1::WpLinuxDrmSyncobjSurfaceV1,
        wp_linux_drm_syncobj_timeline_v1::WpLinuxDrmSyncobjTimelineV1,
    };
    use wayland_server::{Client, Dispatch, DisplayHandle, Resource};

    use super::{
        DrmSyncPoint, DrmSyncobjHandler, DrmSyncobjSurfaceData, DrmSyncobjTimelineData, PendingSyncPointKind,
        commit_hook, destruction_hook, set_pending_sync_point_from_timeline_resource,
    };
    use crate::wayland::compositor::{self, with_states};

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
        D: Dispatch<WpLinuxDrmSyncobjSurfaceV1, DrmSyncobjSurfaceData> + DrmSyncobjHandler + 'static,
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

        let commit_hook_id = compositor::add_pre_commit_hook::<D, _>(surface, commit_hook);
        let destruction_hook_id = compositor::add_destruction_hook::<D, _>(surface, destruction_hook);
        let client = surface
            .client()
            .expect("test WlSurface should still be attached to a live client");
        let syncobj_surface = client
            .create_resource::<WpLinuxDrmSyncobjSurfaceV1, DrmSyncobjSurfaceData, D>(
                handle,
                1,
                DrmSyncobjSurfaceData {
                    surface: surface.downgrade(),
                    commit_hook_id,
                    destruction_hook_id,
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

    fn timeline_resource_for_tests<D>(
        client: &Client,
        handle: &DisplayHandle,
        point: &DrmSyncPoint,
    ) -> WpLinuxDrmSyncobjTimelineV1
    where
        D: Dispatch<WpLinuxDrmSyncobjTimelineV1, DrmSyncobjTimelineData> + 'static,
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
        D: Dispatch<WpLinuxDrmSyncobjTimelineV1, DrmSyncobjTimelineData> + 'static,
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
    commit_hook_id: HookId,
    destruction_hook_id: HookId,
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
                    compositor::remove_pre_commit_hook(&surface, &self.commit_hook_id);
                    compositor::remove_destruction_hook(&surface, &self.destruction_hook_id);
                    with_states(&surface, |states| {
                        *states
                            .data_map
                            .get::<RefCell<Option<WpLinuxDrmSyncobjSurfaceV1>>>()
                            .unwrap()
                            .borrow_mut() = None;
                        // Committed sync points should still be used, but pending points can
                        // be cleared.
                        let mut cached = states.cached_state.get::<DrmSyncobjCachedState>();
                        cached.pending().acquire_point = None;
                        if let Some(release_point) = cached.pending().release_point.take() {
                            if let Err(err) = release_point.signal() {
                                tracing::error!("Failed to signal syncobj release point: {}", err);
                            }
                        }
                    });
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
    use std::{os::unix::net::UnixStream, sync::Arc};

    use wayland_client::{
        Connection, Dispatch, QueueHandle, delegate_noop,
        protocol::{wl_compositor, wl_registry, wl_surface},
    };
    use wayland_protocols::wp::linux_drm_syncobj::v1::client::{
        wp_linux_drm_syncobj_manager_v1, wp_linux_drm_syncobj_surface_v1,
    };
    use wayland_server::{
        Display, DisplayHandle,
        backend::{ClientData, ClientId, DisconnectReason, InitError},
    };

    use super::*;
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

    fn pump_server(display: &mut Display<ProtocolServerState>, state: &mut ProtocolServerState) {
        display
            .dispatch_clients(state)
            .expect("dispatch test client requests");
        display.flush_clients().expect("flush test server events");
    }

    fn syncobj_surface_marker_is_some(surface: &wayland_server::protocol::wl_surface::WlSurface) -> bool {
        compositor::with_states(surface, |states| {
            states
                .data_map
                .get::<RefCell<Option<WpLinuxDrmSyncobjSurfaceV1>>>()
                .map(|resource| resource.borrow().is_some())
                .unwrap_or(false)
        })
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
}
