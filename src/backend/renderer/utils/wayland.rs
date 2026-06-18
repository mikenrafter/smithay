#[cfg(feature = "backend_drm")]
use crate::wayland::drm_syncobj::{DrmSyncPoint, DrmSyncobjCachedState};
use crate::{
    backend::renderer::{
        ContextId, ErasedContextId, ImportAll, Renderer, Texture, buffer_dimensions, buffer_has_alpha,
        element::RenderElement,
    },
    utils::{Buffer as BufferCoord, Coordinate, Logical, Physical, Point, Rectangle, Scale, Size, Transform},
    wayland::{
        compositor::{
            self, BufferAssignment, Damage, RectangleKind, SUBSURFACE_ROLE, SubsurfaceCachedState,
            SurfaceAttributes, SurfaceData, TraversalAction, add_destruction_hook, is_sync_subsurface,
            with_surface_tree_downward, with_surface_tree_upward,
        },
        viewporter,
    },
};

use std::{
    any::Any,
    collections::{HashMap, hash_map::Entry},
    convert::Infallible,
    sync::{Arc, Mutex},
};

use super::{CommitCounter, DamageBag, DamageSet, DamageSnapshot, SurfaceView};
use tracing::{error, instrument, warn};
use wayland_server::protocol::{wl_buffer::WlBuffer, wl_surface::WlSurface};

#[cfg(feature = "backend_drm")]
#[derive(Debug, Default)]
struct ReleasePointSlot(Mutex<Option<DrmSyncPoint>>);

#[cfg(feature = "backend_drm")]
impl ReleasePointSlot {
    fn new(release_point: Option<DrmSyncPoint>) -> Self {
        Self(Mutex::new(release_point))
    }

    fn get(&self) -> Option<DrmSyncPoint> {
        self.0
            .lock()
            .map(|release_point| release_point.as_ref().cloned())
            .unwrap_or_else(|_| {
                tracing::error!("Failed to lock syncobj release point");
                None
            })
    }

    fn take_for_renderer(&self) -> Option<DrmSyncPoint> {
        self.0
            .lock()
            .map(|mut release_point| release_point.take())
            .unwrap_or_else(|_| {
                tracing::error!("Failed to lock syncobj release point");
                None
            })
    }

    fn signal_on_drop(&self) {
        let Some(release_point) = self.take_for_renderer() else {
            return;
        };

        if let Err(err) = release_point.signal() {
            tracing::error!("Failed to signal syncobj release point: {}", err);
        }
    }
}

/// Type stored in WlSurface states data_map
///
/// ```rs
/// compositor::with_states(surface, |states| {
///     let data = states.data_map.get::<RendererSurfaceStateUserData>();
/// });
/// ```
pub type RendererSurfaceStateUserData = Mutex<RendererSurfaceState>;

/// Surface state for rendering related data
#[derive(Default, Debug)]
pub struct RendererSurfaceState {
    pub(crate) buffer_dimensions: Option<Size<i32, BufferCoord>>,
    pub(crate) buffer_scale: i32,
    pub(crate) buffer_transform: Transform,
    pub(crate) buffer_has_alpha: Option<bool>,
    pub(crate) buffer: Option<Buffer>,
    pub(crate) damage: DamageBag<i32, BufferCoord>,
    pub(crate) renderer_seen: HashMap<ErasedContextId, CommitCounter>,
    pub(crate) textures: HashMap<ErasedContextId, Box<dyn Any>>,
    retired_textures: HashMap<ErasedContextId, Vec<Box<dyn Any>>>,
    pub(crate) surface_view: Option<SurfaceView>,
    pub(crate) opaque_regions: Vec<Rectangle<i32, Logical>>,
}

/// SAFETY: Only thing unsafe here is the `Box<dyn Any>`, which are the textures.
/// Those are guarded by our Renderers handling thread-safety and the `ContextId`.
/// Theoretically a renderer could be thread-safe, but its texture type isn't, but that is **very** theoretical.
unsafe impl Send for RendererSurfaceState {}
unsafe impl Sync for RendererSurfaceState {}

#[derive(Debug)]
struct InnerBuffer {
    buffer: WlBuffer,
    #[cfg(feature = "backend_drm")]
    acquire_point: Option<DrmSyncPoint>,
    #[cfg(feature = "backend_drm")]
    release_point: ReleasePointSlot,
}

impl Drop for InnerBuffer {
    #[inline]
    fn drop(&mut self) {
        self.buffer.release();
        #[cfg(feature = "backend_drm")]
        self.release_point.signal_on_drop();
    }
}

/// A wayland buffer
#[derive(Debug, Clone)]
pub struct Buffer {
    inner: Arc<InnerBuffer>,
}

impl Buffer {
    /// Create a buffer with implicit sync
    pub fn with_implicit(buffer: WlBuffer) -> Self {
        Self {
            inner: Arc::new(InnerBuffer {
                buffer,
                #[cfg(feature = "backend_drm")]
                acquire_point: None,
                #[cfg(feature = "backend_drm")]
                release_point: ReleasePointSlot::new(None),
            }),
        }
    }

    /// Create a buffer with explicit acquire and release sync points
    #[cfg(feature = "backend_drm")]
    pub fn with_explicit(buffer: WlBuffer, acquire_point: DrmSyncPoint, release_point: DrmSyncPoint) -> Self {
        Self {
            inner: Arc::new(InnerBuffer {
                buffer,
                acquire_point: Some(acquire_point),
                release_point: ReleasePointSlot::new(Some(release_point)),
            }),
        }
    }

    #[cfg(feature = "backend_drm")]
    #[allow(dead_code)]
    pub(crate) fn acquire_point(&self) -> Option<&DrmSyncPoint> {
        self.inner.acquire_point.as_ref()
    }

    #[cfg(feature = "backend_drm")]
    #[allow(dead_code)]
    pub(crate) fn release_point(&self) -> Option<DrmSyncPoint> {
        self.inner.release_point.get()
    }

    #[cfg(feature = "backend_drm")]
    #[allow(dead_code)]
    pub(crate) fn take_release_point_for_renderer(&self) -> Option<DrmSyncPoint> {
        self.inner.release_point.take_for_renderer()
    }
}

impl std::ops::Deref for Buffer {
    type Target = WlBuffer;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner.buffer
    }
}

impl PartialEq<WlBuffer> for Buffer {
    #[inline]
    fn eq(&self, other: &WlBuffer) -> bool {
        self.inner.buffer == *other
    }
}

impl PartialEq<WlBuffer> for &Buffer {
    #[inline]
    fn eq(&self, other: &WlBuffer) -> bool {
        self.inner.buffer == *other
    }
}

fn should_replace_renderer_buffer(buffer_changed: bool, has_explicit_sync_points: bool) -> bool {
    buffer_changed || has_explicit_sync_points
}

impl RendererSurfaceState {
    #[profiling::function]
    pub(crate) fn update_buffer(&mut self, states: &SurfaceData) {
        #[cfg(feature = "backend_drm")]
        let mut guard = states.cached_state.get::<DrmSyncobjCachedState>();
        #[cfg(feature = "backend_drm")]
        let syncobj_state = guard.current();

        let mut guard = states.cached_state.get::<SurfaceAttributes>();
        let attrs = guard.current();

        let new_buffer = matches!(attrs.buffer, Some(BufferAssignment::NewBuffer(_)));
        match attrs.buffer.take() {
            Some(BufferAssignment::NewBuffer(buffer)) => {
                self.buffer_dimensions = buffer_dimensions(&buffer);
                if self.buffer_dimensions.is_none() {
                    // This results in us rendering nothing (can happen e.g. for failed egl-buffer-calls),
                    // but it is better than crashing the compositor for a bad buffer
                    self.reset();
                    return;
                }
                self.buffer_has_alpha = buffer_has_alpha(&buffer);
                self.buffer_scale = attrs.buffer_scale;
                self.buffer_transform = attrs.buffer_transform.into();

                let buffer_changed = !self.buffer.as_ref().is_some_and(|b| b == buffer);
                #[cfg(feature = "backend_drm")]
                let has_explicit_sync_points =
                    syncobj_state.acquire_point.is_some() || syncobj_state.release_point.is_some();
                #[cfg(not(feature = "backend_drm"))]
                let has_explicit_sync_points = false;

                self.retire_textures();

                // Explicit sync points are per commit, not per wl_buffer object. If the same
                // wl_buffer is attached again with fresh sync points, refresh the renderer-managed
                // wrapper so the commit-specific acquire/release points do not go stale.
                if should_replace_renderer_buffer(buffer_changed, has_explicit_sync_points) {
                    self.buffer = Some(Buffer {
                        inner: Arc::new(InnerBuffer {
                            buffer,
                            #[cfg(feature = "backend_drm")]
                            acquire_point: syncobj_state.acquire_point.take(),
                            #[cfg(feature = "backend_drm")]
                            release_point: ReleasePointSlot::new(syncobj_state.release_point.take()),
                        }),
                    });
                }
            }
            Some(BufferAssignment::Removed) => {
                self.reset();
                return;
            }
            None => {}
        };

        let Some(buffer_dimensions) = self.buffer_dimensions else {
            // nothing to be done without a buffer
            return;
        };

        let surface_size = buffer_dimensions.to_logical(self.buffer_scale, self.buffer_transform);
        let surface_view = SurfaceView::from_states(states, surface_size, attrs.client_scale);
        let surface_view_changed = self.surface_view.replace(surface_view) != Some(surface_view);

        // if we received a new buffer also process the attached damage
        if new_buffer {
            let buffer_damage = attrs.damage.drain(..).flat_map(|dmg| {
                match dmg {
                    Damage::Buffer(rect) => rect,
                    Damage::Surface(rect) => surface_view.rect_to_local(rect).to_i32_up().to_buffer(
                        self.buffer_scale,
                        self.buffer_transform,
                        &surface_size,
                    ),
                }
                .intersection(Rectangle::from_size(buffer_dimensions))
            });
            self.damage.add(buffer_damage);
        }

        // if the buffer or our view changed rebuild our opaque regions
        if new_buffer || surface_view_changed {
            self.opaque_regions.clear();
            if !self.buffer_has_alpha.unwrap_or(true) {
                self.opaque_regions.push(Rectangle::from_size(surface_view.dst))
            } else if let Some(region_attributes) = &attrs.opaque_region {
                let opaque_regions = region_attributes
                    .rects
                    .iter()
                    .map(|(kind, rect)| {
                        let dest_size = surface_view.dst;

                        let rect_constrained_loc = rect.loc.constrain(Rectangle::from_size(dest_size));
                        let rect_clamped_size = rect
                            .size
                            .clamp((0, 0), (dest_size.to_point() - rect_constrained_loc).to_size());

                        let rect = Rectangle::new(rect_constrained_loc, rect_clamped_size);

                        (kind, rect)
                    })
                    .fold(
                        std::mem::take(&mut self.opaque_regions),
                        |mut new_regions, (kind, rect)| {
                            match kind {
                                RectangleKind::Add => {
                                    let added_regions = rect.subtract_rects(
                                        new_regions
                                            .iter()
                                            .filter(|region| region.overlaps_or_touches(rect))
                                            .copied(),
                                    );
                                    new_regions.extend(added_regions);
                                }
                                RectangleKind::Subtract => {
                                    new_regions =
                                        Rectangle::subtract_rects_many_in_place(new_regions, [rect]);
                                }
                            }

                            new_regions
                        },
                    );

                self.opaque_regions = opaque_regions;
            }
        }
    }

    /// Get the current commit position of this surface
    ///
    /// The position should be saved after calling [`damage_since`](RendererSurfaceState::damage_since) and
    /// provided as the commit in the next call.
    pub fn current_commit(&self) -> CommitCounter {
        self.damage.current_commit()
    }

    /// Gets the damage since the last commit
    ///
    /// If either the commit is `None` or the commit is too old
    /// the whole buffer will be returned as damage.
    pub fn damage_since(&self, commit: Option<CommitCounter>) -> DamageSet<i32, BufferCoord> {
        self.damage.damage_since(commit).unwrap_or_else(|| {
            self.buffer_dimensions
                .as_ref()
                .map(|size| DamageSet::from_slice(&[Rectangle::from_size(*size)]))
                .unwrap_or_default()
        })
    }

    /// Gets the current damage of this surface
    pub fn damage(&self) -> DamageSnapshot<i32, BufferCoord> {
        self.damage.snapshot()
    }

    /// Returns the logical size of the current attached buffer
    pub fn buffer_size(&self) -> Option<Size<i32, Logical>> {
        self.buffer_dimensions
            .as_ref()
            .map(|dim| dim.to_logical(self.buffer_scale, self.buffer_transform))
    }

    /// Returns the scale of the current attached buffer
    pub fn buffer_scale(&self) -> i32 {
        self.buffer_scale
    }

    /// Returns the transform of the current attached buffer
    pub fn buffer_transform(&self) -> Transform {
        self.buffer_transform
    }

    /// Returns the logical size of the surface.
    ///
    /// Note: The surface size may not be equal to the buffer size in case
    /// a viewport has been attached to the surface.
    pub fn surface_size(&self) -> Option<Size<i32, Logical>> {
        self.surface_view.map(|view| view.dst)
    }

    /// Get the attached buffer.
    /// Can be used to check if surface is mapped
    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
    }

    /// Gets a reference to the texture for the specified renderer context
    pub fn texture<T>(&self, id: ContextId<T>) -> Option<&T>
    where
        T: Texture + 'static,
    {
        self.textures.get(&id.erased()).and_then(|e| e.downcast_ref())
    }

    pub(crate) fn drain_retired_textures<T, E, F>(
        &mut self,
        id: ContextId<T>,
        mut release_texture: F,
    ) -> Result<(), E>
    where
        T: Texture + 'static,
        F: FnMut(&T) -> Result<(), E>,
    {
        let erased = id.erased();
        let Some(textures) = self.retired_textures.remove(&erased) else {
            return Ok(());
        };

        let mut retained: Vec<Box<dyn Any>> = Vec::new();
        let mut textures = textures.into_iter();
        while let Some(texture) = textures.next() {
            match texture.downcast::<T>() {
                Ok(texture) => {
                    if let Err(err) = release_texture(&texture) {
                        let texture: Box<dyn Any> = texture;
                        retained.push(texture);
                        retained.extend(textures);
                        self.retired_textures.insert(erased, retained);
                        return Err(err);
                    }
                }
                Err(texture) => {
                    error!("Retired renderer texture did not match its context type");
                    retained.push(texture);
                }
            }
        }

        if !retained.is_empty() {
            self.retired_textures.insert(erased, retained);
        }

        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn drop_retired_textures<T>(&mut self, id: ContextId<T>)
    where
        T: Texture + 'static,
    {
        if let Err(err) = self.drain_retired_textures(id, |_| Ok::<(), Infallible>(())) {
            match err {}
        }
    }

    /// Gets the opaque regions of this surface
    pub fn opaque_regions(&self) -> Option<&[Rectangle<i32, Logical>]> {
        // If the surface is unmapped there can be no opaque regions
        self.surface_size()?;

        // To make it easier for upstream, but to no re-allocate a
        // new vec if the opaque regions change for return None
        // on empty regions
        if self.opaque_regions.is_empty() {
            return None;
        }

        Some(&self.opaque_regions[..])
    }

    /// Gets the [`SurfaceView`] of this surface
    pub fn view(&self) -> Option<SurfaceView> {
        self.surface_view
    }

    fn reset(&mut self) {
        self.buffer_dimensions = None;
        self.textures.clear();
        self.buffer = None;
        self.damage.reset();
        self.surface_view = None;
        self.buffer_has_alpha = None;
        self.opaque_regions.clear();
    }

    #[allow(dead_code)]
    fn retire_textures(&mut self) {
        for (context_id, texture) in self.textures.drain() {
            self.retired_textures.entry(context_id).or_default().push(texture);
        }
    }
}

/// Handler to let smithay take over buffer management.
///
/// Needs to be called first on the commit-callback of
/// [`crate::wayland::compositor::CompositorHandler::commit`].
///
/// Consumes the buffer of [`SurfaceAttributes`], the buffer will
/// not be accessible anymore, but [`draw_render_elements`] and other
/// `draw_*` helpers of the [desktop module](`crate::desktop`) will
/// become usable for surfaces handled this way.
#[profiling::function]
pub fn on_commit_buffer_handler<D: 'static>(surface: &WlSurface) {
    if !is_sync_subsurface(surface) {
        let mut new_surfaces = Vec::new();
        with_surface_tree_upward(
            surface,
            (),
            |_, _, _| TraversalAction::DoChildren(()),
            |surf, states, _| {
                if states
                    .data_map
                    .insert_if_missing_threadsafe(|| Mutex::new(RendererSurfaceState::default()))
                {
                    new_surfaces.push(surf.clone());
                }
                let mut data = states
                    .data_map
                    .get::<RendererSurfaceStateUserData>()
                    .unwrap()
                    .lock()
                    .unwrap();
                data.update_buffer(states);
            },
            |_, _, _| true,
        );
        for surf in &new_surfaces {
            add_destruction_hook(surf, |_: &mut D, surface| {
                // We reset the state on destruction before the user_data is dropped
                // to prevent a deadlock which can happen if we try to send a buffer
                // release during drop. This also enables us to free resources earlier
                // like the stored textures
                compositor::with_states(surface, |data| {
                    if let Some(mut state) = data
                        .data_map
                        .get::<RendererSurfaceStateUserData>()
                        .map(|s| s.lock().unwrap())
                    {
                        state.reset();
                    }
                    #[cfg(feature = "renderer_multi")]
                    crate::backend::renderer::multigpu::clear_surface_textures(data);
                });
            });
        }
    }
}

impl SurfaceView {
    fn from_states(states: &SurfaceData, surface_size: Size<i32, Logical>, client_scale: f64) -> SurfaceView {
        viewporter::ensure_viewport_valid(states, surface_size);
        let mut viewport_state = states.cached_state.get::<viewporter::ViewportCachedState>();
        let viewport = viewport_state.current();

        let src = viewport
            .src
            .unwrap_or_else(|| Rectangle::from_size(surface_size.to_f64()));
        let dst = viewport.size().unwrap_or(
            surface_size
                .to_f64()
                .to_client(1.)
                .to_logical(client_scale)
                .to_i32_round(),
        );
        let offset = if states.role == Some(SUBSURFACE_ROLE) {
            states
                .cached_state
                .get::<SubsurfaceCachedState>()
                .current()
                .location
        } else {
            Default::default()
        };
        SurfaceView { src, dst, offset }
    }

    pub(crate) fn rect_to_global<N>(&self, rect: Rectangle<N, Logical>) -> Rectangle<f64, Logical>
    where
        N: Coordinate,
    {
        let scale = self.scale();
        let mut rect = rect.to_f64();
        rect.loc -= self.src.loc;
        rect.upscale(scale)
    }

    pub(crate) fn rect_to_local<N>(&self, rect: Rectangle<N, Logical>) -> Rectangle<f64, Logical>
    where
        N: Coordinate,
    {
        let scale = self.scale();
        let mut rect = rect.to_f64().downscale(scale);
        rect.loc += self.src.loc;
        rect
    }

    fn scale(&self) -> Scale<f64> {
        Scale::from((
            self.dst.w as f64 / self.src.size.w,
            self.dst.h as f64 / self.src.size.h,
        ))
    }
}

/// Access the buffer related states associated to this surface
///
/// Calls [`compositor::with_states`] internally.
///
/// Returns `None`, if there never was a commit processed through `on_commit_buffer_handler`.
pub fn with_renderer_surface_state<F, T>(surface: &WlSurface, cb: F) -> Option<T>
where
    F: FnOnce(&mut RendererSurfaceState) -> T,
{
    compositor::with_states(surface, |states| {
        let data = states.data_map.get::<RendererSurfaceStateUserData>()?;
        Some(cb(&mut data.lock().unwrap()))
    })
}

/// Imports buffers of a surface using a given [`Renderer`]
///
/// This (or `import_surface_tree`) need to be called before`draw_render_elements`, if used later.
///
/// Note: This will do nothing, if you are not using
/// [`crate::backend::renderer::utils::on_commit_buffer_handler`]
/// to let smithay handle buffer management.
#[instrument(level = "trace", skip_all)]
#[profiling::function]
pub fn import_surface<R>(renderer: &mut R, states: &SurfaceData) -> Result<(), R::Error>
where
    R: Renderer + ImportAll,
    R::TextureId: 'static,
{
    if let Some(data) = states.data_map.get::<RendererSurfaceStateUserData>() {
        let context_id = renderer.context_id();
        let erased_context_id = context_id.erased();
        let mut data_ref = data.lock().unwrap();
        let data = &mut *data_ref;

        data.drain_retired_textures(context_id, |texture| {
            renderer.release_imported_texture_for_surface_cache(texture)
        })?;

        let last_commit = data.renderer_seen.get(&erased_context_id);
        let buffer_damage = data.damage_since(last_commit.copied());
        if let Entry::Vacant(e) = data.textures.entry(erased_context_id.clone()) {
            if let Some(buffer) = data.buffer.as_ref() {
                // There is no point in importing a single pixel buffer
                if matches!(
                    crate::backend::renderer::buffer_type(buffer),
                    Some(crate::backend::renderer::BufferType::SinglePixel)
                ) {
                    return Ok(());
                }

                match renderer.import_buffer_from_surface_state(buffer, Some(states), &buffer_damage) {
                    Some(Ok(m)) => {
                        e.insert(Box::new(m));
                        data.renderer_seen
                            .insert(erased_context_id, data.current_commit());
                    }
                    Some(Err(err)) => {
                        warn!("Error loading buffer: {}", err);
                        return Err(err);
                    }
                    None => {
                        error!("Unknown buffer format for: {:?}", buffer);
                    }
                }
            }
        }
    }

    Ok(())
}

/// Imports buffers of a surface and its subsurfaces using a given [`Renderer`].
///
/// This (or `import_surface`) need to be called before `draw_render_elements`, if used later.
///
/// Note: This will do nothing, if you are not using
/// [`crate::backend::renderer::utils::on_commit_buffer_handler`]
/// to let smithay handle buffer management.
#[instrument(level = "trace", skip_all)]
#[profiling::function]
pub fn import_surface_tree<R>(renderer: &mut R, surface: &WlSurface) -> Result<(), R::Error>
where
    R: Renderer + ImportAll,
    R::TextureId: 'static,
{
    let scale = 1.0;
    let location: Point<f64, Physical> = (0.0, 0.0).into();

    let mut result = Ok(());
    with_surface_tree_downward(
        surface,
        location,
        |_surface, states, location| {
            let mut location = *location;
            // Import a new buffer if necessary
            if let Err(err) = import_surface(renderer, states) {
                result = Err(err);
            }

            if let Some(data) = states.data_map.get::<RendererSurfaceStateUserData>() {
                let mut data_ref = data.lock().unwrap();
                let data = &mut *data_ref;
                // Now, should we be drawn ?
                if data.textures.contains_key(&renderer.context_id().erased()) {
                    // if yes, also process the children
                    let surface_view = data.surface_view.unwrap();
                    location += surface_view.offset.to_f64().to_physical(scale);
                    TraversalAction::DoChildren(location)
                } else {
                    // we are not displayed, so our children are neither
                    TraversalAction::SkipChildren
                }
            } else {
                // we are not displayed, so our children are neither
                TraversalAction::SkipChildren
            }
        },
        |_, _, _| {},
        |_, _, _| true,
    );
    result
}

/// Draws the render elements using a given [`Renderer`] and [`Frame`](crate::backend::renderer::Frame)
///
/// - `scale` needs to be equivalent to the fractional scale the rendered result should have.
/// - `location` is the position the surface should be drawn at.
/// - `damage` is the set of regions that should be drawn relative to the same origin as the location.
///
/// Note: This element will render nothing, if you are not using
/// [`crate::backend::renderer::utils::on_commit_buffer_handler`]
/// to let smithay handle buffer management.
#[instrument(level = "trace", skip(frame, scale, elements))]
#[profiling::function]
pub fn draw_render_elements<R, S, E>(
    frame: &mut R::Frame<'_, '_>,
    scale: S,
    elements: &[E],
    damage: &[Rectangle<i32, Physical>],
) -> Result<Option<Vec<Rectangle<i32, Physical>>>, R::Error>
where
    R: Renderer,
    R::TextureId: 'static,
    S: Into<Scale<f64>>,
    E: RenderElement<R>,
{
    let scale = scale.into();

    let mut render_elements: Vec<&E> = Vec::with_capacity(elements.len());
    let mut opaque_regions: Vec<Rectangle<i32, Physical>> = Vec::new();
    let mut render_damage: Vec<Rectangle<i32, Physical>> = Vec::with_capacity(damage.len());

    for element in elements {
        let element_geometry = element.geometry(scale);

        // Then test if the element is completely hidden behind opaque regions
        let is_hidden = element_geometry
            .subtract_rects(opaque_regions.iter().copied())
            .is_empty();

        if is_hidden {
            // No need to draw a completely hidden element
            continue;
        }

        render_damage.extend(Rectangle::subtract_rects_many(
            damage.iter().copied(),
            opaque_regions.iter().copied(),
        ));

        opaque_regions.extend(element.opaque_regions(scale).into_iter().map(|mut region| {
            region.loc += element_geometry.loc;
            region
        }));
        render_elements.insert(0, element);
    }

    // Optimize the damage for rendering
    render_damage.dedup();
    render_damage.retain(|rect| !rect.is_empty());
    // filter damage outside of the output gep and merge overlapping rectangles
    render_damage = render_damage
        .into_iter()
        .fold(Vec::new(), |new_damage, mut rect| {
            // replace with drain_filter, when that becomes stable to reuse the original Vec's memory
            let (overlapping, mut new_damage): (Vec<_>, Vec<_>) = new_damage
                .into_iter()
                .partition(|other| other.overlaps_or_touches(rect));

            for overlap in overlapping {
                rect = rect.merge(overlap);
            }
            new_damage.push(rect);
            new_damage
        });

    if render_damage.is_empty() {
        return Ok(None);
    }

    for element in render_elements.iter() {
        let element_geometry = element.geometry(scale);

        let element_damage = damage
            .iter()
            .filter_map(|d| d.intersection(element_geometry))
            .map(|mut d| {
                d.loc -= element_geometry.loc;
                d
            })
            .collect::<Vec<_>>();

        if element_damage.is_empty() {
            continue;
        }

        element.draw(frame, element.src(), element_geometry, &element_damage, &[], None)?;
    }

    Ok(Some(render_damage))
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "backend_drm")]
    use super::ReleasePointSlot;
    use super::{RendererSurfaceState, should_replace_renderer_buffer};
    use crate::backend::renderer::sync::SyncPoint;
    use crate::backend::renderer::{Color32F, ContextId, Frame, ImportAll, Renderer, RendererSuper, Texture};
    use crate::utils::{Buffer as BufferCoord, Physical, Rectangle, Size, Transform};
    use crate::wayland::compositor::{MultiCache, SurfaceData};
    #[cfg(feature = "backend_drm")]
    use crate::wayland::drm_syncobj::DrmSyncPoint;
    use std::{error::Error, fmt, sync::Mutex};
    use wayland_server::protocol::wl_buffer::WlBuffer;

    #[derive(Debug)]
    struct TestTexture(u32);

    impl Texture for TestTexture {
        fn width(&self) -> u32 {
            1
        }

        fn height(&self) -> u32 {
            1
        }

        fn format(&self) -> Option<crate::backend::allocator::Fourcc> {
            None
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum TestError {
        ReleaseFailed,
    }

    impl fmt::Display for TestError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                TestError::ReleaseFailed => f.write_str("release failed"),
            }
        }
    }

    impl Error for TestError {}

    #[derive(Debug)]
    struct TestFrame;

    impl Frame for TestFrame {
        type Error = TestError;
        type TextureId = TestTexture;

        fn context_id(&self) -> ContextId<Self::TextureId> {
            unreachable!()
        }

        fn clear(&mut self, _color: Color32F, _at: &[Rectangle<i32, Physical>]) -> Result<(), Self::Error> {
            unreachable!()
        }

        fn draw_solid(
            &mut self,
            _dst: Rectangle<i32, Physical>,
            _damage: &[Rectangle<i32, Physical>],
            _color: Color32F,
        ) -> Result<(), Self::Error> {
            unreachable!()
        }

        fn render_texture_from_to(
            &mut self,
            _texture: &Self::TextureId,
            _src: Rectangle<f64, BufferCoord>,
            _dst: Rectangle<i32, Physical>,
            _damage: &[Rectangle<i32, Physical>],
            _opaque_regions: &[Rectangle<i32, Physical>],
            _src_transform: Transform,
            _alpha: f32,
        ) -> Result<(), Self::Error> {
            unreachable!()
        }

        fn transformation(&self) -> Transform {
            unreachable!()
        }

        fn output_size(&self) -> Size<i32, Physical> {
            unreachable!()
        }

        fn wait(&mut self, _sync: &SyncPoint) -> Result<(), Self::Error> {
            unreachable!()
        }

        fn finish(self) -> Result<SyncPoint, Self::Error> {
            unreachable!()
        }
    }

    #[derive(Debug)]
    struct HookRenderer {
        context_id: ContextId<TestTexture>,
        fail_release: bool,
        released: Vec<u32>,
    }

    impl RendererSuper for HookRenderer {
        type Error = TestError;
        type TextureId = TestTexture;
        type Framebuffer<'buffer> = TestTexture;
        type Frame<'frame, 'buffer>
            = TestFrame
        where
            'buffer: 'frame,
            Self: 'frame;
    }

    impl Renderer for HookRenderer {
        fn context_id(&self) -> ContextId<Self::TextureId> {
            self.context_id.clone()
        }

        fn downscale_filter(
            &mut self,
            _filter: crate::backend::renderer::TextureFilter,
        ) -> Result<(), Self::Error> {
            unreachable!()
        }

        fn upscale_filter(
            &mut self,
            _filter: crate::backend::renderer::TextureFilter,
        ) -> Result<(), Self::Error> {
            unreachable!()
        }

        fn set_debug_flags(&mut self, _flags: crate::backend::renderer::DebugFlags) {
            unreachable!()
        }

        fn debug_flags(&self) -> crate::backend::renderer::DebugFlags {
            unreachable!()
        }

        fn render<'frame, 'buffer>(
            &'frame mut self,
            _framebuffer: &'frame mut Self::Framebuffer<'buffer>,
            _output_size: Size<i32, Physical>,
            _dst_transform: Transform,
        ) -> Result<Self::Frame<'frame, 'buffer>, Self::Error>
        where
            'buffer: 'frame,
        {
            unreachable!()
        }

        fn wait(&mut self, _sync: &SyncPoint) -> Result<(), Self::Error> {
            unreachable!()
        }

        fn release_imported_texture_for_surface_cache(
            &mut self,
            texture: &Self::TextureId,
        ) -> Result<(), Self::Error> {
            self.released.push(texture.0);
            if self.fail_release {
                Err(TestError::ReleaseFailed)
            } else {
                Ok(())
            }
        }
    }

    impl ImportAll for HookRenderer {
        fn import_buffer(
            &mut self,
            _buffer: &WlBuffer,
            _surface: Option<&SurfaceData>,
            _damage: &[Rectangle<i32, BufferCoord>],
        ) -> Option<Result<Self::TextureId, Self::Error>> {
            None
        }
    }

    #[test]
    fn explicit_sync_points_refresh_same_renderer_buffer() {
        assert!(!should_replace_renderer_buffer(false, false));
        assert!(should_replace_renderer_buffer(true, false));
        assert!(should_replace_renderer_buffer(false, true));
        assert!(should_replace_renderer_buffer(true, true));
    }

    #[test]
    fn retired_textures_are_drained_through_release_hook() {
        let context_id = ContextId::<TestTexture>::new();
        let mut state = RendererSurfaceState::default();
        state
            .textures
            .insert(context_id.erased(), Box::new(TestTexture(7)));

        state.retire_textures();
        assert!(state.textures.is_empty());
        assert!(state.texture(context_id.clone()).is_none());

        let mut released = Vec::new();
        state
            .drain_retired_textures(context_id.clone(), |texture: &TestTexture| {
                released.push(texture.0);
                Ok::<(), ()>(())
            })
            .unwrap();
        assert_eq!(released, vec![7]);

        state
            .drain_retired_textures(context_id, |texture: &TestTexture| {
                released.push(texture.0);
                Ok::<(), ()>(())
            })
            .unwrap();
        assert_eq!(released, vec![7]);
    }

    #[test]
    fn retired_texture_release_failure_keeps_texture_retired() {
        let context_id = ContextId::<TestTexture>::new();
        let mut state = RendererSurfaceState::default();
        state
            .textures
            .insert(context_id.erased(), Box::new(TestTexture(9)));
        state.retire_textures();
        state
            .textures
            .insert(context_id.erased(), Box::new(TestTexture(10)));
        state.retire_textures();

        let mut attempts = 0;
        assert_eq!(
            state.drain_retired_textures(context_id.clone(), |texture: &TestTexture| {
                attempts += 1;
                assert_eq!(texture.0, 9);
                Err("release failed")
            }),
            Err("release failed")
        );
        assert_eq!(attempts, 1);
        assert_eq!(
            state.retired_textures.get(&context_id.erased()).map(Vec::len),
            Some(2)
        );

        let mut released = Vec::new();
        state
            .drain_retired_textures(context_id.clone(), |texture: &TestTexture| {
                released.push(texture.0);
                Ok::<(), ()>(())
            })
            .unwrap();
        assert_eq!(released, vec![9, 10]);
        assert!(!state.retired_textures.contains_key(&context_id.erased()));
    }

    #[test]
    fn import_surface_drains_retired_textures_through_renderer_hook() {
        let context_id = ContextId::<TestTexture>::new();
        let mut state = RendererSurfaceState::default();
        state
            .textures
            .insert(context_id.erased(), Box::new(TestTexture(12)));
        state.retire_textures();

        let states = SurfaceData {
            role: None,
            data_map: Default::default(),
            cached_state: MultiCache::new(),
        };
        states.data_map.insert_if_missing_threadsafe(|| Mutex::new(state));

        let mut renderer = HookRenderer {
            context_id: context_id.clone(),
            fail_release: true,
            released: Vec::new(),
        };

        assert_eq!(
            super::import_surface(&mut renderer, &states),
            Err(TestError::ReleaseFailed)
        );
        assert_eq!(renderer.released, vec![12]);
        let retained = states
            .data_map
            .get::<super::RendererSurfaceStateUserData>()
            .unwrap()
            .lock()
            .unwrap()
            .retired_textures
            .get(&context_id.erased())
            .map(Vec::len);
        assert_eq!(retained, Some(1));

        renderer.fail_release = false;
        super::import_surface(&mut renderer, &states).unwrap();
        assert_eq!(renderer.released, vec![12, 12]);
        assert!(
            !states
                .data_map
                .get::<super::RendererSurfaceStateUserData>()
                .unwrap()
                .lock()
                .unwrap()
                .retired_textures
                .contains_key(&context_id.erased())
        );
    }

    #[cfg(feature = "backend_drm")]
    #[test]
    fn release_point_slot_take_transfers_drop_responsibility() {
        let slot = ReleasePointSlot::new(Some(DrmSyncPoint::invalid_for_tests(1).unwrap()));

        assert!(slot.get().is_some());
        let renderer_owned_release_point = slot.take_for_renderer().unwrap();
        assert!(slot.get().is_none());

        slot.signal_on_drop();
        assert!(renderer_owned_release_point.signal().is_err());
    }
}
