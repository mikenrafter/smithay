//! Native Vulkan renderer.
//!
//! This module provides a provisional, opt-in Vulkan renderer for an explicit [`PhysicalDevice`]. It
//! can upload sampled textures from CPU memory, render to in-memory/offscreen targets, read back
//! those offscreen targets through Smithay's public [`ImportMem`], [`Bind`], [`Offscreen`],
//! [`Renderer`], and [`ExportMem`] traits, and expose experimental public dmabuf render-target
//! development hooks. It remains intentionally incomplete and does not provide a complete compositor
//! renderer, HDR, colour-management policy, or a fully integrated presentation backend.
//!
//! Downstream compositors must not treat the presence of this module or the `renderer_vulkan`
//! feature as broad Vulkan rendering support. Real enablement must be added incrementally behind
//! explicit capability bits, with tests and stub failure paths before enabling working
//! functionality.
//!
//! Smithay's optional renderer traits are capability surfaces. The CPU-memory/offscreen path is the
//! most complete path. Dmabuf render-target support is intentionally public in this development fork,
//! but its Vulkan external-ownership and synchronization preconditions are explicit on
//! [`VulkanRenderer::bind_dmabuf_render_target`]. Generic safe [`Bind<Dmabuf>`] deliberately does
//! not acquire Vulkan dmabuf render targets, because a plain [`Dmabuf`] cannot carry the required
//! ownership/layout/synchronization proof. `ImportDma`, texture `ExportMem`, `ExportDma`, broad
//! explicit sync, blit/copy, and full presentation remain unsupported until their corresponding
//! capability bits can become true with coverage.
//!
//! Intended implementation order:
//!
//! 1. Vulkan instance/device initialisation
//! 2. extension discovery
//! 3. format/modifier query
//! 4. `ImportMem`
//! 5. basic texture upload
//! 6. offscreen target
//! 7. clear frame
//! 8. textured quad rendering
//! 9. alpha blending
//! 10. crop/scale/transform
//! 11. readback test path
//! 12. `ImportDma`
//! 13. dmabuf modifier handling
//! 14. export dmabuf
//! 15. KMS presentation path
//! 16. explicit sync
//! 17. blit/copy
//! 18. multi-GPU integration
//! 19. colour-capable render targets
//! 20. HDR-ready hooks
//!
//! Every future feature should follow this pattern: capability flag first, test second, stub
//! failure path third, real implementation fourth, enablement last.

use std::os::fd::{AsFd, OwnedFd};

#[cfg(all(
    feature = "wayland_frontend",
    feature = "backend_egl",
    feature = "use_system_lib"
))]
use crate::backend::renderer::ImportAll;
#[cfg(feature = "wayland_frontend")]
use crate::backend::renderer::{ImportDmaWl, ImportMemWl};
use crate::{
    backend::vulkan::PhysicalDevice,
    backend::{
        allocator::{Format, Fourcc, Modifier, dmabuf::Dmabuf, format::FormatSet},
        renderer::{
            Bind, Color32F, ContextId, DebugFlags, ExportMem, ImportDma, ImportMem, Offscreen,
            RenderTargetLifecycle, Renderer, RendererSuper, Texture, TextureFilter,
            sync::{Fence, Interrupted, SyncPoint},
        },
    },
    utils::{Buffer as BufferCoord, Physical, Rectangle, Size, Transform},
};
use ash::vk;
#[cfg(feature = "wayland_frontend")]
use wayland_server::protocol::wl_buffer;

mod capabilities;
mod device;
mod error;
pub mod format;
mod image;

pub use self::{
    capabilities::{
        VulkanColorCapabilities, VulkanDeviceCapabilities, VulkanExportCapabilities,
        VulkanExternalMemoryCapabilities, VulkanExternalSyncCapabilities, VulkanFormatCapabilities,
        VulkanFormatCapabilityRecord, VulkanFormatTiling, VulkanFormatUsage, VulkanImportCapabilities,
        VulkanRendererCapabilities, VulkanRenderingCapabilities, VulkanSyncCapabilities,
    },
    error::VulkanError,
    image::{VulkanFrame, VulkanMemoryMapping, VulkanRenderTarget, VulkanTexture},
};

/// Acquire options for binding a foreign dmabuf as a Vulkan render target.
///
/// This is the public development-fork API for the Vulkan-specific external target contract. It is
/// explicit because a plain [`Dmabuf`] does not encode Vulkan queue-family ownership, image layout,
/// or acquire synchronization.
#[derive(Debug, Clone, Copy)]
pub struct VulkanDmabufRenderTargetAcquire<'a> {
    /// Preserve previous target contents during acquire.
    ///
    /// If this is `true`, the foreign side must have released the image in
    /// `VK_IMAGE_LAYOUT_GENERAL`. If this is `false`, previous contents are discarded and the
    /// renderer will force an effective target age of zero through the generic bridge.
    pub preserve_contents: bool,
    /// Optional producer-completion dependency for the foreign release into Vulkan ownership.
    pub acquire_sync: Option<&'a SyncPoint>,
}

impl<'a> VulkanDmabufRenderTargetAcquire<'a> {
    /// Acquire for a full repaint, discarding previous contents.
    pub fn discard() -> Self {
        Self {
            preserve_contents: false,
            acquire_sync: None,
        }
    }

    /// Acquire while preserving previous contents after `acquire_sync` is satisfied.
    pub fn preserve(acquire_sync: Option<&'a SyncPoint>) -> Self {
        Self {
            preserve_contents: true,
            acquire_sync,
        }
    }
}

impl Default for VulkanDmabufRenderTargetAcquire<'_> {
    fn default() -> Self {
        Self::discard()
    }
}

/// Typed target wrapper for binding a foreign dmabuf through Vulkan's explicit acquire path.
///
/// This wrapper is the safe [`Bind`] target for Vulkan dmabuf render-target development. Creating it
/// is unsafe because the caller must prove the same external-memory ownership, layout, and acquire
/// synchronization requirements as [`VulkanRenderer::bind_dmabuf_render_target`]. Those requirements
/// are not a one-time construction check: they must be true each time the wrapper is bound. Once
/// constructed, generic render paths can bind it without smuggling those requirements through a plain
/// [`Dmabuf`].
#[derive(Debug)]
pub struct VulkanDmabufRenderTarget<'target, 'sync> {
    dmabuf: &'target mut Dmabuf,
    acquire: VulkanDmabufRenderTargetAcquire<'sync>,
}

impl<'target, 'sync> VulkanDmabufRenderTarget<'target, 'sync> {
    /// Creates a Vulkan dmabuf render-target wrapper with explicit acquire options.
    ///
    /// # Safety
    ///
    /// The caller must satisfy the safety requirements documented on
    /// [`VulkanRenderer::bind_dmabuf_render_target`] for `dmabuf` and `acquire` before every later
    /// safe bind of this wrapper. The same underlying dmabuf storage must not be bound through
    /// another alias while this wrapper, or a framebuffer created from it, is live.
    pub unsafe fn new(dmabuf: &'target mut Dmabuf, acquire: VulkanDmabufRenderTargetAcquire<'sync>) -> Self {
        Self { dmabuf, acquire }
    }

    /// Creates a discard/full-repaint target wrapper.
    ///
    /// # Safety
    ///
    /// The caller must satisfy the discard-acquire requirements documented on
    /// [`VulkanRenderer::bind_dmabuf_render_target`] before every later safe bind of this wrapper.
    pub unsafe fn discard(dmabuf: &'target mut Dmabuf) -> Self {
        // SAFETY: Forwarded to this constructor's caller.
        unsafe { Self::new(dmabuf, VulkanDmabufRenderTargetAcquire::discard()) }
    }

    /// Creates a preserve-content target wrapper.
    ///
    /// # Safety
    ///
    /// The caller must satisfy the preserve-acquire requirements documented on
    /// [`VulkanRenderer::bind_dmabuf_render_target`] before every later safe bind of this wrapper,
    /// including the foreign release in `VK_IMAGE_LAYOUT_GENERAL`.
    pub unsafe fn preserve(dmabuf: &'target mut Dmabuf, acquire_sync: Option<&'sync SyncPoint>) -> Self {
        // SAFETY: Forwarded to this constructor's caller.
        unsafe { Self::new(dmabuf, VulkanDmabufRenderTargetAcquire::preserve(acquire_sync)) }
    }

    /// Returns whether this acquire preserves previous contents.
    pub fn preserve_contents(&self) -> bool {
        self.acquire.preserve_contents
    }

    /// Returns the acquire options carried by this wrapper.
    pub fn acquire(&self) -> VulkanDmabufRenderTargetAcquire<'sync> {
        self.acquire
    }
}

use self::{
    device::{
        VulkanDeviceState, VulkanSyncFileSemaphore, image_copy_buffer_offset, tightly_packed_image_size,
    },
    format::{get_format_info, get_render_vk_format},
};

#[derive(Debug)]
struct VulkanSyncFileFence {
    fd: OwnedFd,
}

impl VulkanSyncFileFence {
    fn new(fd: OwnedFd) -> Self {
        Self { fd }
    }
}

impl Fence for VulkanSyncFileFence {
    fn is_signaled(&self) -> bool {
        let mut poll_fd = [rustix::event::PollFd::new(&self.fd, rustix::event::PollFlags::IN)];
        matches!(
            rustix::event::poll(
                &mut poll_fd,
                Some(&rustix::time::Timespec {
                    tv_sec: 0,
                    tv_nsec: 0,
                }),
            ),
            Ok(ready) if ready > 0
        )
    }

    fn wait(&self) -> Result<(), Interrupted> {
        let mut poll_fd = [rustix::event::PollFd::new(&self.fd, rustix::event::PollFlags::IN)];
        loop {
            match rustix::event::poll(&mut poll_fd, None) {
                Ok(ready) if ready > 0 => return Ok(()),
                Ok(_) => continue,
                Err(_) => return Err(Interrupted),
            }
        }
    }

    fn is_exportable(&self) -> bool {
        true
    }

    fn export(&self) -> Option<OwnedFd> {
        self.fd.as_fd().try_clone_to_owned().ok()
    }
}

fn sync_point_from_sync_file(sync_file: Option<OwnedFd>) -> SyncPoint {
    match sync_file {
        Some(fd) => SyncPoint::from(VulkanSyncFileFence::new(fd)),
        None => SyncPoint::signaled(),
    }
}

/// Provisional native Vulkan renderer for explicit-device in-memory/offscreen rendering.
#[derive(Debug)]
pub struct VulkanRenderer {
    context_id: ContextId<VulkanTexture>,
    debug_flags: DebugFlags,
    downscale_filter: TextureFilter,
    upscale_filter: TextureFilter,
    capabilities: VulkanRendererCapabilities,
    device: Option<VulkanDeviceState>,
}

/// Builder for explicit Vulkan renderer initialization.
///
/// The builder initializes a logical Vulkan device from an explicitly provided [`PhysicalDevice`].
/// Public operations remain limited to the capability bits advertised by the initialized renderer.
#[derive(Debug, Default, Clone)]
pub struct VulkanRendererBuilder {
    physical_device: Option<PhysicalDevice>,
}

impl VulkanRendererBuilder {
    /// Creates an empty Vulkan renderer builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the Vulkan physical device to use for renderer initialization.
    ///
    /// The renderer derives its Vulkan instance from this [`PhysicalDevice`], matching Smithay's
    /// backend ownership model and avoiding mismatched instance/device pairs.
    pub fn with_physical_device(mut self, physical_device: PhysicalDevice) -> Self {
        self.physical_device = Some(physical_device);
        self
    }

    /// Builds a Vulkan renderer.
    ///
    /// This returns [`VulkanError::VulkanUnavailable`] unless a [`PhysicalDevice`] was provided.
    pub fn build(self) -> Result<VulkanRenderer, VulkanError> {
        let physical_device = self.physical_device.ok_or(VulkanError::VulkanUnavailable)?;
        let device = VulkanDeviceState::new(physical_device)?;
        let capabilities = device.capabilities.clone();

        Ok(VulkanRenderer {
            context_id: ContextId::new(),
            debug_flags: DebugFlags::empty(),
            downscale_filter: TextureFilter::Linear,
            upscale_filter: TextureFilter::Linear,
            capabilities,
            device: Some(device),
        })
    }
}

impl VulkanRenderer {
    /// Attempts to create a Vulkan renderer.
    ///
    /// This returns [`VulkanError::VulkanUnavailable`] until a default device-selection policy exists.
    /// Use [`VulkanRenderer::builder`] with an explicit [`PhysicalDevice`] to initialize device state.
    pub fn new() -> Result<Self, VulkanError> {
        Self::builder().build()
    }

    /// Creates a builder for explicit Vulkan renderer initialization.
    pub fn builder() -> VulkanRendererBuilder {
        VulkanRendererBuilder::new()
    }

    /// Returns the uninitialized/default capabilities without constructing a renderer.
    ///
    /// All capability bits are false and all format/extension sets are empty until a renderer is
    /// initialized with an explicit [`PhysicalDevice`].
    pub fn scaffold_capabilities() -> VulkanRendererCapabilities {
        VulkanRendererCapabilities::default()
    }

    #[cfg(test)]
    fn new_scaffold_for_tests() -> Self {
        Self {
            context_id: ContextId::new(),
            debug_flags: DebugFlags::empty(),
            downscale_filter: TextureFilter::Linear,
            upscale_filter: TextureFilter::Linear,
            capabilities: VulkanRendererCapabilities::default(),
            device: None,
        }
    }

    /// Returns the discovered Vulkan renderer capabilities.
    ///
    /// Uninitialized test renderers report all capability bits as false and all format/extension
    /// sets as empty. Renderers built with an explicit [`PhysicalDevice`] report discovered device
    /// and format capabilities.
    pub fn capabilities(&self) -> &VulkanRendererCapabilities {
        &self.capabilities
    }

    /// Returns whether Vulkan device state was initialized.
    ///
    /// This does not imply that rendering, import, export, or presentation operations are supported.
    pub fn is_device_initialized(&self) -> bool {
        self.device.is_some() && self.capabilities.device.available
    }

    fn render_target_formats(&self) -> FormatSet {
        self.capabilities.formats.render_target_formats()
    }

    fn render_target_format_supported(&self, format: Fourcc) -> bool {
        self.render_target_formats().contains(&Format {
            code: format,
            modifier: Modifier::Invalid,
        })
    }

    #[allow(dead_code)]
    fn create_offscreen_render_target(
        &mut self,
        format: Fourcc,
        size: Size<i32, BufferCoord>,
    ) -> Result<VulkanRenderTarget<'static>, VulkanError> {
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let vk_format = get_render_vk_format(format)?;
        let extent = extent_from_size(size, "offscreen render target size")?;
        let color_image = device.create_offscreen_color_image(extent, vk_format)?;

        Ok(VulkanRenderTarget::from_offscreen_image(
            self.context_id.clone(),
            size,
            format,
            color_image,
        ))
    }

    #[allow(dead_code)]
    fn create_imported_dmabuf_texture(
        &mut self,
        dmabuf: &Dmabuf,
    ) -> Result<Option<VulkanTexture>, VulkanError> {
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let import = image::VulkanDmabufImportState::from_dmabuf(dmabuf)?;
        let Some(sampled_image) = device.create_dmabuf_sampled_image_resources(
            dmabuf,
            self.downscale_filter,
            self.upscale_filter,
        )?
        else {
            return Ok(None);
        };

        Ok(Some(VulkanTexture::from_dmabuf_sampled_image(
            self.context_id.clone(),
            &import,
            sampled_image,
        )))
    }

    /// Import a dmabuf as a sampled texture when the producer's Vulkan external state is known.
    ///
    /// # Safety
    ///
    /// The caller must ensure the dmabuf producer released the image to
    /// `VK_QUEUE_FAMILY_FOREIGN_EXT` in `VK_IMAGE_LAYOUT_GENERAL`, and that `acquire_semaphore`, if
    /// present, represents the producer's completion dependency for that release. If no semaphore is
    /// supplied, the caller must ensure the producer's writes and ownership release are already
    /// complete and visible to this renderer's Vulkan queue submission.
    #[allow(dead_code)]
    unsafe fn create_imported_dmabuf_texture_with_known_general_layout(
        &mut self,
        dmabuf: &Dmabuf,
        acquire_semaphore: Option<&VulkanSyncFileSemaphore>,
    ) -> Result<Option<VulkanTexture>, VulkanError> {
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let import = image::VulkanDmabufImportState::from_dmabuf(dmabuf)?;
        let Some(sampled_image) = (unsafe {
            // SAFETY: Forwarded from this method's caller.
            device.create_acquired_dmabuf_sampled_image_resources_with_known_general_layout(
                dmabuf,
                self.downscale_filter,
                self.upscale_filter,
                acquire_semaphore,
            )
        })?
        else {
            return Ok(None);
        };

        Ok(Some(VulkanTexture::from_acquired_dmabuf_sampled_image(
            self.context_id.clone(),
            &import,
            sampled_image,
        )))
    }

    /// Import a known-layout dmabuf using a Smithay sync point as the optional acquire dependency.
    ///
    /// # Safety
    ///
    /// The caller must ensure the dmabuf producer released the image to
    /// `VK_QUEUE_FAMILY_FOREIGN_EXT` in `VK_IMAGE_LAYOUT_GENERAL`. If `acquire_sync` is `Some`, it
    /// must represent the producer's completion dependency for that release and signal only after
    /// the producer's writes and ownership release for this dmabuf are complete. If `acquire_sync`
    /// is `None`, the caller must ensure those writes and ownership release are already complete and
    /// visible to this renderer's Vulkan queue submission. If `acquire_sync` exports a fence fd and
    /// this device supports sync-file import, that fd must be a valid Linux sync-file fd suitable for
    /// `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT`.
    #[allow(dead_code)]
    unsafe fn create_imported_dmabuf_texture_with_known_general_layout_and_sync_point(
        &mut self,
        dmabuf: &Dmabuf,
        acquire_sync: Option<&SyncPoint>,
    ) -> Result<Option<VulkanTexture>, VulkanError> {
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let import = image::VulkanDmabufImportState::from_dmabuf(dmabuf)?;
        let Some(sampled_image) = (unsafe {
            // SAFETY: Forwarded from this method's caller.
            device.create_acquired_dmabuf_sampled_image_resources_with_known_general_layout_and_sync_point(
                dmabuf,
                self.downscale_filter,
                self.upscale_filter,
                acquire_sync,
            )
        })?
        else {
            return Ok(None);
        };

        Ok(Some(VulkanTexture::from_acquired_dmabuf_sampled_image(
            self.context_id.clone(),
            &import,
            sampled_image,
        )))
    }

    /// Import a dmabuf as a render target and acquire it for color-attachment rendering.
    ///
    /// This is the implementation helper behind the explicit public dmabuf render-target API.
    ///
    /// # Safety
    ///
    /// The caller must ensure the foreign producer has released ownership to
    /// `VK_QUEUE_FAMILY_FOREIGN_EXT` before this acquire is submitted. If `preserve_contents` is
    /// true, the producer must have released the image in `VK_IMAGE_LAYOUT_GENERAL`. If
    /// `acquire_semaphore` is present, it must signal only after the producer's writes and ownership
    /// release complete. If it is absent, those operations must already be complete and visible to
    /// this renderer's Vulkan queue submission. If `preserve_contents` is false, previous contents
    /// are discarded.
    #[allow(dead_code)]
    pub(crate) unsafe fn create_acquired_dmabuf_render_target(
        &mut self,
        dmabuf: &Dmabuf,
        preserve_contents: bool,
        acquire_semaphore: Option<&VulkanSyncFileSemaphore>,
    ) -> Result<Option<VulkanRenderTarget<'static>>, VulkanError> {
        let import = validate_dmabuf_render_target_metadata(dmabuf)?;
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let Some(color_image) = (unsafe {
            // SAFETY: Forwarded from this method's caller.
            device.create_acquired_dmabuf_render_target_image(dmabuf, preserve_contents, acquire_semaphore)
        })?
        else {
            return Ok(None);
        };

        Ok(Some(VulkanRenderTarget::from_acquired_dmabuf_render_target(
            self.context_id.clone(),
            &import,
            color_image,
        )))
    }

    /// Import a dmabuf as an internal render target and acquire it using a Smithay sync point as the
    /// optional producer-completion dependency.
    ///
    /// This is a sync-point convenience wrapper for the acquired dmabuf render-target path used by
    /// the explicit public dmabuf render-target API.
    ///
    /// # Safety
    ///
    /// The caller must ensure the foreign producer has released ownership to
    /// `VK_QUEUE_FAMILY_FOREIGN_EXT` before this acquire is submitted. If `preserve_contents` is
    /// true, the producer must have released the image in `VK_IMAGE_LAYOUT_GENERAL`. If
    /// `acquire_sync` is present, it must represent the producer's completion dependency for that
    /// release and signal only after the producer's writes and ownership release for this dmabuf are
    /// complete. If it is absent, those operations must already be complete and visible to this
    /// renderer's Vulkan queue submission. If `acquire_sync` exports a fence fd and this device
    /// supports sync-file import, that fd must be a valid Linux sync-file fd suitable for
    /// `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT`. If `preserve_contents` is false, previous
    /// contents are discarded.
    #[allow(dead_code)]
    pub(crate) unsafe fn create_acquired_dmabuf_render_target_with_sync_point<'target>(
        &mut self,
        dmabuf: &'target Dmabuf,
        preserve_contents: bool,
        acquire_sync: Option<&SyncPoint>,
    ) -> Result<Option<VulkanRenderTarget<'target>>, VulkanError> {
        let import = validate_dmabuf_render_target_metadata(dmabuf)?;
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let Some(color_image) = (unsafe {
            // SAFETY: Forwarded from this method's caller.
            device.create_acquired_dmabuf_render_target_image_with_sync_point(
                dmabuf,
                preserve_contents,
                acquire_sync,
            )
        })?
        else {
            return Ok(None);
        };

        Ok(Some(VulkanRenderTarget::from_acquired_dmabuf_render_target(
            self.context_id.clone(),
            &import,
            color_image,
        )))
    }

    /// Bind a foreign dmabuf as a Vulkan render target using explicit Vulkan acquire semantics.
    ///
    /// This method is intentionally public in this development fork. It is the correct Vulkan-shaped
    /// entry point for callers that can reason about external-memory ownership, layout, and acquire
    /// synchronization. Generic safe [`Bind<Dmabuf>`] intentionally does not delegate here; callers
    /// that use this method are opting into the explicit unsafe external-memory contract.
    ///
    /// # Safety
    ///
    /// The caller must ensure the foreign producer has released ownership to
    /// `VK_QUEUE_FAMILY_FOREIGN_EXT` before this acquire is submitted. If
    /// `acquire.preserve_contents` is true, the producer must have released the image in
    /// `VK_IMAGE_LAYOUT_GENERAL`. If `acquire.acquire_sync` is present, it must represent the
    /// producer's completion dependency for that release and signal only after the producer's writes
    /// and ownership release for this dmabuf are complete. If it is absent, those operations must
    /// already be complete and visible to this renderer's Vulkan queue submission. If the sync point
    /// exports a fence fd and this device supports sync-file import, that fd must be a valid Linux
    /// sync-file fd suitable for `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT`. If
    /// `preserve_contents` is false, previous contents are discarded. The caller must not bind the
    /// same underlying dmabuf storage through another alias while the returned target is acquired.
    /// After a successful acquire, renderers must either finish a frame for the returned target or
    /// call [`VulkanRenderer::release_dmabuf_render_target_after_render_error`] before reusing the
    /// dmabuf externally; dropping the target alone does not release Vulkan ownership back to the
    /// foreign queue family.
    pub unsafe fn bind_dmabuf_render_target<'target>(
        &mut self,
        dmabuf: &'target mut Dmabuf,
        acquire: VulkanDmabufRenderTargetAcquire<'_>,
    ) -> Result<Option<VulkanRenderTarget<'target>>, VulkanError> {
        // SAFETY: Forwarded from this public unsafe method's caller.
        unsafe {
            self.create_acquired_dmabuf_render_target_with_sync_point(
                dmabuf,
                acquire.preserve_contents,
                acquire.acquire_sync,
            )
        }
    }

    /// Release an acquired dmabuf texture back to foreign ownership in `VK_IMAGE_LAYOUT_GENERAL`.
    ///
    /// This is an internal counterpart to the known-layout acquire helpers. It does not make public
    /// dmabuf import/export supported; callers must only pass textures created by the acquired
    /// dmabuf import path for this renderer.
    #[allow(dead_code)]
    fn release_imported_dmabuf_texture_to_foreign_general(
        &mut self,
        texture: &VulkanTexture,
        export_sync_file: bool,
    ) -> Result<(bool, Option<OwnedFd>), VulkanError> {
        if texture.context_id != self.context_id {
            return Err(VulkanError::UnsupportedOperation("foreign dmabuf texture"));
        }
        if texture.image.source != image::VulkanImageSource::DmabufImport {
            return Err(VulkanError::UnsupportedOperation("dmabuf texture"));
        }
        let sampled_image = texture
            .sampled_image
            .as_ref()
            .ok_or(VulkanError::UnsupportedOperation("dmabuf texture sampled image"))?;
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;

        device.release_sampled_dmabuf_to_foreign_general(sampled_image.image(), export_sync_file)
    }

    /// Release an acquired dmabuf render target back to foreign ownership in `GENERAL` layout.
    ///
    /// This is the release counterpart to the acquired dmabuf render-target helper used by public
    /// dmabuf render-target binding.
    #[allow(dead_code)]
    pub(crate) fn release_acquired_dmabuf_render_target_to_foreign_general(
        &mut self,
        target: &mut VulkanRenderTarget<'_>,
        export_sync_file: bool,
    ) -> Result<(bool, Option<OwnedFd>), VulkanError> {
        if target.context_id != self.context_id {
            return Err(VulkanError::UnsupportedOperation("foreign dmabuf render target"));
        }
        if target.image.source != image::VulkanImageSource::RenderTarget {
            return Err(VulkanError::UnsupportedOperation("dmabuf render target"));
        }
        if !target.image.sync.is_locally_usable() {
            return Err(VulkanError::UnsupportedOperation("dmabuf import synchronization"));
        }
        let color_image = target
            .color_image
            .as_ref()
            .ok_or(VulkanError::UnsupportedOperation("dmabuf render target image"))?;
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let release =
            device.release_dmabuf_render_target_to_foreign_general(color_image, export_sync_file)?;
        if release.0 {
            // The foreign side now owns the image in GENERAL. Keep the renderer-facing layout
            // unusable until a later explicit acquire restores local color-attachment ownership.
            target.image.layout = image::VulkanImageLayoutState::Undefined;
            target.image.sync = image::VulkanImageSyncState::foreign_known_general_for_dmabuf_import();
        }

        Ok(release)
    }

    /// Release an acquired dmabuf render target and return the exported release fence as a
    /// [`SyncPoint`] when available.
    ///
    /// This is a crate-private helper for public dmabuf render-target binding and DRM error
    /// cleanup.
    #[allow(dead_code)]
    pub(crate) fn release_acquired_dmabuf_render_target_to_foreign_general_sync_point(
        &mut self,
        target: &mut VulkanRenderTarget<'_>,
        export_sync_file: bool,
    ) -> Result<(bool, SyncPoint), VulkanError> {
        let (released, sync_file) =
            self.release_acquired_dmabuf_render_target_to_foreign_general(target, export_sync_file)?;
        Ok((released, sync_point_from_sync_file(sync_file)))
    }

    /// Release an explicitly acquired dmabuf render target after rendering failed before frame finish.
    ///
    /// This is the public error-cleanup counterpart to
    /// [`VulkanRenderer::bind_dmabuf_render_target`]. It returns the target to foreign ownership in
    /// `VK_IMAGE_LAYOUT_GENERAL` without exporting a release fence. Call this before reusing the
    /// dmabuf externally when rendering fails before
    /// [`Frame::finish`](crate::backend::renderer::Frame::finish).
    pub fn release_dmabuf_render_target_after_render_error(
        &mut self,
        target: &mut VulkanRenderTarget<'_>,
    ) -> Result<(), VulkanError> {
        self.release_acquired_dmabuf_render_target_to_foreign_general_sync_point(target, false)
            .map(|_| ())
    }

    #[allow(dead_code)]
    fn clear_offscreen_render_target(
        &mut self,
        target: &mut VulkanRenderTarget<'_>,
        color: Color32F,
    ) -> Result<(), VulkanError> {
        if target.context_id != self.context_id {
            return Err(VulkanError::UnsupportedOperation("foreign offscreen target"));
        }
        if target.image.source != image::VulkanImageSource::Offscreen {
            return Err(VulkanError::UnsupportedOperation("offscreen target"));
        }
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let color_image = target
            .color_image
            .as_ref()
            .ok_or(VulkanError::UnsupportedOperation("offscreen target image"))?;
        let format = target
            .image
            .format
            .ok_or(VulkanError::UnsupportedOperation("offscreen target format"))?;

        device.clear_offscreen_color_image(color_image, clear_color_value_for_format(format, color)?)?;
        target.image.layout = image::VulkanImageLayoutState::TransferDst;
        Ok(())
    }

    #[allow(dead_code)]
    fn read_offscreen_render_target(
        &mut self,
        target: &mut VulkanRenderTarget<'_>,
    ) -> Result<Vec<u8>, VulkanError> {
        if target.context_id != self.context_id {
            return Err(VulkanError::UnsupportedOperation("foreign offscreen target"));
        }
        if target.image.source != image::VulkanImageSource::Offscreen {
            return Err(VulkanError::UnsupportedOperation("offscreen target"));
        }
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let color_image = target
            .color_image
            .as_ref()
            .ok_or(VulkanError::UnsupportedOperation("offscreen target image"))?;

        let data = device.read_image_to_tightly_packed_buffer(color_image)?;
        target.image.layout = image::VulkanImageLayoutState::TransferSrc;
        Ok(data)
    }
}

impl RendererSuper for VulkanRenderer {
    type Error = VulkanError;
    type TextureId = VulkanTexture;
    type Framebuffer<'buffer> = VulkanRenderTarget<'buffer>;
    type Frame<'frame, 'buffer>
        = VulkanFrame<'frame, 'buffer>
    where
        'buffer: 'frame,
        Self: 'frame;
}

impl Renderer for VulkanRenderer {
    fn context_id(&self) -> ContextId<Self::TextureId> {
        self.context_id.clone()
    }

    fn downscale_filter(&mut self, filter: TextureFilter) -> Result<(), Self::Error> {
        self.downscale_filter = filter;
        Ok(())
    }

    fn upscale_filter(&mut self, filter: TextureFilter) -> Result<(), Self::Error> {
        self.upscale_filter = filter;
        Ok(())
    }

    fn set_debug_flags(&mut self, flags: DebugFlags) {
        self.debug_flags = flags;
    }

    fn debug_flags(&self) -> DebugFlags {
        self.debug_flags
    }

    fn render<'frame, 'buffer>(
        &'frame mut self,
        framebuffer: &'frame mut Self::Framebuffer<'buffer>,
        output_size: Size<i32, Physical>,
        dst_transform: Transform,
    ) -> Result<Self::Frame<'frame, 'buffer>, Self::Error>
    where
        'buffer: 'frame,
    {
        if framebuffer.context_id != self.context_id {
            return Err(VulkanError::UnsupportedOperation("foreign render target"));
        }
        if !matches!(
            framebuffer.image.source,
            image::VulkanImageSource::Offscreen | image::VulkanImageSource::RenderTarget
        ) {
            return Err(VulkanError::UnsupportedOperation("render target"));
        }
        if output_size.w <= 0 || output_size.h <= 0 {
            return Err(VulkanError::UnsupportedOperation("frame size"));
        }
        if framebuffer.image.size.w != output_size.w || framebuffer.image.size.h != output_size.h {
            return Err(VulkanError::UnsupportedOperation("frame size"));
        }
        if framebuffer.image.source == image::VulkanImageSource::RenderTarget
            && !framebuffer.image.sync.is_locally_usable()
        {
            return Err(VulkanError::UnsupportedOperation("dmabuf import synchronization"));
        }
        if framebuffer.color_image.is_none() {
            return Err(VulkanError::UnsupportedOperation("render target image"));
        }
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;

        Ok(VulkanFrame {
            context_id: self.context_id.clone(),
            output_size,
            transform: dst_transform,
            device: Some(device),
            target: Some(framebuffer),
            _renderer: std::marker::PhantomData,
        })
    }

    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        sync.wait().map_err(|_| VulkanError::SyncInterrupted)
    }

    fn cleanup_texture_cache(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl<'target> Bind<VulkanRenderTarget<'target>> for VulkanRenderer {
    fn bind<'a>(
        &mut self,
        target: &'a mut VulkanRenderTarget<'target>,
    ) -> Result<Self::Framebuffer<'a>, Self::Error> {
        if target.context_id != self.context_id {
            return Err(VulkanError::UnsupportedOperation("foreign render target"));
        }
        if target.image.source != image::VulkanImageSource::Offscreen {
            return Err(VulkanError::UnsupportedOperation("render target"));
        }
        if target.color_image.is_none() {
            return Err(VulkanError::UnsupportedOperation("render target image"));
        }

        Ok(VulkanRenderTarget {
            context_id: target.context_id.clone(),
            image: target.image.clone(),
            color_image: target.color_image.clone(),
            _target: std::marker::PhantomData,
        })
    }

    fn supported_formats(&self) -> Option<FormatSet> {
        Some(self.render_target_formats())
    }
}

impl<'target, 'sync> Bind<VulkanDmabufRenderTarget<'target, 'sync>> for VulkanRenderer {
    fn bind<'a>(
        &mut self,
        target: &'a mut VulkanDmabufRenderTarget<'target, 'sync>,
    ) -> Result<Self::Framebuffer<'a>, Self::Error> {
        unsafe {
            // SAFETY: `VulkanDmabufRenderTarget` can only be constructed by callers that accepted
            // and upheld the explicit Vulkan external-memory acquire contract. Reborrow the wrapped
            // dmabuf for the lifetime of the returned framebuffer so Rust prevents rebinding through
            // this wrapper while the framebuffer is live.
            self.bind_dmabuf_render_target(&mut *target.dmabuf, target.acquire)
        }?
        .ok_or(VulkanError::UnsupportedOperation("dmabuf render target format"))
    }

    fn supported_formats(&self) -> Option<FormatSet> {
        Some(self.capabilities.formats.dmabuf_render_target.clone())
    }
}

impl<'target, 'sync> RenderTargetLifecycle<VulkanDmabufRenderTarget<'target, 'sync>> for VulkanRenderer {
    fn target_age(&self, target: &VulkanDmabufRenderTarget<'target, 'sync>, age: usize) -> usize {
        if target.preserve_contents() { age } else { 0 }
    }

    fn release_after_render_error(&mut self, target: &mut Self::Framebuffer<'_>) -> Result<(), Self::Error> {
        self.release_dmabuf_render_target_after_render_error(target)
    }
}

impl Bind<Dmabuf> for VulkanRenderer {
    fn bind<'a>(&mut self, _target: &'a mut Dmabuf) -> Result<Self::Framebuffer<'a>, Self::Error> {
        Err(VulkanError::UnsupportedOperation(
            "dmabuf render target requires explicit Vulkan acquire",
        ))
    }

    fn supported_formats(&self) -> Option<FormatSet> {
        Some(FormatSet::default())
    }
}

impl RenderTargetLifecycle<Dmabuf> for VulkanRenderer {}

fn validate_dmabuf_render_target_metadata(
    target: &Dmabuf,
) -> Result<image::VulkanDmabufImportState, VulkanError> {
    let import = image::VulkanDmabufImportState::from_dmabuf(target)?;
    if import.plane_count() != 1 {
        return Err(VulkanError::UnsupportedOperation("dmabuf render target planes"));
    }

    Ok(import)
}

impl Offscreen<VulkanRenderTarget<'static>> for VulkanRenderer {
    fn create_buffer(
        &mut self,
        format: Fourcc,
        size: Size<i32, BufferCoord>,
    ) -> Result<VulkanRenderTarget<'static>, Self::Error> {
        if !self.render_target_format_supported(format) {
            return Err(VulkanError::UnsupportedFormat(format));
        }

        self.create_offscreen_render_target(format, size)
    }
}

impl ImportDma for VulkanRenderer {
    fn dmabuf_formats(&self) -> FormatSet {
        self.capabilities.formats.dmabuf_import.clone()
    }

    fn import_dmabuf(
        &mut self,
        _dmabuf: &Dmabuf,
        _damage: Option<&[Rectangle<i32, BufferCoord>]>,
    ) -> Result<Self::TextureId, Self::Error> {
        Err(VulkanError::UnsupportedOperation("dmabuf import"))
    }
}

#[cfg(feature = "wayland_frontend")]
impl ImportDmaWl for VulkanRenderer {}

#[cfg(all(
    feature = "wayland_frontend",
    feature = "backend_egl",
    feature = "use_system_lib"
))]
impl ImportAll for VulkanRenderer {
    fn import_buffer(
        &mut self,
        buffer: &wl_buffer::WlBuffer,
        surface: Option<&crate::wayland::compositor::SurfaceData>,
        damage: &[Rectangle<i32, BufferCoord>],
    ) -> Option<Result<Self::TextureId, Self::Error>> {
        super::import_shm_dmabuf_buffer(self, buffer, surface, damage)
    }

    fn import_buffer_from_surface_state(
        &mut self,
        buffer: &super::utils::Buffer,
        surface: Option<&crate::wayland::compositor::SurfaceData>,
        damage: &[Rectangle<i32, BufferCoord>],
    ) -> Option<Result<Self::TextureId, Self::Error>> {
        super::import_shm_dmabuf_buffer_from_surface_state(self, buffer, surface, damage)
    }
}

impl ImportMem for VulkanRenderer {
    fn import_memory(
        &mut self,
        data: &[u8],
        format: Fourcc,
        size: Size<i32, BufferCoord>,
        flipped: bool,
    ) -> Result<Self::TextureId, Self::Error> {
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let memory_format = Format {
            code: format,
            modifier: Modifier::Invalid,
        };
        if !self.capabilities.formats.memory_import.contains(&memory_format) {
            return Err(VulkanError::UnsupportedFormat(format));
        }

        let vk_format = get_render_vk_format(format)?;
        let extent = extent_from_size(size, "memory import size")?;
        let required_size = tightly_packed_image_size(vk_format, extent)?;
        if (data.len() as u64) < required_size {
            return Err(VulkanError::UnsupportedOperation("memory import data"));
        }
        let required_len = usize::try_from(required_size)
            .map_err(|_| VulkanError::UnsupportedOperation("memory import data size"))?;

        let sampled_image = device.create_uploaded_sampled_image(
            extent,
            vk_format,
            &data[..required_len],
            self.downscale_filter,
            self.upscale_filter,
        )?;

        Ok(VulkanTexture::from_sampled_image(
            self.context_id.clone(),
            size,
            format,
            sampled_image,
            flipped,
        ))
    }

    fn update_memory(
        &mut self,
        texture: &Self::TextureId,
        data: &[u8],
        region: Rectangle<i32, BufferCoord>,
    ) -> Result<(), Self::Error> {
        if texture.context_id != self.context_id {
            return Err(VulkanError::UnsupportedOperation("foreign memory texture"));
        }
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        if texture.image.source != image::VulkanImageSource::MemoryUpload {
            return Err(VulkanError::UnsupportedOperation("memory texture"));
        }
        let sampled_image = texture
            .sampled_image
            .as_ref()
            .ok_or(VulkanError::UnsupportedOperation("memory texture"))?;
        let format = texture
            .image
            .format
            .ok_or(VulkanError::UnsupportedOperation("memory texture format"))?;
        let vk_format = get_render_vk_format(format)?;
        let texture_extent = extent_from_size(texture.image.size, "memory update size")?;
        let (image_offset, region_extent) = update_region_to_vk(texture.image.size, region)?;
        let required_size = tightly_packed_image_size(vk_format, texture_extent)?;
        if (data.len() as u64) < required_size {
            return Err(VulkanError::UnsupportedOperation("memory update data"));
        }
        let required_len = usize::try_from(required_size)
            .map_err(|_| VulkanError::UnsupportedOperation("memory update data size"))?;
        let buffer_offset = image_copy_buffer_offset(
            vk_format,
            texture_extent.width,
            image_offset.x as u32,
            image_offset.y as u32,
        )?;

        device.update_uploaded_image_region(
            sampled_image.image(),
            &data[..required_len],
            buffer_offset,
            texture_extent.width,
            texture_extent.height,
            image_offset,
            region_extent,
        )
    }

    fn mem_formats(&self) -> Box<dyn Iterator<Item = Fourcc>> {
        Box::new(
            self.capabilities
                .formats
                .memory_import
                .iter()
                .map(|format| format.code)
                .collect::<Vec<_>>()
                .into_iter(),
        )
    }
}

#[cfg(feature = "wayland_frontend")]
impl ImportMemWl for VulkanRenderer {
    fn import_shm_buffer(
        &mut self,
        buffer: &wl_buffer::WlBuffer,
        _surface: Option<&crate::wayland::compositor::SurfaceData>,
        _damage: &[Rectangle<i32, BufferCoord>],
    ) -> Result<Self::TextureId, Self::Error> {
        let texture = crate::wayland::shm::with_buffer_contents(buffer, |ptr, len, data| {
            let fourcc = crate::wayland::shm::shm_format_to_fourcc(data.format)
                .ok_or(VulkanError::UnsupportedOperation("wl_shm format"))?;
            let bits_per_pixel = crate::backend::allocator::format::get_bpp(fourcc)
                .ok_or(VulkanError::UnsupportedFormat(fourcc))?;
            if bits_per_pixel % 8 != 0 {
                return Err(VulkanError::UnsupportedFormat(fourcc));
            }
            let bytes_per_pixel = bits_per_pixel / 8;
            let packed = copy_shm_buffer_to_tightly_packed(
                data.offset,
                data.width,
                data.height,
                data.stride,
                bytes_per_pixel,
                ptr,
                len,
            )?;

            self.import_memory(&packed, fourcc, (data.width, data.height).into(), false)
        })
        .map_err(|_| VulkanError::UnsupportedOperation("wl_shm buffer"))??;

        Ok(texture)
    }
}

impl ExportMem for VulkanRenderer {
    type TextureMapping = VulkanMemoryMapping;

    fn copy_framebuffer(
        &mut self,
        target: &Self::Framebuffer<'_>,
        region: Rectangle<i32, BufferCoord>,
        format: Fourcc,
    ) -> Result<Self::TextureMapping, Self::Error> {
        if target.context_id != self.context_id {
            return Err(VulkanError::UnsupportedOperation("foreign render target"));
        }
        if target.image.source != image::VulkanImageSource::Offscreen {
            return Err(VulkanError::UnsupportedOperation("render target"));
        }
        let target_format = target
            .format()
            .ok_or(VulkanError::UnsupportedOperation("render target format"))?;
        if format != target_format {
            return Err(VulkanError::UnsupportedFormat(format));
        }

        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let color_image = target
            .color_image
            .as_ref()
            .ok_or(VulkanError::UnsupportedOperation("render target image"))?;
        let (image_offset, extent) = image_region_to_vk(target.size(), region, "framebuffer copy region")?;
        let data = device.read_image_region_to_tightly_packed_buffer(color_image, image_offset, extent)?;

        Ok(VulkanMemoryMapping {
            data,
            size: region.size,
            format,
            flipped: false,
        })
    }

    fn copy_texture(
        &mut self,
        texture: &Self::TextureId,
        _region: Rectangle<i32, BufferCoord>,
        _format: Fourcc,
    ) -> Result<Self::TextureMapping, Self::Error> {
        if texture.context_id != self.context_id {
            return Err(VulkanError::UnsupportedOperation("foreign memory texture"));
        }

        Err(VulkanError::UnsupportedOperation("texture memory export"))
    }

    fn can_read_texture(&mut self, texture: &Self::TextureId) -> Result<bool, Self::Error> {
        if texture.context_id != self.context_id {
            return Err(VulkanError::UnsupportedOperation("foreign memory texture"));
        }

        Ok(false)
    }

    fn map_texture<'a>(
        &mut self,
        texture_mapping: &'a Self::TextureMapping,
    ) -> Result<&'a [u8], Self::Error> {
        Ok(&texture_mapping.data)
    }
}

fn extent_from_size(size: Size<i32, BufferCoord>, error: &'static str) -> Result<vk::Extent3D, VulkanError> {
    if size.w <= 0 || size.h <= 0 {
        return Err(VulkanError::UnsupportedOperation(error));
    }

    Ok(vk::Extent3D {
        width: size
            .w
            .try_into()
            .map_err(|_| VulkanError::UnsupportedOperation(error))?,
        height: size
            .h
            .try_into()
            .map_err(|_| VulkanError::UnsupportedOperation(error))?,
        depth: 1,
    })
}

fn image_region_to_vk(
    image_size: Size<i32, BufferCoord>,
    region: Rectangle<i32, BufferCoord>,
    error: &'static str,
) -> Result<(vk::Offset3D, vk::Extent3D), VulkanError> {
    if region.loc.x < 0 || region.loc.y < 0 || region.size.w <= 0 || region.size.h <= 0 {
        return Err(VulkanError::UnsupportedOperation(error));
    }
    if region
        .loc
        .x
        .checked_add(region.size.w)
        .is_none_or(|right| right > image_size.w)
        || region
            .loc
            .y
            .checked_add(region.size.h)
            .is_none_or(|bottom| bottom > image_size.h)
    {
        return Err(VulkanError::UnsupportedOperation(error));
    }

    Ok((
        vk::Offset3D {
            x: region.loc.x,
            y: region.loc.y,
            z: 0,
        },
        vk::Extent3D {
            width: region
                .size
                .w
                .try_into()
                .map_err(|_| VulkanError::UnsupportedOperation(error))?,
            height: region
                .size
                .h
                .try_into()
                .map_err(|_| VulkanError::UnsupportedOperation(error))?,
            depth: 1,
        },
    ))
}

#[allow(dead_code)]
fn copy_shm_buffer_to_tightly_packed(
    offset: i32,
    width: i32,
    height: i32,
    stride: i32,
    bytes_per_pixel: usize,
    ptr: *const u8,
    len: usize,
) -> Result<Vec<u8>, VulkanError> {
    if offset < 0 || width <= 0 || height <= 0 || stride < 0 || bytes_per_pixel == 0 {
        return Err(VulkanError::UnsupportedOperation("wl_shm buffer layout"));
    }
    let offset =
        usize::try_from(offset).map_err(|_| VulkanError::UnsupportedOperation("wl_shm buffer layout"))?;
    let width =
        usize::try_from(width).map_err(|_| VulkanError::UnsupportedOperation("wl_shm buffer layout"))?;
    let height =
        usize::try_from(height).map_err(|_| VulkanError::UnsupportedOperation("wl_shm buffer layout"))?;
    let stride =
        usize::try_from(stride).map_err(|_| VulkanError::UnsupportedOperation("wl_shm buffer layout"))?;
    let row_len = width
        .checked_mul(bytes_per_pixel)
        .ok_or(VulkanError::UnsupportedOperation("wl_shm buffer layout"))?;
    if stride < row_len {
        return Err(VulkanError::UnsupportedOperation("wl_shm buffer layout"));
    }
    let last_row = height
        .checked_sub(1)
        .and_then(|row| row.checked_mul(stride))
        .ok_or(VulkanError::UnsupportedOperation("wl_shm buffer layout"))?;
    let required_len = offset
        .checked_add(last_row)
        .and_then(|start| start.checked_add(row_len))
        .ok_or(VulkanError::UnsupportedOperation("wl_shm buffer layout"))?;
    if required_len > len {
        return Err(VulkanError::UnsupportedOperation("wl_shm buffer length"));
    }
    if ptr.is_null() {
        return Err(VulkanError::UnsupportedOperation("wl_shm buffer"));
    }

    let packed_len = row_len
        .checked_mul(height)
        .ok_or(VulkanError::UnsupportedOperation("wl_shm buffer layout"))?;
    let mut packed = vec![0; packed_len];
    for row in 0..height {
        let row_start = offset
            .checked_add(
                row.checked_mul(stride)
                    .ok_or(VulkanError::UnsupportedOperation("wl_shm buffer layout"))?,
            )
            .ok_or(VulkanError::UnsupportedOperation("wl_shm buffer layout"))?;
        // SAFETY: `with_buffer_contents` provides a raw pointer valid for `len` bytes for the
        // duration of the callback. Bounds above prove `row_start..row_start + row_len` is inside
        // that range, and `packed` is an owned allocation large enough for this row. Use a raw
        // pointer copy instead of creating a Rust slice/reference into client-controlled shared
        // memory, which may be mutated by the client while this copy runs.
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr.add(row_start),
                packed.as_mut_ptr().add(row * row_len),
                row_len,
            );
        }
    }

    Ok(packed)
}

fn update_region_to_vk(
    texture_size: Size<i32, BufferCoord>,
    region: Rectangle<i32, BufferCoord>,
) -> Result<(vk::Offset3D, vk::Extent3D), VulkanError> {
    if region.loc.x < 0 || region.loc.y < 0 || region.size.w <= 0 || region.size.h <= 0 {
        return Err(VulkanError::UnsupportedOperation("memory update region"));
    }
    if region
        .loc
        .x
        .checked_add(region.size.w)
        .is_none_or(|right| right > texture_size.w)
        || region
            .loc
            .y
            .checked_add(region.size.h)
            .is_none_or(|bottom| bottom > texture_size.h)
    {
        return Err(VulkanError::UnsupportedOperation("memory update region"));
    }

    Ok((
        vk::Offset3D {
            x: region.loc.x,
            y: region.loc.y,
            z: 0,
        },
        vk::Extent3D {
            width: region
                .size
                .w
                .try_into()
                .map_err(|_| VulkanError::UnsupportedOperation("memory update region"))?,
            height: region
                .size
                .h
                .try_into()
                .map_err(|_| VulkanError::UnsupportedOperation("memory update region"))?,
            depth: 1,
        },
    ))
}

fn clear_color_value_for_format(format: Fourcc, color: Color32F) -> Result<vk::ClearColorValue, VulkanError> {
    let mut components = color.components();
    if get_format_info(format)?.opaque_alpha {
        components[3] = 1.0;
    }

    Ok(vk::ClearColorValue { float32: components })
}

#[cfg(test)]
mod tests;
