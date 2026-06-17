//! Native Vulkan renderer.
//!
//! This module provides a provisional, opt-in Vulkan renderer for an explicit [`PhysicalDevice`]. It
//! can upload sampled textures from CPU memory, render to in-memory/offscreen targets, read back
//! those offscreen targets through Smithay's public [`ImportMem`], [`Bind`], [`Offscreen`],
//! [`Renderer`], and [`ExportMem`] traits, and expose validation-stage dmabuf render-target
//! development hooks. It remains intentionally incomplete and does not provide a complete compositor
//! renderer, HDR, colour-management policy, or a fully integrated presentation backend.
//!
//! Downstream compositors must not treat the presence of this module or the `renderer_vulkan`
//! feature as broad Vulkan rendering support. Real enablement must be added incrementally behind
//! explicit capability bits, with tests and stub failure paths before enabling working
//! functionality.
//!
//! Smithay's optional renderer traits are capability surfaces. The CPU-memory/offscreen path is the
//! most complete path. Dmabuf render-target support is development-gated in this fork, and its Vulkan
//! external-ownership and synchronization preconditions are explicit on
//! [`VulkanRenderer::bind_dmabuf_render_target`]. Generic [`Bind<Dmabuf>`] uses that path with a
//! conservative discard/full-repaint acquire policy so DRM/GBM compositor rendering follows the same
//! target abstraction as the other renderers once the validation-stage gate is true. Generic
//! `ImportDma`, texture `ExportMem`, `ExportDma`, broad explicit sync, blit/copy, and full
//! presentation remain unsupported until their corresponding capability bits can become true with
//! coverage. Sampled dmabuf import is validation-reachable through the explicit known-layout
//! development helper and the normal `ImportDmaWl` path's staged guards. The normal Wayland path
//! intentionally stops at Smithay's own Wayland/Vulkan interop policy guard until this fork models
//! the initial external ownership/layout contract directly; it is not public-advertised through
//! `ImportDma` yet.
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

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SampledDmabufKnownLayoutEvidence {
    _private: (),
}

#[allow(dead_code)]
impl SampledDmabufKnownLayoutEvidence {
    /// Create evidence that a sampled dmabuf is in foreign ownership with GENERAL layout.
    ///
    /// # Safety
    ///
    /// The caller must ensure the producer released the image to `VK_QUEUE_FAMILY_FOREIGN_EXT` in
    /// `VK_IMAGE_LAYOUT_GENERAL` before this renderer acquires it.
    unsafe fn foreign_general() -> Self {
        Self { _private: () }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SampledDmabufLayoutEvidence {
    /// A normal Wayland linux-dmabuf commit. Explicit acquire sync may prove producer completion,
    /// but it does not prove Vulkan queue-family ownership or image layout.
    WaylandDmabuf,
    /// Smithay-owned Wayland/Vulkan interop policy evidence for normal linux-dmabuf commits.
    ///
    /// This token is the intended continuation point for the normal `ImportDmaWl` path once this
    /// fork defines and tests its own initial external ownership/layout policy. It must only be
    /// produced by the Wayland/Vulkan policy guard, not by copying another renderer's assumptions.
    SmithayWaylandVulkanPolicy(SampledDmabufWaylandVulkanInteropPolicy),
    /// Caller-provided proof that the producer released the image to `FOREIGN` ownership in
    /// `VK_IMAGE_LAYOUT_GENERAL`.
    KnownForeignGeneral(SampledDmabufKnownLayoutEvidence),
}

/// Opaque evidence that a normal Wayland dmabuf commit satisfies Smithay's Vulkan interop policy.
///
/// This is deliberately private and currently unconstructable in production code. The future policy
/// must define, at minimum, the first-import external image layout, subsequent reacquire layout,
/// queue-family ownership transfer, acquire synchronization, release synchronization, and texture
/// cache invalidation rules for Wayland dmabufs. Keeping this as a separate evidence token prevents
/// future work from treating `linux-dmabuf` protocol metadata or explicit sync alone as a Vulkan
/// layout/ownership proof.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SampledDmabufWaylandVulkanInteropPolicy {
    _private: (),
}

/// Evidence for Smithay's chosen first-import external image state for a Wayland dmabuf.
///
/// This must eventually prove the Vulkan `oldLayout` and source queue family used when importing a
/// Wayland dmabuf that has no renderer-local history yet. It is intentionally separate from
/// acquire-sync evidence: synchronization orders producer completion, but does not identify the
/// image's current Vulkan layout or ownership.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SampledDmabufWaylandFirstImportLayoutPolicy {
    _private: (),
}

/// Evidence for Smithay's reacquire layout policy for a previously imported Wayland dmabuf.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SampledDmabufWaylandReacquireLayoutPolicy {
    _private: (),
}

/// Evidence for queue-family ownership transfers used by Smithay's Wayland/Vulkan dmabuf policy.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SampledDmabufWaylandQueueFamilyPolicy {
    _private: (),
}

/// Evidence that Wayland acquire sync is mapped into the Vulkan sampled-dmabuf acquire operation.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SampledDmabufWaylandAcquireSyncPolicy {
    _private: (),
}

/// Evidence that Vulkan sampled-dmabuf release is mapped to the Wayland release point.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SampledDmabufWaylandReleaseSyncPolicy {
    _private: (),
}

/// Evidence that texture-cache reuse preserves per-commit Wayland/Vulkan dmabuf contracts.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SampledDmabufWaylandTextureCachePolicy {
    _private: (),
}

/// Validated normal-path inputs available to Smithay's Wayland/Vulkan sampled-dmabuf policy.
///
/// This deliberately separates protocol/import evidence from the opaque policy evidence tokens. The
/// fields are necessary inputs for producing those tokens, but none of them alone proves Vulkan
/// image layout, queue-family ownership, or release/cache correctness.
#[allow(dead_code)]
struct SampledDmabufWaylandVulkanInteropPolicyContext<'a> {
    import: &'a image::VulkanDmabufImportState,
    acquire_sync: &'a SyncPoint,
    release_evidence: &'a SampledDmabufReleaseEvidence,
}

#[allow(dead_code)]
impl<'a> SampledDmabufWaylandVulkanInteropPolicyContext<'a> {
    fn new(
        import: &'a image::VulkanDmabufImportState,
        acquire_sync: &'a SyncPoint,
        release_evidence: &'a SampledDmabufReleaseEvidence,
    ) -> Self {
        Self {
            import,
            acquire_sync,
            release_evidence,
        }
    }
}

/// Validation evidence for Smithay's normal Wayland dmabuf -> Vulkan sampled-image policy.
///
/// Each field names one contract that must be backed by implementation and tests before the normal
/// `ImportDmaWl` path may construct [`SampledDmabufWaylandVulkanInteropPolicy`]. The default value
/// is deliberately all-`None` so production remains fail-closed at the first missing policy step.
#[allow(dead_code)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct SampledDmabufWaylandVulkanInteropPolicyContracts {
    /// Defines the external ownership and Vulkan image layout used for the first import of a client
    /// Wayland dmabuf into this renderer.
    first_import_layout: Option<SampledDmabufWaylandFirstImportLayoutPolicy>,
    /// Defines the external ownership and Vulkan image layout used when a previously imported
    /// Wayland dmabuf is committed again after Smithay released it.
    reacquire_layout: Option<SampledDmabufWaylandReacquireLayoutPolicy>,
    /// Defines the queue-family ownership transfer to and from this renderer's Vulkan queue.
    queue_family_transfer: Option<SampledDmabufWaylandQueueFamilyPolicy>,
    /// Defines how the Wayland acquire point is converted into a Vulkan wait dependency for the
    /// import/acquire transition.
    acquire_sync: Option<SampledDmabufWaylandAcquireSyncPolicy>,
    /// Defines how Vulkan sampling completion and foreign release are transferred to the Wayland
    /// release point.
    release_sync: Option<SampledDmabufWaylandReleaseSyncPolicy>,
    /// Defines how texture-cache reuse observes per-commit acquire/release obligations and avoids
    /// reusing stale host-side layout evidence.
    texture_cache_reuse: Option<SampledDmabufWaylandTextureCachePolicy>,
}

/// Evidence that a Wayland release point exists for a sampled dmabuf.
///
/// This is only protocol-handle evidence. It does not prove Vulkan has finished sampling, released
/// the image back to foreign ownership, or signaled/satisfied the Wayland release point.
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct SampledDmabufReleaseEvidence {
    release: image::VulkanSampledDmabufRelease,
}

/// Acquire options for binding a foreign dmabuf as a Vulkan render target.
///
/// This is the development-fork API for the Vulkan-specific external target contract. It is explicit
/// because a plain [`Dmabuf`] does not encode Vulkan queue-family ownership, image layout, or acquire
/// synchronization.
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

/// Owned typed target wrapper for binding a foreign dmabuf through Vulkan's explicit acquire path.
///
/// This is useful for generic render paths that receive an owned [`Dmabuf`] handle from an allocator
/// or swapchain before constructing the renderer target. Creating it is unsafe for the same reason as
/// [`VulkanDmabufRenderTarget`]: the caller must prove the Vulkan external-memory ownership, layout,
/// and acquire-synchronization requirements documented on
/// [`VulkanRenderer::bind_dmabuf_render_target`] before every later safe bind.
#[derive(Debug)]
pub struct VulkanOwnedDmabufRenderTarget<'sync> {
    dmabuf: Dmabuf,
    acquire: VulkanDmabufRenderTargetAcquire<'sync>,
}

impl<'sync> VulkanOwnedDmabufRenderTarget<'sync> {
    /// Creates an owned Vulkan dmabuf render-target wrapper with explicit acquire options.
    ///
    /// # Safety
    ///
    /// The caller must satisfy the safety requirements documented on
    /// [`VulkanRenderer::bind_dmabuf_render_target`] for `dmabuf` and `acquire` before every later
    /// safe bind of this wrapper. The same underlying dmabuf storage must not be bound through
    /// another alias while this wrapper, or a framebuffer created from it, is live.
    pub unsafe fn new(dmabuf: Dmabuf, acquire: VulkanDmabufRenderTargetAcquire<'sync>) -> Self {
        Self { dmabuf, acquire }
    }

    /// Creates an owned discard/full-repaint target wrapper.
    ///
    /// # Safety
    ///
    /// The caller must satisfy the discard-acquire requirements documented on
    /// [`VulkanRenderer::bind_dmabuf_render_target`] before every later safe bind of this wrapper.
    pub unsafe fn discard(dmabuf: Dmabuf) -> Self {
        // SAFETY: Forwarded to this constructor's caller.
        unsafe { Self::new(dmabuf, VulkanDmabufRenderTargetAcquire::discard()) }
    }

    /// Creates an owned preserve-content target wrapper.
    ///
    /// # Safety
    ///
    /// The caller must satisfy the preserve-acquire requirements documented on
    /// [`VulkanRenderer::bind_dmabuf_render_target`] before every later safe bind of this wrapper,
    /// including the foreign release in `VK_IMAGE_LAYOUT_GENERAL`.
    pub unsafe fn preserve(dmabuf: Dmabuf, acquire_sync: Option<&'sync SyncPoint>) -> Self {
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

    /// Returns the underlying dmabuf handle.
    pub fn dmabuf(&self) -> &Dmabuf {
        &self.dmabuf
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

    fn development_gated_dmabuf_render_target_formats(&self) -> FormatSet {
        if self.capabilities.rendering.dmabuf_target_development {
            self.capabilities.formats.dmabuf_render_target.clone()
        } else {
            FormatSet::default()
        }
    }

    fn public_dmabuf_import_formats(&self) -> FormatSet {
        if self
            .validate_sampled_dmabuf_public_advertisement_contract()
            .is_ok()
        {
            self.capabilities.formats.dmabuf_import.clone()
        } else {
            FormatSet::default()
        }
    }

    /// Check whether the sampled dmabuf path may be public-advertised through [`ImportDma`].
    ///
    /// This is intentionally stricter than raw Vulkan probing. Future implementation work should
    /// make this pass only after metadata validation, acquire synchronization, ownership/layout
    /// transitions, sampled rendering, release synchronization, and tests all pass through the normal
    /// Smithay renderer path.
    fn validate_sampled_dmabuf_public_advertisement_contract(&self) -> Result<(), VulkanError> {
        if !self.capabilities.import.dmabuf {
            return Err(VulkanError::NotPublicAdvertised("sampled dmabuf import"));
        }
        if self.capabilities.formats.dmabuf_import.iter().next().is_none() {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf advertised formats",
            ));
        }

        // Final enablement marker: even if raw capability data is populated, public `ImportDma`
        // advertisement must stay fail-closed until import, acquire, sampling, release, and tests
        // are complete on the normal Smithay path.
        Err(VulkanError::MissingCapability("sampled dmabuf import lifecycle"))
    }

    /// Convert Wayland explicit-sync state into the renderer sync-point contract used by Vulkan.
    #[cfg(feature = "wayland_frontend")]
    fn sampled_dmabuf_wayland_acquire_sync_point(
        &self,
        #[cfg(feature = "backend_drm")] buffer: &super::utils::Buffer,
        #[cfg(not(feature = "backend_drm"))] _buffer: &super::utils::Buffer,
    ) -> Result<SyncPoint, VulkanError> {
        #[cfg(feature = "backend_drm")]
        {
            let acquire_sync = buffer.acquire_point().cloned().map(SyncPoint::from);
            self.validate_sampled_dmabuf_wayland_acquire_sync_contract(acquire_sync.as_ref())?;
            acquire_sync.ok_or(VulkanError::NotPublicAdvertised("sampled dmabuf implicit sync"))
        }

        #[cfg(not(feature = "backend_drm"))]
        {
            Err(VulkanError::MissingCapability(
                "sampled dmabuf explicit sync contract",
            ))
        }
    }

    /// Extract the Wayland release point needed for the sampled-dmabuf release lifecycle.
    #[cfg(feature = "wayland_frontend")]
    fn sampled_dmabuf_wayland_release_evidence(
        &self,
        #[cfg(feature = "backend_drm")] buffer: &super::utils::Buffer,
        #[cfg(not(feature = "backend_drm"))] _buffer: &super::utils::Buffer,
    ) -> Result<SampledDmabufReleaseEvidence, VulkanError> {
        #[cfg(feature = "backend_drm")]
        {
            let Some(release_point) = buffer.release_point().cloned() else {
                return Err(VulkanError::MissingCapability(
                    "sampled dmabuf release point contract",
                ));
            };

            Ok(SampledDmabufReleaseEvidence {
                release: image::VulkanSampledDmabufRelease::wayland_syncobj(release_point),
            })
        }

        #[cfg(not(feature = "backend_drm"))]
        {
            Err(VulkanError::MissingCapability(
                "sampled dmabuf release point contract",
            ))
        }
    }

    /// Validate the development-stage sampled dmabuf import subset before any Vulkan object work.
    ///
    /// This is intentionally separate from public [`ImportDma`] advertisement. It models the next
    /// intended path in small fail-closed steps so future work has a clear continuation point:
    ///
    /// 1. validate dmabuf metadata and raw Vulkan modifier capability,
    /// 2. consume Wayland explicit acquire synchronization from the surface-state buffer,
    /// 3. require/provide the missing external ownership and image-layout contract,
    /// 4. release the sampled image back to foreign ownership only after Vulkan sampling is done,
    /// 5. public-advertise [`ImportDma`] formats only after the whole lifecycle is covered.
    fn validate_sampled_dmabuf_import_metadata(
        &self,
        dmabuf: &Dmabuf,
    ) -> Result<image::VulkanDmabufImportState, VulkanError> {
        let import = image::VulkanDmabufImportState::from_dmabuf(dmabuf)?;

        if import.plane_count() != 1 {
            return Err(VulkanError::UnsupportedOperation("sampled dmabuf planes"));
        }
        if import.modifier() == Modifier::Invalid {
            return Err(VulkanError::MissingCapability("sampled dmabuf explicit modifier"));
        }
        get_format_info(import.format())?;

        if !self
            .capabilities
            .formats
            .has_sampled_dmabuf_modifier_record(&import)
        {
            return Err(VulkanError::MissingCapability("sampled dmabuf format/modifier"));
        }

        Ok(import)
    }

    /// Validate the Wayland acquire synchronization part of sampled dmabuf import.
    ///
    /// `linux-dmabuf` alone implies implicit synchronization. Vulkan sampled import remains
    /// not public-advertised for that case until a tested implicit-sync policy exists. The current
    /// validation-stage path accepts only commits that carry explicit acquire synchronization through
    /// Smithay's renderer-managed surface-state buffer.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_acquire_sync_contract(
        &self,
        acquire_sync: Option<&SyncPoint>,
    ) -> Result<(), VulkanError> {
        if acquire_sync.is_some_and(SyncPoint::contains_fence) {
            Ok(())
        } else {
            Err(VulkanError::NotPublicAdvertised("sampled dmabuf implicit sync"))
        }
    }

    /// Validate that the Wayland commit provided a release point for sampled dmabuf import.
    ///
    /// The presence of a release point is only evidence that the compositor has a protocol object to
    /// satisfy later. It does not by itself make the sampled dmabuf release lifecycle complete.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_release_point_contract(
        &self,
        has_release_point: bool,
    ) -> Result<SampledDmabufReleaseEvidence, VulkanError> {
        if has_release_point {
            Ok(SampledDmabufReleaseEvidence {
                release: image::VulkanSampledDmabufRelease::validation_stage_without_wayland_point(),
            })
        } else {
            Err(VulkanError::MissingCapability(
                "sampled dmabuf release point contract",
            ))
        }
    }

    /// Validate Smithay's own Wayland/Vulkan sampled-dmabuf interop policy.
    ///
    /// `linux-dmabuf` metadata plus explicit acquire/release sync is enough to identify memory,
    /// format/modifier, and producer/compositor ordering, but it is not by itself a Vulkan
    /// queue-family ownership or image-layout contract. This guard is the named development-gated
    /// continuation point for the normal `ImportDmaWl` path. Future implementation must replace
    /// this fail-closed marker with a Smithay-owned policy that defines how initial import,
    /// subsequent reacquire, sampling, release, and texture-cache reuse map onto Vulkan external
    /// image state.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_vulkan_interop_policy(
        &self,
        context: &SampledDmabufWaylandVulkanInteropPolicyContext<'_>,
    ) -> Result<SampledDmabufLayoutEvidence, VulkanError> {
        let first_import_layout = self.validate_sampled_dmabuf_wayland_first_import_layout_policy(context)?;
        let reacquire_layout = self.validate_sampled_dmabuf_wayland_reacquire_layout_policy(context)?;
        let queue_family_transfer = self.validate_sampled_dmabuf_wayland_queue_family_policy(context)?;
        let acquire_sync = self.validate_sampled_dmabuf_wayland_acquire_sync_policy(context)?;
        let release_sync = self.validate_sampled_dmabuf_wayland_release_sync_policy(context)?;
        let texture_cache_reuse = self.validate_sampled_dmabuf_wayland_texture_cache_policy(context)?;
        self.validate_sampled_dmabuf_wayland_vulkan_interop_policy_contracts(
            SampledDmabufWaylandVulkanInteropPolicyContracts {
                first_import_layout: Some(first_import_layout),
                reacquire_layout: Some(reacquire_layout),
                queue_family_transfer: Some(queue_family_transfer),
                acquire_sync: Some(acquire_sync),
                release_sync: Some(release_sync),
                texture_cache_reuse: Some(texture_cache_reuse),
            },
        )
        .map(SampledDmabufLayoutEvidence::SmithayWaylandVulkanPolicy)
    }

    /// Validate the first-import external image layout policy for a normal Wayland dmabuf.
    ///
    /// This is the first Smithay-owned policy item that must be implemented before the normal
    /// `ImportDmaWl` path can acquire an arbitrary client dmabuf as a sampled Vulkan image.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_first_import_layout_policy(
        &self,
        _context: &SampledDmabufWaylandVulkanInteropPolicyContext<'_>,
    ) -> Result<SampledDmabufWaylandFirstImportLayoutPolicy, VulkanError> {
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan first-import layout policy",
        ))
    }

    /// Validate the reacquire external image layout policy for a normal Wayland dmabuf.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_reacquire_layout_policy(
        &self,
        _context: &SampledDmabufWaylandVulkanInteropPolicyContext<'_>,
    ) -> Result<SampledDmabufWaylandReacquireLayoutPolicy, VulkanError> {
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan reacquire layout policy",
        ))
    }

    /// Validate queue-family ownership transfer policy for a normal Wayland dmabuf.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_queue_family_policy(
        &self,
        _context: &SampledDmabufWaylandVulkanInteropPolicyContext<'_>,
    ) -> Result<SampledDmabufWaylandQueueFamilyPolicy, VulkanError> {
        if !self.capabilities.external_memory.foreign_queue_family {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan queue-family capability",
            ));
        }

        Ok(SampledDmabufWaylandQueueFamilyPolicy { _private: () })
    }

    /// Validate acquire-sync import/wait policy for a normal Wayland dmabuf.
    ///
    /// The normal Wayland path requires an explicit acquire fence. The known-layout import helper
    /// then maps that [`SyncPoint`] into the Vulkan acquire submission by importing a sync-file wait
    /// semaphore when the device supports it and the sync point exports a suitable sync-file fd, or
    /// by waiting on the CPU before submitting the acquire barrier. This token does not prove image
    /// layout or queue-family ownership; those remain separate policy contracts.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_acquire_sync_policy(
        &self,
        context: &SampledDmabufWaylandVulkanInteropPolicyContext<'_>,
    ) -> Result<SampledDmabufWaylandAcquireSyncPolicy, VulkanError> {
        self.validate_sampled_dmabuf_wayland_acquire_sync_contract(Some(context.acquire_sync))?;
        Ok(SampledDmabufWaylandAcquireSyncPolicy { _private: () })
    }

    /// Validate release-sync export/transfer policy for a normal Wayland dmabuf.
    ///
    /// The release evidence carries the renderer-owned obligation to satisfy the Wayland release
    /// point only after Vulkan releases the sampled image back to foreign ownership. The release
    /// helper transfers an exported release sync-file into that release point when available, or
    /// signals it directly only after synchronous release completion.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_release_sync_policy(
        &self,
        context: &SampledDmabufWaylandVulkanInteropPolicyContext<'_>,
    ) -> Result<SampledDmabufWaylandReleaseSyncPolicy, VulkanError> {
        self.validate_sampled_dmabuf_release_lifecycle_contract(context.release_evidence.clone())?;
        Ok(SampledDmabufWaylandReleaseSyncPolicy { _private: () })
    }

    /// Validate texture-cache reuse policy for a normal Wayland dmabuf.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_texture_cache_policy(
        &self,
        _context: &SampledDmabufWaylandVulkanInteropPolicyContext<'_>,
    ) -> Result<SampledDmabufWaylandTextureCachePolicy, VulkanError> {
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan texture-cache policy",
        ))
    }

    /// Validate each named part of Smithay's Wayland/Vulkan sampled-dmabuf policy.
    ///
    /// The fully satisfied path is only a validation-stage scaffold until each field is replaced or
    /// backed by a concrete implementation predicate and test. Future work must produce evidence for
    /// each contract instead of setting these markers from protocol metadata or another renderer's
    /// assumptions.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_vulkan_interop_policy_contracts(
        &self,
        contracts: SampledDmabufWaylandVulkanInteropPolicyContracts,
    ) -> Result<SampledDmabufWaylandVulkanInteropPolicy, VulkanError> {
        if contracts.first_import_layout.is_none() {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan first-import layout policy",
            ));
        }
        if contracts.reacquire_layout.is_none() {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan reacquire layout policy",
            ));
        }
        if contracts.queue_family_transfer.is_none() {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan queue-family policy",
            ));
        }
        if contracts.acquire_sync.is_none() {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan acquire sync policy",
            ));
        }
        if contracts.release_sync.is_none() {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan release sync policy",
            ));
        }
        if contracts.texture_cache_reuse.is_none() {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan texture-cache policy",
            ));
        }

        Ok(SampledDmabufWaylandVulkanInteropPolicy { _private: () })
    }

    /// Validate the external ownership and image-layout contract for sampled dmabuf import.
    ///
    /// A Wayland acquire point proves producer completion, but not Vulkan queue-family ownership or
    /// image layout. Explicit known-layout evidence may pass directly. The normal Wayland path may
    /// pass only after Smithay's own Wayland/Vulkan interop policy evidence exists.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_known_layout_contract(
        &self,
        evidence: SampledDmabufLayoutEvidence,
    ) -> Result<(), VulkanError> {
        match evidence {
            SampledDmabufLayoutEvidence::KnownForeignGeneral(_)
            | SampledDmabufLayoutEvidence::SmithayWaylandVulkanPolicy(_) => Ok(()),
            SampledDmabufLayoutEvidence::WaylandDmabuf => Err(VulkanError::MissingCapability(
                "sampled dmabuf known-layout contract",
            )),
        }
    }

    /// Convert release-point evidence into the renderer-owned release obligation for a sampled
    /// dmabuf import.
    ///
    /// The obligation is satisfied by [`VulkanRenderer::release_imported_dmabuf_texture_to_foreign_general`]
    /// after Vulkan releases the sampled image back to foreign ownership. Synchronous releases signal
    /// the Wayland release point directly; exported release fences are imported into the Wayland DRM
    /// syncobj timeline point.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_release_lifecycle_contract(
        &self,
        evidence: SampledDmabufReleaseEvidence,
    ) -> Result<image::VulkanSampledDmabufRelease, VulkanError> {
        Ok(evidence.release)
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
        let import = self.validate_sampled_dmabuf_import_metadata(dmabuf)?;
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
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
        let layout_evidence = SampledDmabufLayoutEvidence::KnownForeignGeneral(unsafe {
            // SAFETY: This helper is unsafe and forwards the same known-layout contract to its
            // caller: the producer must have released to FOREIGN ownership in GENERAL layout.
            SampledDmabufKnownLayoutEvidence::foreign_general()
        });
        self.validate_sampled_dmabuf_known_layout_contract(layout_evidence)?;
        let import = self.validate_sampled_dmabuf_import_metadata(dmabuf)?;
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
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
        let layout_evidence = SampledDmabufLayoutEvidence::KnownForeignGeneral(unsafe {
            // SAFETY: This helper is unsafe and forwards the same known-layout contract to its
            // caller: the producer must have released to FOREIGN ownership in GENERAL layout.
            SampledDmabufKnownLayoutEvidence::foreign_general()
        });
        self.validate_sampled_dmabuf_known_layout_contract(layout_evidence)?;
        let import = self.validate_sampled_dmabuf_import_metadata(dmabuf)?;
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
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

    /// Import a known-layout dmabuf and attach the Wayland release obligation to the texture.
    ///
    /// # Safety
    ///
    /// The caller must satisfy the same known-layout and acquire-sync requirements as
    /// [`VulkanRenderer::create_imported_dmabuf_texture_with_known_general_layout_and_sync_point`].
    #[allow(dead_code)]
    unsafe fn create_imported_dmabuf_texture_with_known_general_layout_release_and_sync_point(
        &mut self,
        dmabuf: &Dmabuf,
        acquire_sync: Option<&SyncPoint>,
        release_evidence: SampledDmabufReleaseEvidence,
    ) -> Result<Option<VulkanTexture>, VulkanError> {
        let layout_evidence = SampledDmabufLayoutEvidence::KnownForeignGeneral(unsafe {
            // SAFETY: This helper is unsafe and forwards the same known-layout contract to its
            // caller: the producer must have released to FOREIGN ownership in GENERAL layout.
            SampledDmabufKnownLayoutEvidence::foreign_general()
        });
        self.validate_sampled_dmabuf_known_layout_contract(layout_evidence)?;
        let import = self.validate_sampled_dmabuf_import_metadata(dmabuf)?;
        let release = self.validate_sampled_dmabuf_release_lifecycle_contract(release_evidence)?;
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
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

        Ok(Some(
            VulkanTexture::from_acquired_dmabuf_sampled_image_with_release(
                self.context_id.clone(),
                &import,
                sampled_image,
                release,
            ),
        ))
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
    /// synchronization. Generic [`Bind<Dmabuf>`] delegates here with discard/full-repaint acquire
    /// policy; callers that use this method directly are opting into the explicit unsafe
    /// external-memory contract and may provide preserve/acquire-sync policy themselves.
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

    /// Import a dmabuf as a sampled texture when the producer's Vulkan external state is known.
    ///
    /// This is a validation-stage development helper for the intended sampled dmabuf path. It does
    /// not make generic [`ImportDma`] public-advertised, and `dmabuf_formats()` remains gated by
    /// [`VulkanImportCapabilities::dmabuf`] until arbitrary client-buffer acquire/layout/sync
    /// contracts are implemented and tested.
    ///
    /// Use [`VulkanRenderer::release_imported_dmabuf_texture_to_foreign_general_sync_point`] before
    /// handing the dmabuf back to a foreign Vulkan producer/consumer.
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
    pub unsafe fn import_dmabuf_texture_with_known_general_layout(
        &mut self,
        dmabuf: &Dmabuf,
        acquire_sync: Option<&SyncPoint>,
    ) -> Result<Option<VulkanTexture>, VulkanError> {
        // SAFETY: Forwarded from this public unsafe method's caller.
        unsafe {
            self.create_imported_dmabuf_texture_with_known_general_layout_and_sync_point(dmabuf, acquire_sync)
        }
    }

    /// Release an acquired dmabuf texture back to foreign ownership in `VK_IMAGE_LAYOUT_GENERAL`.
    ///
    /// This is the release counterpart to
    /// [`VulkanRenderer::import_dmabuf_texture_with_known_general_layout`]. It does not make public
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

        let (released, release_sync_file) =
            device.release_sampled_dmabuf_to_foreign_general(sampled_image.image(), export_sync_file)?;
        if released {
            texture.signal_sampled_dmabuf_release_point(release_sync_file.as_ref().map(OwnedFd::as_fd))?;
        }
        Ok((released, release_sync_file))
    }

    /// Release an acquired dmabuf texture and return the exported release fence as a [`SyncPoint`]
    /// when available.
    ///
    /// This is a validation-stage development helper for explicit known-layout sampled dmabuf
    /// imports. It does not make generic [`ImportDma`] public-advertised. When `released` is true,
    /// the texture's dmabuf image has been returned to foreign ownership and must not be sampled by
    /// this renderer again until it is reacquired. The returned [`SyncPoint`] is the release
    /// dependency for the foreign producer or consumer.
    pub fn release_imported_dmabuf_texture_to_foreign_general_sync_point(
        &mut self,
        texture: &VulkanTexture,
        export_sync_file: bool,
    ) -> Result<(bool, SyncPoint), VulkanError> {
        let (released, sync_file) =
            self.release_imported_dmabuf_texture_to_foreign_general(texture, export_sync_file)?;
        Ok((released, sync_point_from_sync_file(sync_file)))
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
            return Err(VulkanError::UnsupportedOperation("dmabuf external ownership"));
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
            return Err(VulkanError::UnsupportedOperation("dmabuf external ownership"));
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
        .ok_or(VulkanError::MissingCapability(
            "dmabuf render target format/modifier",
        ))
    }

    fn supported_formats(&self) -> Option<FormatSet> {
        Some(self.development_gated_dmabuf_render_target_formats())
    }
}

impl<'target, 'sync> RenderTargetLifecycle<VulkanDmabufRenderTarget<'target, 'sync>> for VulkanRenderer {
    fn target_age(&self, target: &VulkanDmabufRenderTarget<'target, 'sync>, age: usize) -> usize {
        if target.preserve_contents() { age } else { 0 }
    }

    fn release_after_render_error(&mut self, target: &mut Self::Framebuffer<'_>) -> Result<(), Self::Error> {
        self.release_dmabuf_render_target_after_render_error(target)
    }

    fn release_after_no_render(&mut self, target: &mut Self::Framebuffer<'_>) -> Result<(), Self::Error> {
        self.release_dmabuf_render_target_after_render_error(target)
    }
}

impl<'sync> Bind<VulkanOwnedDmabufRenderTarget<'sync>> for VulkanRenderer {
    fn bind<'a>(
        &mut self,
        target: &'a mut VulkanOwnedDmabufRenderTarget<'sync>,
    ) -> Result<Self::Framebuffer<'a>, Self::Error> {
        unsafe {
            // SAFETY: `VulkanOwnedDmabufRenderTarget` can only be constructed by callers that
            // accepted and upheld the explicit Vulkan external-memory acquire contract. Borrow the
            // owned dmabuf for the lifetime of the returned framebuffer so Rust prevents rebinding
            // through this wrapper while the framebuffer is live.
            self.bind_dmabuf_render_target(&mut target.dmabuf, target.acquire)
        }?
        .ok_or(VulkanError::MissingCapability(
            "dmabuf render target format/modifier",
        ))
    }

    fn supported_formats(&self) -> Option<FormatSet> {
        Some(self.development_gated_dmabuf_render_target_formats())
    }
}

impl<'sync> RenderTargetLifecycle<VulkanOwnedDmabufRenderTarget<'sync>> for VulkanRenderer {
    fn target_age(&self, target: &VulkanOwnedDmabufRenderTarget<'sync>, age: usize) -> usize {
        if target.preserve_contents() { age } else { 0 }
    }

    fn release_after_render_error(&mut self, target: &mut Self::Framebuffer<'_>) -> Result<(), Self::Error> {
        self.release_dmabuf_render_target_after_render_error(target)
    }

    fn release_after_no_render(&mut self, target: &mut Self::Framebuffer<'_>) -> Result<(), Self::Error> {
        self.release_dmabuf_render_target_after_render_error(target)
    }
}

impl Bind<Dmabuf> for VulkanRenderer {
    fn bind<'a>(&mut self, target: &'a mut Dmabuf) -> Result<Self::Framebuffer<'a>, Self::Error> {
        if !self.capabilities.rendering.dmabuf_target_development {
            return Err(VulkanError::NotPublicAdvertised("dmabuf render target"));
        }

        unsafe {
            // SAFETY: `Bind<Dmabuf>` follows Smithay's renderer target contract. For externally
            // shared targets, that contract requires callers to ensure no concurrent foreign access
            // and to satisfy renderer-specific external ownership/layout requirements before
            // binding. The Vulkan dmabuf target path uses a discard/full-repaint acquire policy here,
            // so previous contents are not preserved and no acquire fence is required by this
            // binding. Successful frames release the image for foreign/KMS use from `Frame::finish`;
            // failed or skipped renders are handled by `RenderTargetLifecycle<Dmabuf>` below.
            self.bind_dmabuf_render_target(target, VulkanDmabufRenderTargetAcquire::discard())
        }?
        .ok_or(VulkanError::MissingCapability(
            "dmabuf render target format/modifier",
        ))
    }

    fn supported_formats(&self) -> Option<FormatSet> {
        Some(self.development_gated_dmabuf_render_target_formats())
    }
}

impl RenderTargetLifecycle<Dmabuf> for VulkanRenderer {
    fn target_age(&self, _target: &Dmabuf, _age: usize) -> usize {
        // The generic dmabuf binding currently uses discard acquire, so preserved contents are not
        // part of the contract. Force full repaint until a future preserve/acquire-sync policy is
        // modeled in the standard path.
        0
    }

    fn release_after_render_error(&mut self, target: &mut Self::Framebuffer<'_>) -> Result<(), Self::Error> {
        self.release_dmabuf_render_target_after_render_error(target)
    }

    fn release_after_no_render(&mut self, target: &mut Self::Framebuffer<'_>) -> Result<(), Self::Error> {
        self.release_dmabuf_render_target_after_render_error(target)
    }
}

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
        self.public_dmabuf_import_formats()
    }

    fn import_dmabuf(
        &mut self,
        _dmabuf: &Dmabuf,
        _damage: Option<&[Rectangle<i32, BufferCoord>]>,
    ) -> Result<Self::TextureId, Self::Error> {
        Err(VulkanError::NotPublicAdvertised("sampled dmabuf import"))
    }
}

#[cfg(feature = "wayland_frontend")]
impl ImportDmaWl for VulkanRenderer {
    fn import_dma_buffer_from_surface_state(
        &mut self,
        buffer: &super::utils::Buffer,
        _surface: Option<&crate::wayland::compositor::SurfaceData>,
        _damage: &[Rectangle<i32, BufferCoord>],
    ) -> Result<Self::TextureId, Self::Error> {
        let dmabuf = crate::wayland::dmabuf::get_dmabuf(buffer)
            .expect("import_dma_buffer_from_surface_state without checking buffer type?");

        let import = self.validate_sampled_dmabuf_import_metadata(dmabuf)?;

        let acquire_sync = self.sampled_dmabuf_wayland_acquire_sync_point(buffer)?;
        let release_evidence = self.sampled_dmabuf_wayland_release_evidence(buffer)?;
        let policy_context =
            SampledDmabufWaylandVulkanInteropPolicyContext::new(&import, &acquire_sync, &release_evidence);
        let layout_evidence = self.validate_sampled_dmabuf_wayland_vulkan_interop_policy(&policy_context)?;
        self.validate_sampled_dmabuf_known_layout_contract(layout_evidence)?;

        let texture = unsafe {
            // SAFETY: The validation-stage Wayland path above currently fails closed at the
            // Smithay Wayland/Vulkan interop policy contract. When that guard is replaced by real
            // Wayland dmabuf ownership/layout evidence, the same policy must satisfy this helper's
            // unsafe precondition before the import can run.
            self.create_imported_dmabuf_texture_with_known_general_layout_release_and_sync_point(
                dmabuf,
                Some(&acquire_sync),
                release_evidence,
            )?
        };

        texture.ok_or(VulkanError::MissingCapability("sampled dmabuf texture import"))
    }
}

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
