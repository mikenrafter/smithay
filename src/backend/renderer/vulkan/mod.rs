//! Native Vulkan renderer scaffold.
//!
//! This module establishes the renderer-side structure for a future native Vulkan backend. It is
//! intentionally non-functional: it does not create a Vulkan instance or device, import buffers,
//! render frames, present to KMS, implement HDR, or perform colour-management policy.
//!
//! Downstream compositors must not treat the presence of this module or the `renderer_vulkan`
//! feature as Vulkan rendering support. Real enablement must be added incrementally behind explicit
//! capability bits, with tests and stub failure paths before enabling working functionality.
//!
//! Smithay's optional renderer traits, such as `ImportMem`, `ImportDma`, `Bind`, `Offscreen`,
//! `ExportMem`, and `Blit`, are capability surfaces. The scaffold intentionally does not implement
//! those traits for [`VulkanRenderer`] until the corresponding capability bit can become true.
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

use crate::{
    backend::renderer::{ContextId, DebugFlags, Renderer, RendererSuper, TextureFilter, sync::SyncPoint},
    backend::vulkan::PhysicalDevice,
    utils::{Physical, Size, Transform},
};

mod capabilities;
mod device;
mod error;
pub mod format;
mod image;

pub use self::{
    capabilities::{
        VulkanColorCapabilities, VulkanDeviceCapabilities, VulkanExportCapabilities,
        VulkanFormatCapabilities, VulkanFormatCapabilityRecord, VulkanFormatUsage, VulkanImportCapabilities,
        VulkanRendererCapabilities, VulkanRenderingCapabilities, VulkanSyncCapabilities,
    },
    error::VulkanError,
    image::{VulkanFrame, VulkanRenderTarget, VulkanTexture},
};

use self::device::VulkanDeviceState;

/// Native Vulkan renderer scaffold.
#[derive(Debug)]
pub struct VulkanRenderer {
    context_id: ContextId<VulkanTexture>,
    debug_flags: DebugFlags,
    downscale_filter: TextureFilter,
    upscale_filter: TextureFilter,
    capabilities: VulkanRendererCapabilities,
    device: Option<VulkanDeviceState>,
}

/// Builder for future Vulkan renderer initialization.
///
/// The builder records the ownership shape expected by the real implementation while still returning
/// [`VulkanError::VulkanUnavailable`] in the scaffold. Device creation, queue selection, and capability
/// discovery must be added here before any optional renderer traits are implemented.
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
    /// This always returns [`VulkanError::VulkanUnavailable`] in the scaffold. Future implementation
    /// steps should populate renderer device state and capabilities here before exposing functionality.
    pub fn build(self) -> Result<VulkanRenderer, VulkanError> {
        let _ = self;
        Err(VulkanError::VulkanUnavailable)
    }
}

impl VulkanRenderer {
    /// Attempts to create a Vulkan renderer.
    ///
    /// This always returns [`VulkanError::VulkanUnavailable`] in the scaffold. A future implementation
    /// must only return `Ok` after real device initialization and capability discovery are wired up.
    pub fn new() -> Result<Self, VulkanError> {
        Self::builder().build()
    }

    /// Creates a builder for future Vulkan renderer initialization.
    pub fn builder() -> VulkanRendererBuilder {
        VulkanRendererBuilder::new()
    }

    /// Returns the scaffold capabilities without constructing a renderer.
    ///
    /// This is the only capability query available until real Vulkan device initialization exists.
    /// All capability bits are false and all format/extension sets are empty.
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
    /// All capability bits are false and all format/extension sets are empty in the scaffold.
    pub fn capabilities(&self) -> &VulkanRendererCapabilities {
        &self.capabilities
    }

    /// Returns whether this renderer is usable for rendering.
    pub fn is_usable(&self) -> bool {
        self.device.is_some() && self.capabilities.device.available
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
        _framebuffer: &'frame mut Self::Framebuffer<'buffer>,
        _output_size: Size<i32, Physical>,
        _dst_transform: Transform,
    ) -> Result<Self::Frame<'frame, 'buffer>, Self::Error>
    where
        'buffer: 'frame,
    {
        Err(VulkanError::UnsupportedOperation("frame creation"))
    }

    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        sync.wait().map_err(|_| VulkanError::SyncInterrupted)
    }

    fn cleanup_texture_cache(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
