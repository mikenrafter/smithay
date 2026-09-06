//! Vulkan renderer-family support for Anvil's udev backend.
//!
//! Uses smithay's [`VulkanGbmBackend`]. Anvil opts into Linux dma-buf interop so the
//! linux-dmabuf global can advertise [`wayland_sampled_dmabuf_formats`]. Generic `ImportDma`
//! stays fail-closed.

use smithay::backend::{
    allocator::{dmabuf::Dmabuf, format::FormatSet, gbm::GbmDevice},
    drm::{DrmDeviceFd, DrmNode},
    renderer::{Bind, multigpu::vulkan::VulkanGbmBackend},
    vulkan::{Instance, version::Version},
};

/// The Vulkan udev graphics API type.
pub type Graphics = VulkanGbmBackend<DrmDeviceFd>;
/// The Vulkan udev GPU manager type.
pub type GpuManager = smithay::backend::renderer::multigpu::GpuManager<Graphics>;
/// The Vulkan udev renderer type.
pub type Renderer<'a> = smithay::backend::renderer::multigpu::MultiRenderer<'a, 'a, Graphics, Graphics>;
/// The Vulkan udev GPU manager creation error type.
pub type GpuManagerError = smithay::backend::renderer::multigpu::Error<Graphics, Graphics>;
pub use smithay::backend::renderer::multigpu::vulkan::Error as VulkanGbmError;

/// Create the Vulkan udev GPU manager with Wayland linux-dmabuf interop enabled.
pub fn create_gpu_manager() -> Result<GpuManager, GpuManagerError> {
    let instance = Instance::new(Version::VERSION_1_3, None)
        .map_err(|err| GpuManagerError::RenderApiError(VulkanGbmError::from(err)))?;
    smithay::backend::renderer::multigpu::GpuManager::new(
        VulkanGbmBackend::new(instance).with_wayland_linux_dmabuf_interop(true),
    )
}

/// Add a GBM device to the Vulkan udev GPU manager after verifying Vulkan can identify it.
pub fn add_gpu_node(
    gpus: &mut GpuManager,
    node: DrmNode,
    gbm: GbmDevice<DrmDeviceFd>,
) -> Result<DrmNode, VulkanGbmError> {
    let node = gpus.as_ref().preferred_node_for_node(node)?;
    gpus.as_mut().add_node(node, gbm);
    Ok(node)
}

/// Return Vulkan's public GBM scanout render-target formats for Anvil's DRM output manager.
pub fn render_target_formats(renderer: &mut Renderer<'_>, _has_render_node: bool) -> FormatSet {
    <Renderer<'_> as Bind<Dmabuf>>::supported_formats(renderer).unwrap_or_default()
}

/// Formats the linux-dmabuf global may advertise for Vulkan Wayland sampled import.
///
/// This is not generic `ImportDma::dmabuf_formats`. Empty unless Linux dma-buf interop is enabled
/// on the inner renderer.
pub fn wayland_sampled_dmabuf_formats(renderer: &Renderer<'_>) -> FormatSet {
    renderer.as_ref().wayland_sampled_dmabuf_formats()
}

/// Protocol-create admission for a sampled dmabuf. Does not import or acquire.
pub fn sampled_dmabuf_import_supported(renderer: &Renderer<'_>, dmabuf: &Dmabuf) -> bool {
    renderer.as_ref().sampled_dmabuf_import_supported(dmabuf)
}
