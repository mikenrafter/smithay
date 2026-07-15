//! Vulkan renderer-family support for Anvil's udev backend.
//!
//! This module models the renderer-family side in the same `GpuManager`/`MultiRenderer` shape used
//! by the existing GLES path, while keeping GBM as the dmabuf allocator for DRM scanout buffers.
//!
//! Anvil selects this family for no-default `udev_vulkan` builds. Builds that enable Anvil's `egl`
//! feature keep the GLES renderer family so wl_drm/EGL buffer support is not advertised through a
//! Vulkan renderer that cannot provide it.

use std::{
    collections::HashMap,
    fmt,
    sync::atomic::{AtomicBool, Ordering},
};

use smithay::backend::{
    SwapBuffersError,
    allocator::{
        Allocator,
        dmabuf::{AnyError, Dmabuf, DmabufAllocator},
        gbm::{GbmAllocator, GbmBufferFlags, GbmDevice},
    },
    drm::{DrmDeviceFd, DrmNode},
    renderer::{
        Bind,
        multigpu::{ApiDevice, GraphicsApi},
        vulkan::{VulkanError, VulkanOwnedDmabufRenderTarget, VulkanRenderer},
    },
    vulkan::{Instance, InstanceError, PhysicalDevice, version::Version},
};
use tracing::warn;

/// The Vulkan udev graphics API type.
pub type Graphics = VulkanGbmBackend;
/// The Vulkan udev GPU manager type.
pub type GpuManager = smithay::backend::renderer::multigpu::GpuManager<Graphics>;
/// The Vulkan udev renderer type.
pub type Renderer<'a> = smithay::backend::renderer::multigpu::MultiRenderer<'a, 'a, Graphics, Graphics>;
/// The Vulkan udev GPU manager creation error type.
pub type GpuManagerError = smithay::backend::renderer::multigpu::Error<Graphics, Graphics>;

/// Create the Vulkan udev GPU manager.
pub fn create_gpu_manager() -> Result<GpuManager, GpuManagerError> {
    let instance = Instance::new(Version::VERSION_1_3, None)
        .map_err(|err| GpuManagerError::RenderApiError(VulkanGbmError::from(err)))?;
    smithay::backend::renderer::multigpu::GpuManager::new(VulkanGbmBackend::new(instance))
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

/// Return Vulkan's explicit dmabuf render-target formats for Anvil's DRM output manager.
pub fn render_target_formats(
    renderer: &mut Renderer<'_>,
    _has_render_node: bool,
) -> smithay::backend::allocator::format::FormatSet {
    <Renderer<'_> as Bind<VulkanOwnedDmabufRenderTarget<'static>>>::supported_formats(renderer)
        .unwrap_or_default()
}

/// A Vulkan [`GraphicsApi`] backed by GBM dmabuf allocation for DRM scanout targets.
pub struct VulkanGbmBackend {
    instance: Instance,
    devices: HashMap<DrmNode, GbmAllocator<DrmDeviceFd>>,
    needs_enumeration: AtomicBool,
}

impl fmt::Debug for VulkanGbmBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanGbmBackend")
            .field("devices", &self.devices.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl VulkanGbmBackend {
    /// Create a Vulkan renderer family from an existing Vulkan instance.
    pub fn new(instance: Instance) -> Self {
        Self {
            instance,
            devices: HashMap::new(),
            needs_enumeration: AtomicBool::new(true),
        }
    }

    /// Add a GBM device Anvil may use for Vulkan dmabuf render targets.
    pub fn add_node(&mut self, node: DrmNode, gbm: GbmDevice<DrmDeviceFd>) {
        self.devices
            .entry(node)
            .or_insert_with(|| GbmAllocator::new(gbm, GbmBufferFlags::RENDERING | GbmBufferFlags::SCANOUT));
        self.needs_enumeration.store(true, Ordering::SeqCst);
    }

    /// Remove a previously-added GBM device.
    pub fn remove_node(&mut self, node: &DrmNode) {
        if self.devices.remove(node).is_some() {
            self.needs_enumeration.store(true, Ordering::SeqCst);
        }
    }

    /// Return the preferred Vulkan node matching a DRM node opened by Anvil.
    pub fn preferred_node_for_node(&self, node: DrmNode) -> Result<DrmNode, VulkanGbmError> {
        let mut physical_devices = PhysicalDevice::enumerate(&self.instance).map_err(VulkanError::from)?;
        physical_devices
            .find(|physical_device| physical_device_matches_node(physical_device, node))
            .and_then(|physical_device| preferred_physical_device_node(&physical_device))
            .ok_or(VulkanGbmError::NoPhysicalDevice(node))
    }
}

/// Errors raised by the Vulkan udev renderer family.
#[derive(Debug, thiserror::Error)]
pub enum VulkanGbmError {
    /// Vulkan instance creation failed.
    #[error(transparent)]
    Instance(#[from] InstanceError),
    /// Vulkan renderer error.
    #[error(transparent)]
    Vulkan(#[from] VulkanError),
    /// No Vulkan physical device maps to the DRM node.
    #[error("No Vulkan physical device maps to DRM node {0}")]
    NoPhysicalDevice(DrmNode),
}

impl From<VulkanGbmError> for SwapBuffersError {
    fn from(err: VulkanGbmError) -> Self {
        match err {
            VulkanGbmError::Vulkan(err) => err.into(),
            VulkanGbmError::Instance(_) | VulkanGbmError::NoPhysicalDevice(_) => {
                SwapBuffersError::ContextLost(Box::new(err))
            }
        }
    }
}

impl GraphicsApi for VulkanGbmBackend {
    type Device = VulkanGbmDevice;
    type Error = VulkanGbmError;

    fn enumerate(&self, list: &mut Vec<Self::Device>) -> Result<(), Self::Error> {
        self.needs_enumeration.store(false, Ordering::SeqCst);
        list.retain(|renderer| {
            self.devices
                .keys()
                .any(|node| renderer.node.dev_id() == node.dev_id())
        });

        let physical_devices = PhysicalDevice::enumerate(&self.instance).map_err(VulkanError::from)?;
        for physical_device in physical_devices {
            let Some(node) = preferred_physical_device_node(&physical_device) else {
                continue;
            };
            let Some((configured_node, allocator)) = self.devices.iter().find(|(configured_node, _)| {
                physical_device_matches_node(&physical_device, **configured_node)
            }) else {
                continue;
            };
            if list
                .iter()
                .any(|renderer| renderer.node.dev_id() == node.dev_id())
            {
                continue;
            }

            let renderer = match VulkanRenderer::builder()
                .with_physical_device(physical_device.clone())
                .build()
            {
                Ok(renderer) => renderer,
                Err(err) => {
                    warn!(?node, ?err, "Skipping Vulkan renderer device");
                    continue;
                }
            };

            list.push(VulkanGbmDevice {
                node: *configured_node,
                renderer,
                allocator: Box::new(DmabufAllocator(allocator.clone())),
            });
        }

        Ok(())
    }

    fn needs_enumeration(&self) -> bool {
        self.needs_enumeration.load(Ordering::Acquire)
    }

    fn identifier() -> &'static str {
        "vulkan_gbm"
    }
}

/// A Vulkan renderer device tracked by Anvil's udev renderer family.
pub struct VulkanGbmDevice {
    node: DrmNode,
    renderer: VulkanRenderer,
    allocator: Box<dyn Allocator<Buffer = Dmabuf, Error = AnyError>>,
}

impl fmt::Debug for VulkanGbmDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanGbmDevice")
            .field("node", &self.node)
            .field("renderer", &self.renderer)
            .finish_non_exhaustive()
    }
}

impl ApiDevice for VulkanGbmDevice {
    type Renderer = VulkanRenderer;

    fn renderer(&self) -> &Self::Renderer {
        &self.renderer
    }

    fn renderer_mut(&mut self) -> &mut Self::Renderer {
        &mut self.renderer
    }

    fn allocator(&mut self) -> &mut dyn Allocator<Buffer = Dmabuf, Error = AnyError> {
        self.allocator.as_mut()
    }

    fn node(&self) -> &DrmNode {
        &self.node
    }

    fn can_do_cross_device_imports(&self) -> bool {
        false
    }
}

fn preferred_physical_device_node(physical_device: &PhysicalDevice) -> Option<DrmNode> {
    physical_device
        .render_node()
        .ok()
        .flatten()
        .or_else(|| physical_device.primary_node().ok().flatten())
}

fn physical_device_matches_node(physical_device: &PhysicalDevice, node: DrmNode) -> bool {
    physical_device.render_node().ok().flatten() == Some(node)
        || physical_device.primary_node().ok().flatten() == Some(node)
}
