//! Vulkan renderer-family support for Anvil's udev backend.
//!
//! This module intentionally does not select Vulkan for `--tty-udev` yet. It only models the
//! renderer-family side in the same `GpuManager`/`MultiRenderer` shape used by the existing GLES
//! path, while keeping GBM as the dmabuf allocator for DRM scanout buffers.

use std::{
    collections::HashMap,
    fmt,
    sync::atomic::{AtomicBool, Ordering},
};

use smithay::backend::{
    allocator::{
        Allocator,
        dmabuf::{AnyError, Dmabuf, DmabufAllocator},
        gbm::{GbmAllocator, GbmBufferFlags, GbmDevice},
    },
    drm::{DrmDeviceFd, DrmNode},
    renderer::{
        multigpu::{ApiDevice, GraphicsApi},
        vulkan::{VulkanError, VulkanRenderer},
    },
    vulkan::{Instance, PhysicalDevice},
};
use tracing::warn;

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
}

/// Errors raised by the Vulkan udev renderer family.
#[derive(Debug, thiserror::Error)]
pub enum VulkanGbmError {
    /// Vulkan renderer error.
    #[error(transparent)]
    Vulkan(#[from] VulkanError),
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
            let Some(node) = physical_device_node(&physical_device) else {
                continue;
            };
            if !self
                .devices
                .keys()
                .any(|configured| configured.dev_id() == node.dev_id())
            {
                continue;
            }
            if list
                .iter()
                .any(|renderer| renderer.node.dev_id() == node.dev_id())
            {
                continue;
            }

            let Some((configured_node, allocator)) = self
                .devices
                .iter()
                .find(|(configured_node, _)| configured_node.dev_id() == node.dev_id())
            else {
                continue;
            };

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

fn physical_device_node(physical_device: &PhysicalDevice) -> Option<DrmNode> {
    physical_device
        .render_node()
        .ok()
        .flatten()
        .or_else(|| physical_device.primary_node().ok().flatten())
}
