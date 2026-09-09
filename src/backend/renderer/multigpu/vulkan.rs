//! Multi-gpu [`GraphicsApi`] using user-provided GBM devices and [`VulkanRenderer`].
//!
//! GBM allocates DRM scanout buffers. [`VulkanRenderer`] renders into those dmabufs.
//! Sampled Wayland import stays opt-in via [`VulkanGbmBackend::with_wayland_linux_dmabuf_interop`];
//! generic [`crate::backend::renderer::ImportDma`] stays fail-closed on the renderer.

use std::{
    collections::HashMap,
    fmt,
    os::unix::prelude::AsFd,
    sync::atomic::{AtomicBool, Ordering},
};

use tracing::warn;

use crate::backend::{
    SwapBuffersError,
    allocator::{
        Allocator,
        dmabuf::{AnyError, Dmabuf, DmabufAllocator},
        format::FormatSet,
        gbm::{GbmAllocator, GbmBufferFlags, GbmDevice},
    },
    drm::{CreateDrmNodeError, DrmNode},
    renderer::{
        multigpu::{ApiDevice, Error as MultiError, GraphicsApi},
        vulkan::{VulkanError, VulkanRenderer, VulkanTexture},
    },
    vulkan::{Instance, InstanceError, PhysicalDevice},
};

/// Errors raised by [`VulkanGbmBackend`].
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Vulkan instance creation failed.
    #[error(transparent)]
    Instance(#[from] InstanceError),
    /// Vulkan renderer error.
    #[error(transparent)]
    Vulkan(#[from] VulkanError),
    /// Error creating a DRM node.
    #[error(transparent)]
    DrmNode(#[from] CreateDrmNodeError),
    /// No Vulkan physical device maps to the DRM node.
    #[error("No Vulkan physical device maps to DRM node {0}")]
    NoPhysicalDevice(DrmNode),
}

impl From<Error> for SwapBuffersError {
    #[inline]
    fn from(err: Error) -> SwapBuffersError {
        match err {
            Error::Vulkan(err) => err.into(),
            Error::Instance(_) | Error::DrmNode(_) | Error::NoPhysicalDevice(_) => {
                SwapBuffersError::ContextLost(Box::new(err))
            }
        }
    }
}

/// A [`GraphicsApi`] using GBM devices and [`VulkanRenderer`] for DRM scanout.
pub struct VulkanGbmBackend<A: AsFd + 'static> {
    instance: Instance,
    devices: HashMap<DrmNode, GbmAllocator<A>>,
    allocator_flags: GbmBufferFlags,
    #[cfg(feature = "wayland_frontend")]
    wayland_linux_dmabuf_interop: bool,
    needs_enumeration: AtomicBool,
}

impl<A: AsFd + fmt::Debug + 'static> fmt::Debug for VulkanGbmBackend<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanGbmBackend")
            .field("devices", &self.devices.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl<A: AsFd + Clone + 'static> VulkanGbmBackend<A> {
    /// Create a Vulkan renderer family from an existing Vulkan instance.
    pub fn new(instance: Instance) -> Self {
        Self {
            instance,
            devices: HashMap::new(),
            allocator_flags: GbmBufferFlags::RENDERING | GbmBufferFlags::SCANOUT,
            #[cfg(feature = "wayland_frontend")]
            wayland_linux_dmabuf_interop: false,
            needs_enumeration: AtomicBool::new(true),
        }
    }

    /// Opt enumerated renderers into Wayland linux-dmabuf sampled import.
    ///
    /// Default is off. Generic `ImportDma` stays fail-closed either way.
    #[cfg(feature = "wayland_frontend")]
    pub fn with_wayland_linux_dmabuf_interop(mut self, enabled: bool) -> Self {
        self.wayland_linux_dmabuf_interop = enabled;
        self
    }

    /// Sets the default flags used for GBM allocations created after this call.
    pub fn set_allocator_flags(&mut self, flags: GbmBufferFlags) {
        self.allocator_flags = flags;
    }

    /// Add a GBM device for a DRM node.
    pub fn add_node(&mut self, node: DrmNode, gbm: GbmDevice<A>) {
        self.devices
            .entry(node)
            .or_insert_with(|| GbmAllocator::new(gbm, self.allocator_flags));
        self.needs_enumeration.store(true, Ordering::SeqCst);
    }

    /// Remove a previously-added GBM device.
    pub fn remove_node(&mut self, node: &DrmNode) {
        if self.devices.remove(node).is_some() {
            self.needs_enumeration.store(true, Ordering::SeqCst);
        }
    }

    /// Preferred Vulkan DRM node matching an opened node.
    pub fn preferred_node_for_node(&self, node: DrmNode) -> Result<DrmNode, Error> {
        let mut physical_devices = PhysicalDevice::enumerate(&self.instance).map_err(VulkanError::from)?;
        physical_devices
            .find(|physical_device| physical_device_matches_node(physical_device, node))
            .and_then(|physical_device| preferred_physical_device_node(&physical_device))
            .ok_or(Error::NoPhysicalDevice(node))
    }
}

impl<A: AsFd + Clone + 'static> GraphicsApi for VulkanGbmBackend<A> {
    type Device = VulkanGbmDevice;
    type Error = Error;

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
                Ok(mut renderer) => {
                    #[cfg(feature = "wayland_frontend")]
                    if self.wayland_linux_dmabuf_interop {
                        renderer.set_wayland_linux_dmabuf_interop(true);
                    }
                    renderer
                }
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

impl<T: GraphicsApi, A: AsFd + Clone + 'static> From<VulkanError> for MultiError<VulkanGbmBackend<A>, T>
where
    T::Error: 'static,
    <<T::Device as ApiDevice>::Renderer as crate::backend::renderer::RendererSuper>::Error: 'static,
{
    #[inline]
    fn from(err: VulkanError) -> MultiError<VulkanGbmBackend<A>, T> {
        MultiError::Render(err)
    }
}

/// [`ApiDevice`] of [`VulkanGbmBackend`].
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

    fn compositor_owned_sampled_dmabuf_formats(&self) -> FormatSet {
        self.renderer.capabilities().formats.dmabuf_import.clone()
    }

    fn can_import_released_compositor_dmabuf(&self) -> bool {
        self.renderer.capabilities().rendering.dmabuf_targets
            && self
                .renderer
                .capabilities()
                .formats
                .dmabuf_import
                .iter()
                .next()
                .is_some()
    }

    fn import_released_compositor_dmabuf(
        &mut self,
        dmabuf: &Dmabuf,
        acquire: Option<&crate::backend::renderer::sync::SyncPoint>,
    ) -> Result<VulkanTexture, VulkanError> {
        // SAFETY: The MultiRenderer hybrid copy path binds this compositor-owned dmabuf on
        // the source GPU, draws, and `Frame::finish` releases it to FOREIGN+GENERAL before
        // this import. Generic client `ImportDma` stays fail-closed.
        unsafe {
            self.renderer
                .import_dmabuf_texture_with_known_general_layout(dmabuf, acquire)
        }?
        .ok_or(VulkanError::MissingCapability("compositor-owned sampled dmabuf"))
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
