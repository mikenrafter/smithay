//! Vulkan window presentation for the winit backend.
//!
//! This is a separate present path from [`super::WinitGraphicsBackend`], which is the EGL/GLES
//! window implementation. The event loop and window come from [`super::init_window`]; this module
//! owns the `VkSurfaceKHR` / swapchain and binds acquired images as
//! [`VulkanRenderTarget`]s.
//!
//! Vulkan WSI framebuffers are top-left, matching Wayland. Nested compositors should keep the
//! output transform at [`crate::utils::Transform::Normal`]. GLES winit's `Flipped180` compensates
//! for OpenGL's bottom-left origin and is not needed here; [`VulkanRenderer`] still rejects
//! non-identity frame transforms.

use std::sync::Arc;

use ash::{khr, vk};
use tracing::{debug, info, instrument};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use winit::window::Window as WinitWindow;

use crate::{
    backend::{
        SwapBuffersError,
        allocator::Fourcc,
        renderer::{
            Bind,
            vulkan::{VulkanError, VulkanRenderTarget, VulkanRenderer},
        },
        vulkan::{Instance, InstanceError, PhysicalDevice, version::Version},
    },
    utils::{Physical, Rectangle, Size},
};

use super::{Error, WinitEventLoop, init_window};

/// Window with a Vulkan swapchain created by `winit`.
///
/// This type is the Vulkan counterpart of [`super::WinitGraphicsBackend`]. It does not replace the
/// EGL/GLES window implementation; compositors that want GLES keep using [`super::init`].
pub struct WinitVulkanGraphicsBackend {
    renderer: VulkanRenderer,
    window: Arc<dyn WinitWindow>,
    surface: vk::SurfaceKHR,
    surface_fn: khr::surface::Instance,
    swapchain_fn: khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    format: Fourcc,
    extent: vk::Extent2D,
    current_image: Option<u32>,
    bound_target: Option<VulkanRenderTarget<'static>>,
    acquire_fence: vk::Fence,
}

/// Creates a [`WinitVulkanGraphicsBackend`] and corresponding [`WinitEventLoop`].
pub fn init_vulkan() -> Result<(WinitVulkanGraphicsBackend, WinitEventLoop), Error> {
    init_vulkan_from_attributes(
        winit::window::WindowAttributes::default()
            .with_surface_size(winit::dpi::LogicalSize::new(1280.0, 800.0))
            .with_title("Smithay")
            .with_visible(true),
    )
}

/// Creates a Vulkan winit graphics backend from window attributes.
pub fn init_vulkan_from_attributes(
    attributes: winit::window::WindowAttributes,
) -> Result<(WinitVulkanGraphicsBackend, WinitEventLoop), Error> {
    let (window, event_loop) = init_window(attributes)?;
    let backend = WinitVulkanGraphicsBackend::new(window)?;
    Ok((backend, event_loop))
}

impl std::fmt::Debug for WinitVulkanGraphicsBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WinitVulkanGraphicsBackend")
            .field("renderer", &self.renderer)
            .field("window", &self.window)
            .field("surface", &self.surface)
            .field("swapchain", &self.swapchain)
            .field("images", &self.images)
            .field("format", &self.format)
            .field("extent", &self.extent)
            .field("current_image", &self.current_image)
            .field("bound_target", &self.bound_target)
            .field("acquire_fence", &self.acquire_fence)
            .finish_non_exhaustive()
    }
}

impl WinitVulkanGraphicsBackend {
    fn new(window: Arc<dyn WinitWindow>) -> Result<Self, Error> {
        info!("Initializing a Vulkan winit present backend");

        let entry = Instance::loaded_entry().map_err(|_| VulkanError::VulkanUnavailable)?;
        let instance_extensions = instance_extensions_for_window(&window)?;
        // SAFETY: `instance_extensions` is the WSI pair required by the window system, including
        // `VK_KHR_surface` as the dependency of the platform surface extension
        // (`VUID-vkCreateInstance-ppEnabledExtensionNames-01388`).
        let instance = unsafe { Instance::with_extensions(Version::VERSION_1_3, None, &instance_extensions) }
            .map_err(vulkan_instance_error)?;

        let surface_fn = khr::surface::Instance::new(entry, instance.handle());
        // SAFETY: `instance` enabled the matching platform surface extension, and `window` outlives
        // the created `VkSurfaceKHR` because it is stored on the backend.
        let surface = unsafe { create_window_surface(entry, &instance, &window)? };

        let physical_device = select_physical_device(&instance, &surface_fn, surface)?;
        info!(
            device = physical_device.name(),
            ty = ?physical_device.ty(),
            "Selected Vulkan physical device for winit presentation"
        );

        let renderer = VulkanRenderer::builder()
            .with_physical_device(physical_device)
            .with_device_extensions(&[khr::swapchain::NAME])
            .build()?;
        let logical_device = renderer.logical_device()?;
        let swapchain_fn = khr::swapchain::Device::new(instance.handle(), logical_device);
        let (_, graphics_family) = renderer.graphics_queue()?;
        let physical_device_handle = renderer
            .physical_device()
            .ok_or(VulkanError::VulkanUnavailable)?
            .handle();
        // SAFETY: `physical_device_handle` and `graphics_family` belong to the renderer device,
        // and `surface` was created from the same instance.
        let present_supported = unsafe {
            surface_fn.get_physical_device_surface_support(physical_device_handle, graphics_family, surface)
        }
        .map_err(VulkanError::from)?;
        if !present_supported {
            return Err(VulkanError::QueueFamilyUnsupported.into());
        }

        // SAFETY: `logical_device` is the live renderer device.
        let acquire_fence = unsafe { logical_device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(VulkanError::from)?;

        let mut backend = Self {
            renderer,
            window,
            surface,
            surface_fn,
            swapchain_fn,
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            format: Fourcc::Abgr8888,
            extent: vk::Extent2D::default(),
            current_image: None,
            bound_target: None,
            acquire_fence,
        };
        backend.recreate_swapchain()?;
        Ok(backend)
    }

    /// Window size of the underlying window
    pub fn window_size(&self) -> Size<i32, Physical> {
        let (w, h): (i32, i32) = self.window.surface_size().into();
        (w, h).into()
    }

    /// Scale factor of the underlying window.
    pub fn scale_factor(&self) -> f64 {
        self.window.scale_factor()
    }

    /// Reference to the underlying window
    pub fn window(&self) -> &dyn WinitWindow {
        &*self.window
    }

    /// Access the underlying renderer
    pub fn renderer(&mut self) -> &mut VulkanRenderer {
        &mut self.renderer
    }

    /// Bind the current swapchain image to the renderer.
    #[instrument(level = "trace", skip(self))]
    #[profiling::function]
    pub fn bind(&mut self) -> Result<(&mut VulkanRenderer, VulkanRenderTarget<'_>), SwapBuffersError> {
        let window_size = self.window_size();
        if self.extent.width != window_size.w.max(0) as u32
            || self.extent.height != window_size.h.max(0) as u32
        {
            self.recreate_swapchain()?;
        }

        let logical_device = self.renderer.logical_device()?;
        // SAFETY: `acquire_fence` was created from `logical_device` and is not in use after the
        // previous acquire wait or swapchain recreate.
        unsafe { logical_device.reset_fences(&[self.acquire_fence]) }.map_err(VulkanError::from)?;
        let (index, _suboptimal) = match unsafe {
            self.swapchain_fn.acquire_next_image(
                self.swapchain,
                u64::MAX,
                vk::Semaphore::null(),
                self.acquire_fence,
            )
        } {
            Ok(acquired) => acquired,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain()?;
                return Err(SwapBuffersError::TemporaryFailure(Box::new(
                    VulkanError::UnsupportedOperation("swapchain out of date"),
                )));
            }
            Err(err) => return Err(VulkanError::from(err).into()),
        };
        // SAFETY: acquire was submitted against `acquire_fence` on this device.
        unsafe { logical_device.wait_for_fences(&[self.acquire_fence], true, u64::MAX) }
            .map_err(VulkanError::from)?;

        let image = *self
            .images
            .get(index as usize)
            .ok_or(VulkanError::UnsupportedOperation("swapchain image index"))?;
        let target =
            self.renderer
                .wrap_swapchain_image(image, (window_size.w, window_size.h).into(), self.format)?;
        self.renderer
            .transition_swapchain_target(&target, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)?;
        self.current_image = Some(index);
        self.bound_target = Some(target);
        let target = self
            .bound_target
            .as_mut()
            .expect("swapchain target was just stored");
        let fb = Bind::bind(&mut self.renderer, target)?;
        Ok((&mut self.renderer, fb))
    }

    /// Buffer age is not tracked for the Vulkan winit swapchain.
    pub fn buffer_age(&self) -> Option<usize> {
        Some(0)
    }

    /// Presents the bound swapchain image.
    #[instrument(level = "trace", skip(self, _damage))]
    #[profiling::function]
    pub fn submit(&mut self, _damage: Option<&[Rectangle<i32, Physical>]>) -> Result<(), SwapBuffersError> {
        let index = self
            .current_image
            .take()
            .ok_or(VulkanError::UnsupportedOperation("swapchain not bound"))?;
        let target = self
            .bound_target
            .take()
            .ok_or(VulkanError::UnsupportedOperation("swapchain not bound"))?;
        self.renderer
            .transition_swapchain_target(&target, vk::ImageLayout::PRESENT_SRC_KHR)?;

        self.window.pre_present_notify();
        let present_info = vk::PresentInfoKHR::default()
            .swapchains(std::slice::from_ref(&self.swapchain))
            .image_indices(std::slice::from_ref(&index));
        let (queue, _) = self.renderer.graphics_queue()?;
        // SAFETY: `queue` is the renderer graphics queue, which was checked to support present on
        // `surface`, and `index` was acquired from `swapchain`.
        match unsafe { self.swapchain_fn.queue_present(queue, &present_info) } {
            Ok(_) | Err(vk::Result::SUBOPTIMAL_KHR) => Ok(()),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain()?;
                Ok(())
            }
            Err(err) => Err(VulkanError::from(err).into()),
        }
    }

    fn recreate_swapchain(&mut self) -> Result<(), VulkanError> {
        self.current_image = None;
        self.bound_target = None;
        let physical_device = self
            .renderer
            .physical_device()
            .ok_or(VulkanError::VulkanUnavailable)?;
        let logical_device = self.renderer.logical_device()?;
        // SAFETY: no command buffers from this device are recorded; dropping `bound_target` above
        // released the previous swapchain image wrappers.
        unsafe { logical_device.device_wait_idle() }.map_err(VulkanError::from)?;

        // SAFETY: `surface` was created from the same instance as `physical_device`.
        let capabilities = unsafe {
            self.surface_fn
                .get_physical_device_surface_capabilities(physical_device.handle(), self.surface)
        }?;
        let formats = unsafe {
            self.surface_fn
                .get_physical_device_surface_formats(physical_device.handle(), self.surface)
        }?;
        let present_modes = unsafe {
            self.surface_fn
                .get_physical_device_surface_present_modes(physical_device.handle(), self.surface)
        }?;

        let window_size = self.window_size();
        let extent = vk::Extent2D {
            width: window_size.w.max(0).clamp(
                capabilities.min_image_extent.width as i32,
                capabilities.max_image_extent.width as i32,
            ) as u32,
            height: window_size.h.max(0).clamp(
                capabilities.min_image_extent.height as i32,
                capabilities.max_image_extent.height as i32,
            ) as u32,
        };
        if extent.width == 0 || extent.height == 0 {
            return Err(VulkanError::UnsupportedOperation("swapchain extent"));
        }

        let (vk_format, fourcc) = select_swapchain_format(&formats)?;
        let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        };
        let composite_alpha = select_composite_alpha(capabilities.supported_composite_alpha)?;
        let mut image_count = capabilities.min_image_count.saturating_add(1);
        if capabilities.max_image_count > 0 {
            image_count = image_count.min(capabilities.max_image_count);
        }

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(vk_format)
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(composite_alpha)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(self.swapchain);

        // SAFETY: `create_info` references a live surface and optional old swapchain from this
        // device. Image usage is color attachment only.
        let new_swapchain = unsafe { self.swapchain_fn.create_swapchain(&create_info, None) }?;
        if self.swapchain != vk::SwapchainKHR::null() {
            // SAFETY: `device_wait_idle` ran above; no remaining references to old swapchain images.
            unsafe { self.swapchain_fn.destroy_swapchain(self.swapchain, None) };
        }
        self.swapchain = new_swapchain;
        // SAFETY: `swapchain` was just created from `swapchain_fn`.
        self.images = unsafe { self.swapchain_fn.get_swapchain_images(self.swapchain) }?;
        self.extent = extent;
        self.format = fourcc;
        debug!(
            ?extent,
            ?vk_format,
            ?fourcc,
            images = self.images.len(),
            "Created Vulkan winit swapchain"
        );
        Ok(())
    }
}

impl Drop for WinitVulkanGraphicsBackend {
    fn drop(&mut self) {
        self.bound_target = None;
        self.current_image = None;
        if let Ok(device) = self.renderer.logical_device() {
            // SAFETY: the renderer created this device; waiting idle is valid while it is live.
            let _ = unsafe { device.device_wait_idle() };
            // SAFETY: `acquire_fence` was created from this device and is no longer waited on.
            unsafe { device.destroy_fence(self.acquire_fence, None) };
        }
        if self.swapchain != vk::SwapchainKHR::null() {
            // SAFETY: swapchain images are not wrapped after `bound_target` was dropped, and the
            // device was idled above when available.
            unsafe { self.swapchain_fn.destroy_swapchain(self.swapchain, None) };
        }
        // SAFETY: `surface` was created from `surface_fn` and the swapchain that used it is gone.
        unsafe { self.surface_fn.destroy_surface(self.surface, None) };
    }
}

fn vulkan_instance_error(err: InstanceError) -> VulkanError {
    match err {
        InstanceError::Load(_) => VulkanError::VulkanUnavailable,
        InstanceError::UnsupportedVersion => {
            VulkanError::DeviceInitializationFailed("Smithay requires at least Vulkan 1.1".to_owned())
        }
        InstanceError::Vk(result) => VulkanError::from(result),
    }
}

fn instance_extensions_for_window(
    window: &Arc<dyn WinitWindow>,
) -> Result<Vec<&'static std::ffi::CStr>, VulkanError> {
    let display = window
        .display_handle()
        .map_err(|_| VulkanError::UnsupportedOperation("window display handle"))?;
    let mut extensions = vec![khr::surface::NAME];
    match display.as_raw() {
        RawDisplayHandle::Wayland(_) => extensions.push(khr::wayland_surface::NAME),
        RawDisplayHandle::Xlib(_) => extensions.push(khr::xlib_surface::NAME),
        RawDisplayHandle::Xcb(_) => extensions.push(khr::xcb_surface::NAME),
        _ => return Err(VulkanError::UnsupportedOperation("window system")),
    }
    Ok(extensions)
}

unsafe fn create_window_surface(
    entry: &ash::Entry,
    instance: &Instance,
    window: &Arc<dyn WinitWindow>,
) -> Result<vk::SurfaceKHR, VulkanError> {
    let display = window
        .display_handle()
        .map_err(|_| VulkanError::UnsupportedOperation("window display handle"))?;
    let window_handle = window
        .window_handle()
        .map_err(|_| VulkanError::UnsupportedOperation("window handle"))?;
    match (display.as_raw(), window_handle.as_raw()) {
        (RawDisplayHandle::Wayland(display), RawWindowHandle::Wayland(window)) => {
            let create_info = vk::WaylandSurfaceCreateInfoKHR::default()
                .display(display.display.as_ptr().cast())
                .surface(window.surface.as_ptr().cast());
            let loader = khr::wayland_surface::Instance::new(entry, instance.handle());
            // SAFETY: `create_info` pointers come from the live winit window, and
            // `VK_KHR_wayland_surface` is enabled on `instance`.
            unsafe { loader.create_wayland_surface(&create_info, None) }.map_err(VulkanError::from)
        }
        (RawDisplayHandle::Xlib(display), RawWindowHandle::Xlib(window)) => {
            let create_info = vk::XlibSurfaceCreateInfoKHR::default()
                .dpy(display.display.unwrap().as_ptr().cast())
                .window(window.window);
            let loader = khr::xlib_surface::Instance::new(entry, instance.handle());
            // SAFETY: `dpy`/`window` come from the live winit Xlib window, and
            // `VK_KHR_xlib_surface` is enabled on `instance`.
            unsafe { loader.create_xlib_surface(&create_info, None) }.map_err(VulkanError::from)
        }
        (RawDisplayHandle::Xcb(display), RawWindowHandle::Xcb(window)) => {
            let create_info = vk::XcbSurfaceCreateInfoKHR::default()
                .connection(display.connection.unwrap().as_ptr().cast())
                .window(window.window.get());
            let loader = khr::xcb_surface::Instance::new(entry, instance.handle());
            // SAFETY: `connection`/`window` come from the live winit XCB window, and
            // `VK_KHR_xcb_surface` is enabled on `instance`.
            unsafe { loader.create_xcb_surface(&create_info, None) }.map_err(VulkanError::from)
        }
        _ => Err(VulkanError::UnsupportedOperation("window system")),
    }
}

fn select_physical_device(
    instance: &Instance,
    surface_fn: &khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<PhysicalDevice, VulkanError> {
    // Nested windows usually sit on the same GPU as the parent compositor. Prefer integrated
    // devices when several physical devices can present to the surface.
    PhysicalDevice::enumerate(instance)
        .map_err(VulkanError::from)?
        .filter(|device| {
            device.has_device_extension(khr::swapchain::NAME)
                && queue_supports_present(surface_fn, device, surface)
        })
        .min_by_key(|device| match device.ty() {
            vk::PhysicalDeviceType::INTEGRATED_GPU => 0u8,
            vk::PhysicalDeviceType::DISCRETE_GPU => 1,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
            _ => 3,
        })
        .ok_or(VulkanError::QueueFamilyUnsupported)
}

fn select_swapchain_format(formats: &[vk::SurfaceFormatKHR]) -> Result<(vk::Format, Fourcc), VulkanError> {
    const PREFERRED: [(vk::Format, Fourcc); 2] = [
        (vk::Format::R8G8B8A8_UNORM, Fourcc::Abgr8888),
        (vk::Format::B8G8R8A8_UNORM, Fourcc::Argb8888),
    ];
    for (vk_format, fourcc) in PREFERRED {
        if formats.iter().any(|format| {
            format.format == vk_format && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        }) {
            return Ok((vk_format, fourcc));
        }
    }
    Err(VulkanError::UnsupportedOperation("swapchain format"))
}

fn select_composite_alpha(
    supported: vk::CompositeAlphaFlagsKHR,
) -> Result<vk::CompositeAlphaFlagsKHR, VulkanError> {
    const PREFERRED: [vk::CompositeAlphaFlagsKHR; 4] = [
        vk::CompositeAlphaFlagsKHR::OPAQUE,
        vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::INHERIT,
    ];
    PREFERRED
        .into_iter()
        .find(|mode| supported.contains(*mode))
        .ok_or(VulkanError::UnsupportedOperation("swapchain composite alpha"))
}

fn queue_supports_present(
    surface_fn: &khr::surface::Instance,
    device: &PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> bool {
    let properties = unsafe {
        device
            .instance()
            .handle()
            .get_physical_device_queue_family_properties(device.handle())
    };
    properties.iter().enumerate().any(|(index, properties)| {
        properties.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            && unsafe {
                surface_fn
                    .get_physical_device_surface_support(device.handle(), index as u32, surface)
                    .unwrap_or(false)
            }
    })
}
