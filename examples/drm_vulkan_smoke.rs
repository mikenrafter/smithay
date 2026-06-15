//! Minimal real-screen DRM/Vulkan smoke test.
//!
//! This is local hardware bring-up tooling, not a normal compositor example. Run it manually from an
//! active physical VT with DRM master permissions. It opens a DRM card node through Smithay's
//! libseat session backend, may take over the connected display, and does not restore previous DRM
//! state. It renders a solid colour into a GBM scanout buffer through [`VulkanRenderer`]'s public
//! dmabuf render-target development path and commits it with DRM.

use std::{env, error::Error, path::Path, thread, time::Duration};

use ash::ext;
use smithay::{
    backend::{
        allocator::{
            Fourcc,
            dmabuf::Dmabuf,
            gbm::{GbmAllocator, GbmBufferFlags, GbmDevice},
        },
        drm::{
            DrmDevice, DrmDeviceFd, DrmNode,
            compositor::{DrmCompositor, DrmRenderTarget, FrameFlags, PrimaryPlaneElement},
            exporter::gbm::GbmFramebufferExporter,
        },
        renderer::{
            Color32F,
            element::{Id, Kind, solid::SolidColorRenderElement},
            vulkan::{VulkanOwnedDmabufRenderTarget, VulkanRenderer},
        },
        session::{Session, libseat::LibSeatSession},
        vulkan::{Instance, PhysicalDevice, version::Version},
    },
    output::OutputModeSource,
    reexports::{
        drm::control::{Device as ControlDevice, ModeTypeFlags, connector, crtc},
        rustix::fs::OFlags,
    },
    utils::{DeviceFd, Physical, Rectangle, Size, Transform},
};

#[derive(Debug, Default)]
struct VulkanDiscardRenderTarget;

impl DrmRenderTarget<VulkanRenderer> for VulkanDiscardRenderTarget {
    type Target = VulkanOwnedDmabufRenderTarget<'static>;

    fn target_from_dmabuf(&mut self, dmabuf: Dmabuf) -> Self::Target {
        unsafe {
            // SAFETY: `DrmCompositor` only calls this for a swapchain slot it acquired for the
            // primary plane through Smithay's normal GBM swapchain path. This smoke test always does
            // a full repaint and discards previous contents, so the Vulkan acquire path may treat
            // previous contents as undefined instead of requiring a preserved layout. Successful
            // Vulkan frames release the image to `VK_QUEUE_FAMILY_FOREIGN_EXT` in `GENERAL` before
            // KMS sees the framebuffer; reused slots have already left scanout before the swapchain
            // returns them, so no additional acquire sync point is available or required here.
            VulkanOwnedDmabufRenderTarget::discard(dmabuf)
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    if let Ok(env_filter) = tracing_subscriber::EnvFilter::try_from_default_env() {
        tracing_subscriber::fmt().with_env_filter(env_filter).init();
    } else {
        tracing_subscriber::fmt().init();
    }

    let mut args = env::args().skip(1);
    let device_path = args.next().unwrap_or_else(|| "/dev/dri/card0".to_owned());
    let seconds = args
        .next()
        .and_then(|seconds| seconds.parse::<u64>().ok())
        .unwrap_or(5);

    let (mut session, _session_notifier) = LibSeatSession::new()?;
    if !session.is_active() {
        return Err(format!(
            "session for seat {} is not active; run from an active physical VT or via a seat manager",
            session.seat()
        )
        .into());
    }
    let fd = session.open(
        Path::new(&device_path),
        OFlags::RDWR | OFlags::CLOEXEC | OFlags::NOCTTY | OFlags::NONBLOCK,
    )?;
    let drm_fd = DrmDeviceFd::new(DeviceFd::from(fd));
    let drm_node = DrmNode::from_path(&device_path)?;
    let (connector, crtc, mode) = pick_connector_crtc_mode(&drm_fd)?;
    let (width, height) = mode.size();
    let size = Size::<i32, Physical>::from((width as i32, height as i32));

    let (mut drm, _notifier) = DrmDevice::new(drm_fd.clone(), false)?;
    let gbm = GbmDevice::new(drm_fd)?;
    let allocator = GbmAllocator::new(gbm.clone(), GbmBufferFlags::RENDERING | GbmBufferFlags::SCANOUT);
    let exporter = GbmFramebufferExporter::new(gbm.clone(), None.into());

    let instance = Instance::new(Version::VERSION_1_3, None)?;
    let physical_device = physical_device_for_node(&instance, drm_node)?;
    let mut renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device)
        .build()?;

    let renderer_formats = VulkanDiscardRenderTarget
        .supported_formats(&renderer)
        .unwrap_or_default();
    if renderer_formats.iter().next().is_none() {
        return Err("Vulkan renderer did not advertise any dmabuf render-target formats".into());
    }
    let color_formats = renderer_formats
        .iter()
        .map(|format| format.code)
        .filter(|format| matches!(*format, Fourcc::Abgr8888 | Fourcc::Argb8888))
        .collect::<Vec<_>>();
    if color_formats.is_empty() {
        return Err("Vulkan renderer did not advertise an 8-bit ARGB/ABGR dmabuf target format".into());
    }

    let mode_source = OutputModeSource::Static {
        size,
        scale: 1.0.into(),
        transform: Transform::Normal,
    };
    let cursor_size = drm.cursor_size();
    let surface = drm.create_surface(crtc, mode, &[connector])?;
    let mut compositor = DrmCompositor::<_, _, (), _>::new(
        mode_source,
        surface,
        None,
        allocator,
        exporter,
        color_formats,
        renderer_formats,
        cursor_size,
        Some(gbm),
    )?;

    let element = SolidColorRenderElement::new(
        Id::new(),
        Rectangle::from_size(size),
        1usize,
        Color32F::new(0.0, 0.25, 0.8, 1.0),
        Kind::Unspecified,
    );
    let elements = [element];
    let mut render_target = VulkanDiscardRenderTarget;
    let frame = compositor.render_frame_with_render_target(
        &mut renderer,
        &mut render_target,
        &elements,
        Color32F::new(0.0, 0.0, 0.0, 1.0),
        FrameFlags::empty(),
    )?;
    if frame.needs_sync() {
        if let PrimaryPlaneElement::Swapchain(primary) = frame.primary_element {
            primary.sync.wait()?;
        }
    }
    compositor.commit_frame()?;

    tracing::info!(
        ?device_path,
        ?connector,
        ?crtc,
        ?size,
        seconds,
        "DRM Vulkan smoke frame committed"
    );
    thread::sleep(Duration::from_secs(seconds));
    Ok(())
}

fn pick_connector_crtc_mode(
    drm: &DrmDeviceFd,
) -> Result<
    (
        connector::Handle,
        crtc::Handle,
        smithay::reexports::drm::control::Mode,
    ),
    Box<dyn Error>,
> {
    let resources = drm.resource_handles()?;
    for connector_handle in resources.connectors() {
        let connector = drm.get_connector(*connector_handle, true)?;
        if connector.state() != connector::State::Connected || connector.modes().is_empty() {
            continue;
        }

        let crtc = connector
            .current_encoder()
            .and_then(|encoder| drm.get_encoder(encoder).ok())
            .and_then(|encoder| encoder.crtc())
            .or_else(|| {
                connector.encoders().iter().find_map(|encoder| {
                    drm.get_encoder(*encoder).ok().and_then(|encoder| {
                        resources
                            .filter_crtcs(encoder.possible_crtcs())
                            .into_iter()
                            .next()
                    })
                })
            });
        let Some(crtc) = crtc else {
            continue;
        };
        let mode = connector
            .modes()
            .iter()
            .copied()
            .find(|mode| mode.mode_type().contains(ModeTypeFlags::PREFERRED))
            .unwrap_or_else(|| connector.modes()[0]);

        return Ok((*connector_handle, crtc, mode));
    }

    Err("no connected DRM connector with a usable CRTC/mode".into())
}

fn physical_device_for_node(instance: &Instance, node: DrmNode) -> Result<PhysicalDevice, Box<dyn Error>> {
    PhysicalDevice::enumerate(instance)?
        .filter(|physical_device| physical_device.has_device_extension(ext::physical_device_drm::NAME))
        .find(|physical_device| {
            physical_device.primary_node().ok().flatten() == Some(node)
                || physical_device.render_node().ok().flatten() == Some(node)
        })
        .ok_or_else(|| format!("no Vulkan physical device matched DRM node {node}").into())
}
