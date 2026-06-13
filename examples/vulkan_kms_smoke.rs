//! Minimal real-screen Vulkan/DRM smoke test.
//!
//! This example is intended to be run manually from an active physical VT with DRM master
//! permissions. It opens a DRM card node through Smithay's libseat session backend, may take over
//! the connected display, and does not restore previous KMS state. It renders a solid colour into a
//! GBM scanout buffer through [`VulkanRenderer`]'s public `Bind<Dmabuf>` path and commits it with
//! KMS.

use std::{env, error::Error, path::Path, thread, time::Duration};

use ash::ext;
use smithay::{
    backend::{
        allocator::{
            Fourcc,
            gbm::{GbmAllocator, GbmBufferFlags, GbmDevice},
        },
        drm::{
            DrmDevice, DrmDeviceFd, DrmNode,
            compositor::{FrameFlags, PrimaryPlaneElement},
            exporter::gbm::GbmFramebufferExporter,
            output::{DrmOutputManager, DrmOutputRenderElements},
        },
        renderer::{
            Bind, Color32F,
            element::{Id, Kind, solid::SolidColorRenderElement},
            vulkan::VulkanRenderer,
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

    let (drm, _notifier) = DrmDevice::new(drm_fd.clone(), false)?;
    let gbm = GbmDevice::new(drm_fd)?;
    let allocator = GbmAllocator::new(gbm.clone(), GbmBufferFlags::RENDERING | GbmBufferFlags::SCANOUT);
    let exporter = GbmFramebufferExporter::new(gbm.clone(), None.into());

    let instance = Instance::new(Version::VERSION_1_3, None)?;
    let physical_device = physical_device_for_node(&instance, drm_node)?;
    let mut renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device)
        .build()?;

    let renderer_formats =
        <VulkanRenderer as Bind<smithay::backend::allocator::dmabuf::Dmabuf>>::supported_formats(&renderer)
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

    let mut output_manager = DrmOutputManager::<_, _, (), _>::new(
        drm,
        allocator,
        exporter,
        Some(gbm),
        color_formats,
        renderer_formats,
    );
    let mode_source = OutputModeSource::Static {
        size,
        scale: 1.0.into(),
        transform: Transform::Normal,
    };
    let mut output = output_manager.lock().initialize_output(
        crtc,
        mode,
        &[connector],
        mode_source,
        None,
        &mut renderer,
        &DrmOutputRenderElements::<VulkanRenderer, SolidColorRenderElement>::default(),
    )?;

    let element = SolidColorRenderElement::new(
        Id::new(),
        Rectangle::from_size(size),
        1usize,
        Color32F::new(0.0, 0.25, 0.8, 1.0),
        Kind::Unspecified,
    );
    let elements = [element];
    let frame = output.render_frame(
        &mut renderer,
        &elements,
        Color32F::new(0.0, 0.0, 0.0, 1.0),
        FrameFlags::empty(),
    )?;
    if frame.needs_sync() {
        if let PrimaryPlaneElement::Swapchain(primary) = frame.primary_element {
            primary.sync.wait()?;
        }
    }
    output.commit_frame()?;

    tracing::info!(
        ?device_path,
        ?connector,
        ?crtc,
        ?size,
        seconds,
        "Vulkan KMS smoke frame committed"
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
