//! Minimal real-screen DRM/Vulkan smoke test.
//!
//! This is local hardware bring-up tooling, not a normal compositor example. Run it manually from an
//! active physical VT with DRM master permissions. It opens a DRM card node through Smithay's
//! libseat session backend, may take over the connected display, and does not restore previous DRM
//! state. It renders a solid colour into a GBM scanout buffer through Smithay's normal
//! [`Bind<Dmabuf>`] DRM compositor path and commits it with DRM.
//!
//! Set `SMITHAY_DRM_VULKAN_ALLOCATOR_PROBE=1` to stop after probing the Vulkan allocator path through
//! `DrmCompositor::new`. That mode expects the current fail-closed Vulkan allocator framebuffer export
//! guard and does not render or commit a frame.
//! Set `SMITHAY_DRM_VULKAN_ALLOCATOR_METADATA_PROBE=1` to test only whether a single explicit-modifier
//! Vulkan-exported dmabuf can be imported by GBM and added as a DRM framebuffer, then immediately
//! destroyed. That mode is metadata evidence only; it is not presentation or reuse evidence.

use std::{env, error::Error, path::Path, thread, time::Duration};

use ash::ext;
use smithay::{
    backend::{
        allocator::{
            Allocator, Fourcc, Modifier,
            dmabuf::{AsDmabuf, Dmabuf},
            format::FormatSet,
            gbm::{GbmAllocator, GbmBufferFlags, GbmDevice},
            vulkan::{ImageUsageFlags, VulkanAllocator},
        },
        drm::{
            DrmDevice, DrmDeviceFd, DrmNode,
            compositor::{DrmCompositor, FrameError, FrameFlags, PrimaryPlaneElement},
            exporter::gbm::{GbmFramebufferExporter, VulkanError as GbmVulkanError},
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
    utils::{Buffer as BufferCoords, DeviceFd, Physical, Rectangle, Size, Transform},
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
    let open_flags = OFlags::RDWR | OFlags::CLOEXEC | OFlags::NOCTTY | OFlags::NONBLOCK;
    let fd = session.open(Path::new(&device_path), open_flags)?;
    let drm_fd = DrmDeviceFd::new(DeviceFd::from(fd));
    let drm_node = DrmNode::from_path(&device_path)?;
    let (connector, crtc, mode) = pick_connector_crtc_mode(&drm_fd)?;
    let (width, height) = mode.size();
    let size = Size::<i32, Physical>::from((width as i32, height as i32));

    let (mut drm, _notifier) = DrmDevice::new(drm_fd.clone(), false)?;
    let gbm = GbmDevice::new(drm_fd)?;
    let exporter = GbmFramebufferExporter::new(gbm.clone(), None.into());

    let instance = Instance::new(Version::VERSION_1_3, None)?;
    let physical_device = physical_device_for_node(&instance, drm_node)?;
    let mut renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device.clone())
        .build()?;

    let renderer_formats = <VulkanRenderer as Bind<Dmabuf>>::supported_formats(&renderer).unwrap_or_default();
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
    let cursor_size = drm.cursor_size();

    if env::var_os("SMITHAY_DRM_VULKAN_ALLOCATOR_METADATA_PROBE").is_some() {
        return probe_vulkan_allocator_framebuffer_metadata(
            &drm,
            crtc,
            &gbm,
            physical_device,
            renderer_formats,
        );
    }

    if env::var_os("SMITHAY_DRM_VULKAN_ALLOCATOR_PROBE").is_some() {
        return probe_vulkan_allocator_framebuffer_guard(
            &mut drm,
            crtc,
            mode,
            connector,
            size,
            cursor_size,
            gbm,
            exporter,
            physical_device,
            renderer_formats,
        );
    }

    let allocator = GbmAllocator::new(gbm.clone(), GbmBufferFlags::RENDERING | GbmBufferFlags::SCANOUT);
    let mode_source = OutputModeSource::Static {
        size,
        scale: 1.0.into(),
        transform: Transform::Normal,
    };
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
    let frame = compositor.render_frame(
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

fn probe_vulkan_allocator_framebuffer_metadata(
    drm: &DrmDevice,
    crtc: crtc::Handle,
    gbm: &GbmDevice<DrmDeviceFd>,
    physical_device: PhysicalDevice,
    renderer_formats: FormatSet,
) -> Result<(), Box<dyn Error>> {
    let usage = ImageUsageFlags::COLOR_ATTACHMENT;
    let mut allocator = VulkanAllocator::new(&physical_device, usage)?;
    let planes = drm.planes(&crtc)?;
    let primary_plane = planes
        .primary
        .first()
        .ok_or("DRM device did not report a primary plane for the selected CRTC")?;
    let formats = renderer_formats
        .iter()
        .copied()
        .filter(|format| {
            format.modifier != Modifier::Invalid
                && matches!(format.code, Fourcc::Abgr8888 | Fourcc::Argb8888)
                && primary_plane.formats.contains(format)
                && allocator.is_format_supported(*format, usage)
        })
        .collect::<Vec<_>>();
    if formats.is_empty() {
        return Err(
            "Vulkan allocator and renderer did not share an explicit-modifier 8-bit ARGB/ABGR target format"
                .into(),
        );
    }

    let mut errors = Vec::new();
    for format in formats {
        let image = match allocator.create_buffer(64, 64, format.code, &[format.modifier]) {
            Ok(image) => image,
            Err(err) => {
                errors.push(format!("{format:?}: create buffer failed: {err}"));
                continue;
            }
        };
        let dmabuf = match image.export() {
            Ok(dmabuf) => dmabuf,
            Err(err) => {
                errors.push(format!("{format:?}: export dmabuf failed: {err}"));
                continue;
            }
        };
        if dmabuf.num_planes() != 1 {
            errors.push(format!(
                "{format:?}: metadata probe currently requires a single-plane Vulkan dmabuf"
            ));
            continue;
        }

        match smithay::backend::drm::gbm::framebuffer_from_dmabuf(drm.device_fd(), gbm, &dmabuf, false, false)
        {
            Ok(framebuffer) => {
                drop(framebuffer);

                tracing::info!(
                    ?format,
                    "Vulkan allocator metadata probe imported dmabuf and created/destroyed DRM framebuffer"
                );
                return Ok(());
            }
            Err(err) => {
                errors.push(format!("{format:?}: framebuffer import failed: {err}"));
            }
        }
    }

    Err(format!(
        "no explicit-modifier Vulkan allocator metadata candidate produced a DRM framebuffer; attempted {} candidates: {}",
        errors.len(),
        errors.join("; ")
    )
    .into())
}

#[allow(clippy::too_many_arguments)]
fn probe_vulkan_allocator_framebuffer_guard(
    drm: &mut DrmDevice,
    crtc: crtc::Handle,
    mode: smithay::reexports::drm::control::Mode,
    connector: connector::Handle,
    size: Size<i32, Physical>,
    cursor_size: Size<u32, BufferCoords>,
    gbm: GbmDevice<DrmDeviceFd>,
    exporter: GbmFramebufferExporter<DrmDeviceFd>,
    physical_device: PhysicalDevice,
    renderer_formats: FormatSet,
) -> Result<(), Box<dyn Error>> {
    let usage = ImageUsageFlags::COLOR_ATTACHMENT;
    let allocator = VulkanAllocator::new(&physical_device, usage)?;
    let color_formats = renderer_formats
        .iter()
        .filter(|format| allocator.is_format_supported(**format, usage))
        .map(|format| format.code)
        .filter(|format| matches!(*format, Fourcc::Abgr8888 | Fourcc::Argb8888))
        .collect::<Vec<_>>();
    if color_formats.is_empty() {
        return Err("Vulkan allocator and renderer did not share an 8-bit ARGB/ABGR target format".into());
    }

    let mode_source = OutputModeSource::Static {
        size,
        scale: 1.0.into(),
        transform: Transform::Normal,
    };
    let surface = drm.create_surface(crtc, mode, &[connector])?;

    match DrmCompositor::<VulkanAllocator, GbmFramebufferExporter<DrmDeviceFd>, (), DrmDeviceFd>::new(
        mode_source,
        surface,
        None,
        allocator,
        exporter,
        color_formats,
        renderer_formats,
        cursor_size,
        Some(gbm),
    ) {
        Err(FrameError::FramebufferExport(GbmVulkanError::MissingCapability(
            "Vulkan allocator DRM framebuffer external-state contract",
        ))) => {
            tracing::info!("Vulkan allocator DRM compositor probe reached expected framebuffer export guard");
            Ok(())
        }
        Ok(_) => Err("Vulkan allocator DRM compositor unexpectedly passed framebuffer export guard".into()),
        Err(err) => Err(format!("unexpected Vulkan allocator DRM compositor probe error: {err:?}").into()),
    }
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
