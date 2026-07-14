//! Minimal real-screen DRM/Vulkan smoke test.
//!
//! This is local hardware bring-up tooling, not a normal compositor example. Run it manually from an
//! active physical VT with DRM master permissions. It opens a DRM card node through Smithay's
//! libseat session backend, may take over the connected display, and does not restore previous DRM
//! state. It renders a solid colour into a GBM scanout buffer through Smithay's normal
//! [`Bind<Dmabuf>`] DRM compositor path, queues it with DRM, and waits for the page-flip event.
//!
//! Set `SMITHAY_DRM_VULKAN_ALLOCATOR_PROBE=1` to stop after probing the Vulkan allocator path through
//! `DrmCompositor::new`. That mode expects the current fail-closed Vulkan allocator framebuffer export
//! guard and does not render or commit a frame.
//! Set `SMITHAY_DRM_VULKAN_ALLOCATOR_METADATA_PROBE=1` to test only whether a single explicit-modifier
//! Vulkan-exported dmabuf can be imported by GBM and added as a DRM framebuffer, then immediately
//! destroyed. That mode reports GBM-allocation and GBM-exported dmabuf baselines, GBM-import vs
//! AddFB2 errno details, a GBM-imported BO metadata AddFB2 comparison, and a direct PRIME fd -> GEM
//! handle AddFB2 comparison as metadata evidence only; it is not presentation or reuse evidence.
//! Set `SMITHAY_DRM_VULKAN_GBM_TARGET_PROBE=1` to bind a GBM-allocated scanout dmabuf as a Vulkan
//! render target, clear it, finish, and release it without a DRM commit.
//! Set `SMITHAY_DRM_VULKAN_RENDER_FRAME_PROBE=1` to run the normal `DrmCompositor::render_frame`
//! path and wait for render completion without committing the frame.
//! Set `SMITHAY_DRM_VULKAN_PAGEFLIP_PROBE=1` to queue two normal DRM frames, wait for page-flip
//! events, and call `frame_submitted` after each event. The probe logs whether Smithay expects to
//! submit a KMS `IN_FENCE_FD` for each frame.
//! Set `SMITHAY_DRM_VULKAN_REUSE_PROBE=1` to queue several normal DRM frames and require at least
//! one swapchain buffer object to be reused after page-flip submission. Override the frame count with
//! `SMITHAY_DRM_VULKAN_PAGEFLIP_FRAMES`.
//! Any mode that queues a DRM pageflip requires `SMITHAY_DRM_VULKAN_REAL_GPU_PAGEFLIP_OK=1`. Prefer
//! Mesa llvmpipe/lavapipe software Vulkan checks and no-commit probes before setting that flag.

use std::{
    env,
    error::Error,
    os::unix::io::AsRawFd,
    path::Path,
    thread,
    time::{Duration, Instant},
};

use ash::ext;
use smithay::{
    backend::{
        allocator::{
            Allocator, Buffer as AllocatorBuffer, Fourcc, Modifier,
            dmabuf::{AsDmabuf, Dmabuf},
            format::FormatSet,
            gbm::{GbmAllocator, GbmBuffer, GbmBufferFlags, GbmDevice},
            vulkan::{ImageUsageFlags, VulkanAllocator},
        },
        drm::{
            DrmDevice, DrmDeviceFd, DrmDeviceNotifier, DrmEvent, DrmEventMetadata, DrmNode,
            compositor::{DrmCompositor, FrameError, FrameFlags, PrimaryPlaneElement},
            exporter::gbm::{GbmFramebufferExporter, VulkanError as GbmVulkanError},
            gbm::Error as DrmGbmError,
        },
        renderer::{
            Bind, Color32F, Frame, RenderTargetLifecycle, Renderer,
            element::{Id, Kind, solid::SolidColorRenderElement},
            vulkan::{VulkanDmabufRenderTarget, VulkanRenderer},
        },
        session::{Session, libseat::LibSeatSession},
        vulkan::{Instance, PhysicalDevice, version::Version},
    },
    output::OutputModeSource,
    reexports::{
        calloop,
        drm::{
            buffer,
            control::{Device as ControlDevice, FbCmd2Flags, ModeTypeFlags, connector, crtc},
        },
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

    let gbm_target_probe = env::var_os("SMITHAY_DRM_VULKAN_GBM_TARGET_PROBE").is_some();
    let allocator_metadata_probe = env::var_os("SMITHAY_DRM_VULKAN_ALLOCATOR_METADATA_PROBE").is_some();
    let allocator_probe = env::var_os("SMITHAY_DRM_VULKAN_ALLOCATOR_PROBE").is_some();
    let render_frame_only_probe = env::var_os("SMITHAY_DRM_VULKAN_RENDER_FRAME_PROBE").is_some();
    let pageflip_probe = env::var_os("SMITHAY_DRM_VULKAN_PAGEFLIP_PROBE").is_some();
    let reuse_probe = env::var_os("SMITHAY_DRM_VULKAN_REUSE_PROBE").is_some();

    let mode = if gbm_target_probe {
        "gbm-target-no-commit"
    } else if allocator_metadata_probe {
        "allocator-metadata-no-commit"
    } else if allocator_probe {
        "allocator-guard-no-commit"
    } else if reuse_probe {
        "pageflip-reuse"
    } else if pageflip_probe {
        "pageflip"
    } else if render_frame_only_probe {
        "render-frame-no-commit"
    } else {
        "default-pageflip"
    };
    let queues_pageflip = !gbm_target_probe
        && !allocator_metadata_probe
        && !allocator_probe
        && (pageflip_probe || reuse_probe || !render_frame_only_probe);
    let real_gpu_pageflip_acknowledged = real_gpu_pageflip_acknowledged();
    tracing::info!(
        mode,
        queues_pageflip,
        real_gpu_pageflip_acknowledged,
        ?device_path,
        "DRM Vulkan smoke selected mode"
    );
    if queues_pageflip {
        require_real_gpu_pageflip_ack()?;
    }

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

    let (mut drm, drm_notifier) = DrmDevice::new(drm_fd.clone(), false)?;
    let gbm = GbmDevice::new(drm_fd)?;
    let exporter = GbmFramebufferExporter::new(gbm.clone(), None.into());

    let instance = Instance::new(Version::VERSION_1_3, None)?;
    let physical_device = physical_device_for_node(&instance, drm_node)?;
    let mut renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device.clone())
        .build()?;

    let renderer_formats =
        <VulkanRenderer as Bind<VulkanDmabufRenderTarget<'static, 'static>>>::supported_formats(&renderer)
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
    let cursor_size = drm.cursor_size();

    if gbm_target_probe {
        return probe_gbm_dmabuf_vulkan_render_target(&mut drm, &gbm, &mut renderer, renderer_formats);
    }

    if allocator_metadata_probe {
        return probe_vulkan_allocator_framebuffer_metadata(
            &mut drm,
            crtc,
            &gbm,
            physical_device,
            renderer_formats,
        );
    }

    if allocator_probe {
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
    let mut compositor = DrmCompositor::<_, _, usize, _>::new(
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
    let _pause_drm_on_return = render_frame_only_probe.then(|| PauseDrmOnReturn(&mut drm));

    let element = SolidColorRenderElement::new(
        Id::new(),
        Rectangle::from_size(size),
        1usize,
        Color32F::new(0.0, 0.25, 0.8, 1.0),
        Kind::Unspecified,
    );
    let elements = [element];

    if pageflip_probe || reuse_probe {
        let (mut event_loop, mut event_state) = pageflip_event_loop(drm_notifier, crtc)?;
        let frame_count = env::var("SMITHAY_DRM_VULKAN_PAGEFLIP_FRAMES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(if reuse_probe { 8 } else { 2 });
        if frame_count < 2 {
            return Err("pageflip probe requires at least two frames".into());
        }

        let mut previous_sequence = None;
        let mut swapchain_buffer_ids = Vec::new();
        let mut observed_reuse = false;
        for frame_index in 0..frame_count {
            let element = SolidColorRenderElement::new(
                Id::new(),
                Rectangle::from_size(size),
                frame_index + 1,
                probe_frame_color(frame_index),
                Kind::Unspecified,
            );
            let elements = [element];
            let frame = compositor.render_frame(
                &mut renderer,
                &elements,
                Color32F::new(0.0, 0.0, 0.0, 1.0),
                FrameFlags::empty(),
            )?;
            let needs_sync = frame.needs_sync();
            let mut primary_was_swapchain = false;
            let mut waited_render_sync = false;
            let mut swapchain_buffer_id = None;
            let mut sync_contains_fence = false;
            let mut sync_exportable = false;
            let mut kms_in_fence_expected = false;
            if let PrimaryPlaneElement::Swapchain(primary) = frame.primary_element {
                primary_was_swapchain = true;
                let buffer_id = primary.buffer() as *const _ as usize;
                observed_reuse |= swapchain_buffer_ids.contains(&buffer_id);
                swapchain_buffer_ids.push(buffer_id);
                swapchain_buffer_id = Some(buffer_id);
                sync_contains_fence = primary.sync.contains_fence();
                sync_exportable = primary.sync.is_exportable();
                kms_in_fence_expected = sync_exportable && !needs_sync;
                if needs_sync {
                    wait_sync_point(&primary.sync, "pageflip probe primary swapchain sync")?;
                    waited_render_sync = true;
                }
            }
            if !primary_was_swapchain {
                return Err(
                    format!("pageflip probe frame {frame_index} did not use the primary swapchain").into(),
                );
            }

            compositor.queue_frame(frame_index)?;
            let metadata = wait_for_pageflip_event(&mut event_loop, &mut event_state)?
                .ok_or("pageflip probe received VBlank without metadata")?;
            if let Some(previous_sequence) = previous_sequence {
                if metadata.sequence <= previous_sequence {
                    return Err(format!(
                        "pageflip probe sequence did not advance: previous={previous_sequence}, current={}",
                        metadata.sequence
                    )
                    .into());
                }
            }
            previous_sequence = Some(metadata.sequence);
            let submitted = compositor.frame_submitted()?;
            if submitted != Some(frame_index) {
                return Err(format!(
                    "pageflip probe frame_submitted returned {submitted:?}, expected Some({frame_index})"
                )
                .into());
            }
            tracing::info!(
                ?device_path,
                ?connector,
                ?crtc,
                ?size,
                frame_index,
                primary_was_swapchain,
                needs_sync,
                sync_contains_fence,
                sync_exportable,
                kms_in_fence_expected,
                waited_render_sync,
                ?swapchain_buffer_id,
                ?metadata,
                submitted,
                "DRM Vulkan smoke queued frame, observed pageflip, and submitted frame"
            );
        }
        if reuse_probe && !observed_reuse {
            return Err(format!(
                "reuse probe did not observe swapchain buffer reuse after {frame_count} frames; ids={swapchain_buffer_ids:?}"
            )
            .into());
        }

        tracing::info!(
            ?device_path,
            ?connector,
            ?crtc,
            ?size,
            frame_count,
            reuse_probe,
            observed_reuse,
            seconds,
            "DRM Vulkan smoke pageflip probe completed"
        );
        thread::sleep(Duration::from_secs(seconds));
        return Ok(());
    }

    let frame = compositor.render_frame(
        &mut renderer,
        &elements,
        Color32F::new(0.0, 0.0, 0.0, 1.0),
        FrameFlags::empty(),
    )?;
    let needs_sync = frame.needs_sync();
    let mut primary_was_swapchain = false;
    let mut waited_render_sync = false;
    let mut sync_contains_fence = false;
    let mut sync_exportable = false;
    let mut kms_in_fence_expected = false;
    if let PrimaryPlaneElement::Swapchain(primary) = frame.primary_element {
        primary_was_swapchain = true;
        sync_contains_fence = primary.sync.contains_fence();
        sync_exportable = primary.sync.is_exportable();
        kms_in_fence_expected = sync_exportable && !needs_sync;
        if render_frame_only_probe || needs_sync {
            wait_sync_point(&primary.sync, "render_frame primary swapchain sync")?;
            waited_render_sync = true;
        }
    }
    if render_frame_only_probe {
        tracing::info!(
            ?device_path,
            ?connector,
            ?crtc,
            ?size,
            needs_sync,
            sync_contains_fence,
            sync_exportable,
            kms_in_fence_expected,
            waited_render_sync,
            "DRM Vulkan smoke render_frame completed without commit"
        );
        return Ok(());
    }
    if !primary_was_swapchain {
        return Err("default smoke frame did not use the primary swapchain".into());
    }
    let (mut event_loop, mut event_state) = pageflip_event_loop(drm_notifier, crtc)?;
    compositor.queue_frame(0)?;
    let metadata = wait_for_pageflip_event(&mut event_loop, &mut event_state)?
        .ok_or("default smoke received VBlank without metadata")?;
    let submitted = compositor.frame_submitted()?;
    if submitted != Some(0) {
        return Err(format!("default smoke frame_submitted returned {submitted:?}, expected Some(0)").into());
    }

    tracing::info!(
        ?device_path,
        ?connector,
        ?crtc,
        ?size,
        primary_was_swapchain,
        needs_sync,
        sync_contains_fence,
        sync_exportable,
        kms_in_fence_expected,
        waited_render_sync,
        ?metadata,
        submitted,
        seconds,
        "DRM Vulkan smoke frame queued, pageflipped, and submitted"
    );
    thread::sleep(Duration::from_secs(seconds));
    Ok(())
}

fn probe_frame_color(frame_index: usize) -> Color32F {
    match frame_index % 3 {
        0 => Color32F::new(0.0, 0.25, 0.8, 1.0),
        1 => Color32F::new(0.8, 0.25, 0.0, 1.0),
        _ => Color32F::new(0.1, 0.6, 0.2, 1.0),
    }
}

fn require_real_gpu_pageflip_ack() -> Result<(), Box<dyn Error>> {
    if real_gpu_pageflip_acknowledged() {
        return Ok(());
    }

    Err("DRM pageflip probes submit work to the real GPU/display; run llvmpipe-safe checks and no-commit probes first, then set SMITHAY_DRM_VULKAN_REAL_GPU_PAGEFLIP_OK=1 to acknowledge the risk".into())
}

fn real_gpu_pageflip_acknowledged() -> bool {
    matches!(
        env::var("SMITHAY_DRM_VULKAN_REAL_GPU_PAGEFLIP_OK").as_deref(),
        Ok("1")
    )
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

fn probe_gbm_dmabuf_vulkan_render_target(
    drm: &mut DrmDevice,
    gbm: &GbmDevice<DrmDeviceFd>,
    renderer: &mut VulkanRenderer,
    renderer_formats: FormatSet,
) -> Result<(), Box<dyn Error>> {
    let _pause_drm_on_return = PauseDrmOnReturn(drm);
    let mut allocator = GbmAllocator::new(gbm.clone(), GbmBufferFlags::RENDERING | GbmBufferFlags::SCANOUT);
    let formats = renderer_formats
        .iter()
        .copied()
        .filter(|format| format.modifier != Modifier::Invalid)
        .filter(|format| matches!(format.code, Fourcc::Abgr8888 | Fourcc::Argb8888))
        .collect::<Vec<_>>();
    if formats.is_empty() {
        return Err(
            "Vulkan renderer did not advertise an explicit-modifier 8-bit ARGB/ABGR dmabuf target format"
                .into(),
        );
    }

    let mut errors = Vec::new();
    for format in formats {
        for (width, height) in [(64, 64), (256, 256)] {
            let buffer = match allocator.create_buffer(width, height, format.code, &[format.modifier]) {
                Ok(buffer) => buffer,
                Err(err) => {
                    errors.push(format!(
                        "{format:?} at {width}x{height}: GBM allocation failed: {}",
                        describe_io_error(&err)
                    ));
                    continue;
                }
            };
            let mut dmabuf = match buffer.export() {
                Ok(dmabuf) => dmabuf,
                Err(err) => {
                    errors.push(format!(
                        "{format:?} at {width}x{height}: GBM dmabuf export failed: {err}"
                    ));
                    continue;
                }
            };

            tracing::info!(
                ?format,
                width,
                height,
                gbm_metadata = %describe_gbm_buffer_metadata(&buffer),
                dmabuf_metadata = %describe_dmabuf_metadata(&dmabuf),
                "trying GBM dmabuf Vulkan render-target probe candidate"
            );

            let mut framebuffer = match <VulkanRenderer as Bind<Dmabuf>>::bind(renderer, &mut dmabuf) {
                Ok(framebuffer) => framebuffer,
                Err(err) => {
                    errors.push(format!(
                        "{format:?} at {width}x{height}: Vulkan Bind<Dmabuf> failed: {err}; {}",
                        describe_dmabuf_metadata(&dmabuf)
                    ));
                    continue;
                }
            };

            let output_size = Size::<i32, Physical>::from((width as i32, height as i32));
            let damage = [Rectangle::from_size(output_size)];
            let render_result = (|| {
                let mut frame = renderer.render(&mut framebuffer, output_size, Transform::Normal)?;
                frame.clear(Color32F::new(0.1, 0.2, 0.4, 1.0), &damage)?;
                frame.finish()
            })();

            match render_result {
                Ok(sync) => {
                    wait_sync_point(&sync, "GBM dmabuf Vulkan render-target sync")?;
                    tracing::info!(
                        ?format,
                        width,
                        height,
                        gbm_metadata = %describe_gbm_buffer_metadata(&buffer),
                        dmabuf_metadata = %describe_dmabuf_metadata(&dmabuf),
                        "GBM dmabuf Vulkan render-target probe rendered and released without DRM commit"
                    );
                    return Ok(());
                }
                Err(err) => {
                    let release_result =
                        <VulkanRenderer as RenderTargetLifecycle<Dmabuf>>::release_after_render_error(
                            renderer,
                            &mut framebuffer,
                        );
                    if let Err(release_err) = release_result {
                        return Err(format!(
                            "{format:?} at {width}x{height}: Vulkan render/finish failed: {err}; release_after_render_error failed: {release_err}; {}",
                            describe_dmabuf_metadata(&dmabuf)
                        )
                        .into());
                    }
                    errors.push(format!(
                        "{format:?} at {width}x{height}: Vulkan render/finish failed: {err}; release_after_render_error=Ok(()); {}",
                        describe_dmabuf_metadata(&dmabuf)
                    ));
                }
            }
        }
    }

    Err(format!(
        "no GBM-exported dmabuf candidate rendered through Vulkan Bind<Dmabuf>; attempted {} candidates: {}",
        errors.len(),
        errors.join("; ")
    )
    .into())
}

fn wait_sync_point(
    sync: &smithay::backend::renderer::sync::SyncPoint,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    for _ in 0..1024 {
        if sync.wait().is_ok() {
            return Ok(());
        }
        std::thread::yield_now();
    }

    Err(format!("{label}: sync wait was repeatedly interrupted").into())
}

struct PageflipProbeEventState {
    target_crtc: crtc::Handle,
    seen: bool,
    metadata: Option<DrmEventMetadata>,
    errors: Vec<String>,
}

fn pageflip_event_loop(
    drm_notifier: DrmDeviceNotifier,
    crtc: crtc::Handle,
) -> Result<
    (
        calloop::EventLoop<'static, PageflipProbeEventState>,
        PageflipProbeEventState,
    ),
    Box<dyn Error>,
> {
    let event_loop = calloop::EventLoop::<PageflipProbeEventState>::try_new()?;
    event_loop
        .handle()
        .insert_source(drm_notifier, |event, metadata, state| match event {
            DrmEvent::VBlank(event_crtc) if event_crtc == state.target_crtc => {
                state.seen = true;
                state.metadata = *metadata;
            }
            DrmEvent::VBlank(_) => {}
            DrmEvent::Error(error) => state.errors.push(format!("{error:?}")),
        })?;

    Ok((
        event_loop,
        PageflipProbeEventState {
            target_crtc: crtc,
            seen: false,
            metadata: None,
            errors: Vec::new(),
        },
    ))
}

fn wait_for_pageflip_event(
    event_loop: &mut calloop::EventLoop<PageflipProbeEventState>,
    state: &mut PageflipProbeEventState,
) -> Result<Option<DrmEventMetadata>, Box<dyn Error>> {
    state.seen = false;
    state.metadata = None;
    state.errors.clear();

    let deadline = Instant::now() + Duration::from_secs(3);
    while !state.seen {
        event_loop.dispatch(Duration::from_millis(100), state)?;
        if !state.errors.is_empty() {
            return Err(format!("DRM event processing failed: {}", state.errors.join("; ")).into());
        }
        if Instant::now() >= deadline {
            return Err("timed out waiting for DRM pageflip event".into());
        }
    }

    Ok(state.metadata)
}

fn probe_vulkan_allocator_framebuffer_metadata(
    drm: &mut DrmDevice,
    crtc: crtc::Handle,
    gbm: &GbmDevice<DrmDeviceFd>,
    physical_device: PhysicalDevice,
    renderer_formats: FormatSet,
) -> Result<(), Box<dyn Error>> {
    let pause_drm_on_return = PauseDrmOnReturn(drm);
    let usage = ImageUsageFlags::COLOR_ATTACHMENT;
    let mut allocator = VulkanAllocator::new(&physical_device, usage)?;
    let planes = pause_drm_on_return.0.planes(&crtc)?;
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
    let mut gbm_allocator = GbmAllocator::new(gbm.clone(), GbmBufferFlags::SCANOUT);
    for format in formats {
        for (width, height) in [(64, 64), (256, 256)] {
            let image = match allocator.create_buffer(width, height, format.code, &[format.modifier]) {
                Ok(image) => image,
                Err(err) => {
                    errors.push(format!(
                        "{format:?} at {width}x{height}: create buffer failed: {err}"
                    ));
                    continue;
                }
            };
            let dmabuf = match image.export() {
                Ok(dmabuf) => dmabuf,
                Err(err) => {
                    errors.push(format!(
                        "{format:?} at {width}x{height}: export dmabuf failed: {err}"
                    ));
                    continue;
                }
            };
            if dmabuf.num_planes() != 1 {
                errors.push(format!(
                    "{format:?} at {width}x{height}: metadata probe currently requires a single-plane Vulkan dmabuf; {}",
                    describe_dmabuf_metadata(&dmabuf)
                ));
                continue;
            }

            tracing::info!(
                ?format,
                width,
                height,
                image = ?image,
                metadata = %describe_dmabuf_metadata(&dmabuf),
                "trying Vulkan allocator dmabuf metadata framebuffer probe candidate"
            );

            match smithay::backend::drm::gbm::framebuffer_from_dmabuf(
                pause_drm_on_return.0.device_fd(),
                gbm,
                &dmabuf,
                false,
                false,
            ) {
                Ok(framebuffer) => {
                    drop(framebuffer);

                    tracing::info!(
                        ?format,
                        width,
                        height,
                        metadata = %describe_dmabuf_metadata(&dmabuf),
                        "Vulkan allocator metadata probe imported dmabuf and created/destroyed DRM framebuffer"
                    );
                    return Ok(());
                }
                Err(err) => {
                    let gbm_baseline_result = probe_gbm_allocator_addfb2(
                        &mut gbm_allocator,
                        pause_drm_on_return.0.device_fd(),
                        width,
                        height,
                        format.code,
                        format.modifier,
                    )
                    .map_err(|err| format!("GBM allocator baseline cleanup failed: {err}"))?;
                    let direct_addfb2_result =
                        probe_direct_prime_addfb2(pause_drm_on_return.0.device_fd(), &dmabuf)
                            .map_err(|err| format!("direct PRIME AddFB2 cleanup failed: {err}"))?;
                    let imported_bo_addfb2_result =
                        probe_gbm_imported_bo_addfb2(gbm, pause_drm_on_return.0.device_fd(), &dmabuf)
                            .map_err(|err| format!("GBM-imported BO AddFB2 cleanup failed: {err}"))?;
                    errors.push(format!(
                        "{format:?} at {width}x{height}: framebuffer import failed: {}; GBM allocator baseline: {}; GBM-imported BO AddFB2 probe: {}; direct PRIME AddFB2 probe: {}; {}; image={image:?}; {err:?}",
                        describe_drm_gbm_error(&err),
                        gbm_baseline_result,
                        imported_bo_addfb2_result,
                        direct_addfb2_result,
                        describe_dmabuf_metadata(&dmabuf)
                    ));
                }
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

fn probe_gbm_allocator_addfb2(
    gbm_allocator: &mut GbmAllocator<DrmDeviceFd>,
    drm: &DrmDeviceFd,
    width: u32,
    height: u32,
    fourcc: Fourcc,
    modifier: Modifier,
) -> Result<String, String> {
    let buffer = match gbm_allocator.create_buffer(width, height, fourcc, &[modifier]) {
        Ok(buffer) => buffer,
        Err(err) => return Ok(format!("GBM allocation failed: {}", describe_io_error(&err))),
    };
    let actual_format = AllocatorBuffer::format(&buffer);
    let flags = if actual_format.modifier != Modifier::Invalid {
        FbCmd2Flags::MODIFIERS
    } else {
        FbCmd2Flags::empty()
    };

    match drm.add_planar_framebuffer(&buffer, flags) {
        Ok(framebuffer) => match drm.destroy_framebuffer(framebuffer) {
            Ok(()) => Ok(format!(
                "GBM AddFB2 succeeded and framebuffer was destroyed; {}; exported dmabuf direct PRIME AddFB2: {}",
                describe_gbm_buffer_metadata(&buffer),
                probe_gbm_exported_dmabuf_direct_prime_addfb2(drm, &buffer)?
            )),
            Err(err) => Err(format!(
                "GBM AddFB2 succeeded but framebuffer destroy failed: {}; {}",
                describe_io_error(&err),
                describe_gbm_buffer_metadata(&buffer)
            )),
        },
        Err(err) => Ok(format!(
            "GBM AddFB2 failed: {}; {}; exported dmabuf direct PRIME AddFB2: {}",
            describe_io_error(&err),
            describe_gbm_buffer_metadata(&buffer),
            probe_gbm_exported_dmabuf_direct_prime_addfb2(drm, &buffer)?
        )),
    }
}

fn probe_gbm_exported_dmabuf_direct_prime_addfb2(
    drm: &DrmDeviceFd,
    buffer: &GbmBuffer,
) -> Result<String, String> {
    match buffer.export() {
        Ok(dmabuf) => Ok(format!(
            "{}; {}",
            probe_direct_prime_addfb2(drm, &dmabuf)?,
            describe_dmabuf_metadata(&dmabuf)
        )),
        Err(err) => Ok(format!("GBM dmabuf export failed: {err}")),
    }
}

fn probe_gbm_imported_bo_addfb2(
    gbm: &GbmDevice<DrmDeviceFd>,
    drm: &DrmDeviceFd,
    dmabuf: &Dmabuf,
) -> Result<String, String> {
    let buffer = match dmabuf.import_to(gbm, GbmBufferFlags::SCANOUT) {
        Ok(buffer) => buffer,
        Err(err) => return Ok(format!("GBM import failed: {}", describe_io_error(&err))),
    };
    let modifier = dmabuf.format().modifier;
    let flags = if modifier != Modifier::Invalid {
        FbCmd2Flags::MODIFIERS
    } else {
        FbCmd2Flags::empty()
    };
    let buffer_with_modifier = ImportedBoWithDmabufModifier {
        buffer: &buffer,
        modifier,
    };

    match drm.add_planar_framebuffer(&buffer_with_modifier, flags) {
        Ok(framebuffer) => match drm.destroy_framebuffer(framebuffer) {
            Ok(()) => Ok(format!(
                "GBM-imported BO AddFB2 succeeded and framebuffer was destroyed; {}",
                describe_imported_bo_with_dmabuf_modifier(&buffer_with_modifier)
            )),
            Err(err) => Err(format!(
                "GBM-imported BO AddFB2 succeeded but framebuffer destroy failed: {}; {}",
                describe_io_error(&err),
                describe_imported_bo_with_dmabuf_modifier(&buffer_with_modifier)
            )),
        },
        Err(err) => Ok(format!(
            "GBM-imported BO AddFB2 failed: {}; {}",
            describe_io_error(&err),
            describe_imported_bo_with_dmabuf_modifier(&buffer_with_modifier)
        )),
    }
}

struct ImportedBoWithDmabufModifier<'buffer> {
    buffer: &'buffer GbmBuffer,
    modifier: Modifier,
}

impl buffer::PlanarBuffer for ImportedBoWithDmabufModifier<'_> {
    fn size(&self) -> (u32, u32) {
        buffer::PlanarBuffer::size(self.buffer)
    }

    fn format(&self) -> Fourcc {
        buffer::PlanarBuffer::format(self.buffer)
    }

    fn modifier(&self) -> Option<Modifier> {
        match self.modifier {
            Modifier::Invalid => None,
            modifier => Some(modifier),
        }
    }

    fn pitches(&self) -> [u32; 4] {
        buffer::PlanarBuffer::pitches(self.buffer)
    }

    fn handles(&self) -> [Option<buffer::Handle>; 4] {
        buffer::PlanarBuffer::handles(self.buffer)
    }

    fn offsets(&self) -> [u32; 4] {
        buffer::PlanarBuffer::offsets(self.buffer)
    }
}

fn describe_imported_bo_with_dmabuf_modifier(buffer: &ImportedBoWithDmabufModifier<'_>) -> String {
    let reported_format = AllocatorBuffer::format(buffer.buffer);
    let handles = buffer::PlanarBuffer::handles(buffer)
        .iter()
        .map(|handle| handle.map(u32::from))
        .collect::<Vec<_>>();
    let (width, height) = buffer::PlanarBuffer::size(buffer);
    format!(
        "GBM-imported BO size={}x{}, reported_format={:?}, reported_modifier={:?}, AddFB2_modifier={:?}, handles={:?}, pitches={:?}, offsets={:?}",
        width,
        height,
        reported_format.code,
        reported_format.modifier,
        buffer.modifier,
        handles,
        buffer::PlanarBuffer::pitches(buffer),
        buffer::PlanarBuffer::offsets(buffer)
    )
}

fn probe_direct_prime_addfb2(drm: &DrmDeviceFd, dmabuf: &Dmabuf) -> Result<String, String> {
    let direct_buffer = match DirectDmabufBuffer::new(drm, dmabuf) {
        Ok(buffer) => buffer,
        Err(err) => return Ok(format!("PRIME fd import failed: {err}")),
    };
    let flags = if direct_buffer.modifier != Modifier::Invalid {
        FbCmd2Flags::MODIFIERS
    } else {
        FbCmd2Flags::empty()
    };

    match drm.add_planar_framebuffer(&direct_buffer, flags) {
        Ok(framebuffer) => match drm.destroy_framebuffer(framebuffer) {
            Ok(()) => Ok(format!(
                "direct AddFB2 succeeded and framebuffer was destroyed; {}",
                describe_direct_dmabuf_buffer(&direct_buffer)
            )),
            Err(err) => Err(format!(
                "direct AddFB2 succeeded but framebuffer destroy failed: {}; {}",
                describe_io_error(&err),
                describe_direct_dmabuf_buffer(&direct_buffer)
            )),
        },
        Err(err) => Ok(format!(
            "direct AddFB2 failed: {}; {}",
            describe_io_error(&err),
            describe_direct_dmabuf_buffer(&direct_buffer)
        )),
    }
}

fn describe_direct_dmabuf_buffer(buffer: &DirectDmabufBuffer<'_>) -> String {
    let handles = buffer
        .handles
        .iter()
        .map(|handle| handle.map(u32::from))
        .collect::<Vec<_>>();

    format!(
        "direct GEM buffer size={}x{}, format={:?}, modifier={:?}, handles={:?}, pitches={:?}, offsets={:?}",
        buffer.size.0, buffer.size.1, buffer.format, buffer.modifier, handles, buffer.pitches, buffer.offsets
    )
}

struct DirectDmabufBuffer<'drm> {
    drm: &'drm DrmDeviceFd,
    size: (u32, u32),
    format: Fourcc,
    modifier: Modifier,
    handles: [Option<buffer::Handle>; 4],
    pitches: [u32; 4],
    offsets: [u32; 4],
}

impl<'drm> DirectDmabufBuffer<'drm> {
    fn new(drm: &'drm DrmDeviceFd, dmabuf: &Dmabuf) -> Result<Self, std::io::Error> {
        let mut handles = [None; 4];
        for (index, fd) in dmabuf.handles().take(4).enumerate() {
            match drm.prime_fd_to_buffer(fd) {
                Ok(handle) => handles[index] = Some(handle),
                Err(err) => {
                    close_imported_prime_handles(drm, &mut handles);
                    return Err(err);
                }
            }
        }

        let mut pitches = [0; 4];
        for (index, stride) in dmabuf.strides().take(4).enumerate() {
            pitches[index] = stride;
        }

        let mut offsets = [0; 4];
        for (index, offset) in dmabuf.offsets().take(4).enumerate() {
            offsets[index] = offset;
        }

        Ok(Self {
            drm,
            size: (dmabuf.width(), dmabuf.height()),
            format: dmabuf.format().code,
            modifier: dmabuf.format().modifier,
            handles,
            pitches,
            offsets,
        })
    }
}

impl Drop for DirectDmabufBuffer<'_> {
    fn drop(&mut self) {
        close_imported_prime_handles(self.drm, &mut self.handles);
    }
}

fn close_imported_prime_handles(drm: &DrmDeviceFd, handles: &mut [Option<buffer::Handle>; 4]) {
    let mut closed = [0u32; 4];
    let mut closed_count = 0;
    for handle in handles.iter_mut().filter_map(Option::take) {
        let raw = u32::from(handle);
        if closed[..closed_count].contains(&raw) {
            continue;
        }
        closed[closed_count] = raw;
        closed_count += 1;
        if let Err(err) = drm.close_buffer(handle) {
            tracing::warn!(?handle, ?err, "failed to close direct PRIME-imported GEM handle");
        }
    }
}

impl buffer::PlanarBuffer for DirectDmabufBuffer<'_> {
    fn size(&self) -> (u32, u32) {
        self.size
    }

    fn format(&self) -> Fourcc {
        self.format
    }

    fn modifier(&self) -> Option<Modifier> {
        match self.modifier {
            Modifier::Invalid => None,
            modifier => Some(modifier),
        }
    }

    fn pitches(&self) -> [u32; 4] {
        self.pitches
    }

    fn handles(&self) -> [Option<buffer::Handle>; 4] {
        self.handles
    }

    fn offsets(&self) -> [u32; 4] {
        self.offsets
    }
}

struct PauseDrmOnReturn<'drm>(&'drm mut DrmDevice);

impl Drop for PauseDrmOnReturn<'_> {
    fn drop(&mut self) {
        self.0.pause();
    }
}

fn describe_drm_gbm_error(err: &DrmGbmError) -> String {
    match err {
        DrmGbmError::Import(source) => format!("GBM import failed: {}", describe_io_error(source)),
        DrmGbmError::Drm(access) => format!("DRM AddFB2 failed: {}", describe_io_error(&access.source)),
    }
}

fn describe_io_error(source: &std::io::Error) -> String {
    format!(
        "kind={:?}, raw_os_error={:?}, message={source}",
        source.kind(),
        source.raw_os_error()
    )
}

fn describe_dmabuf_metadata(dmabuf: &Dmabuf) -> String {
    let fds = dmabuf.handles().map(|fd| fd.as_raw_fd()).collect::<Vec<_>>();
    let fdinfo = fds
        .iter()
        .map(|fd| describe_dmabuf_fdinfo(*fd))
        .collect::<Vec<_>>();

    format!(
        "size={}x{}, format={:?}, modifier={:?}, planes={}, fds={:?}, offsets={:?}, strides={:?}, fdinfo={:?}",
        dmabuf.width(),
        dmabuf.height(),
        dmabuf.format().code,
        dmabuf.format().modifier,
        dmabuf.num_planes(),
        fds,
        dmabuf.offsets().collect::<Vec<_>>(),
        dmabuf.strides().collect::<Vec<_>>(),
        fdinfo
    )
}

fn describe_dmabuf_fdinfo(fd: i32) -> String {
    let path = format!("/proc/self/fdinfo/{fd}");
    let Ok(contents) = std::fs::read_to_string(path) else {
        return "fdinfo=unavailable".to_owned();
    };

    contents
        .lines()
        .filter(|line| {
            line.starts_with("flags:")
                || line.starts_with("mnt_id:")
                || line.starts_with("ino:")
                || line.starts_with("size:")
                || line.starts_with("count:")
                || line.starts_with("exp_name:")
                || line.starts_with("name:")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn describe_gbm_buffer_metadata(buffer: &GbmBuffer) -> String {
    let format = AllocatorBuffer::format(buffer);
    format!(
        "size={}x{}, format={:?}, modifier={:?}",
        buffer.width(),
        buffer.height(),
        format.code,
        format.modifier
    )
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
