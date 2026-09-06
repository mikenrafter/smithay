//! Vulkan-window Anvil backend.
//!
//! This is the `--winit-vulkan` present path. GLES nested compositing stays in [`crate::winit`].

use std::{
    sync::{Mutex, atomic::Ordering},
    time::Duration,
};

#[cfg(feature = "debug")]
use smithay::backend::{allocator::Fourcc, renderer::ImportMem};

use smithay::{
    backend::{
        SwapBuffersError,
        allocator::dmabuf::Dmabuf,
        renderer::{
            ImportDma, ImportMemWl,
            damage::{Error as OutputDamageTrackerError, OutputDamageTracker},
            element::AsRenderElements,
            vulkan::VulkanRenderer,
        },
        winit::{self, WinitEvent, WinitVulkanGraphicsBackend},
    },
    input::{
        keyboard::LedState,
        pointer::{CursorImageAttributes, CursorImageStatus},
    },
    output::{Mode, Output, PhysicalProperties, Subpixel},
    reexports::{
        calloop::EventLoop,
        wayland_protocols::wp::presentation_time::server::wp_presentation_feedback,
        wayland_server::{Display, protocol::wl_surface},
        winit::event_loop::pump_events::PumpStatus,
    },
    utils::{IsAlive, Scale, Transform},
    wayland::{
        compositor,
        dmabuf::{
            DmabufFeedback, DmabufFeedbackBuilder, DmabufGlobal, DmabufHandler, DmabufState, ImportNotifier,
        },
        presentation::Refresh,
    },
};
use tracing::{error, info, warn};

use crate::state::{AnvilState, Backend, take_presentation_feedback, update_primary_scanout_output};
use crate::{drawing::*, render::*};

pub const OUTPUT_NAME: &str = "winit";

pub struct WinitVulkanData {
    backend: WinitVulkanGraphicsBackend,
    damage_tracker: OutputDamageTracker,
    dmabuf_state: (DmabufState, DmabufGlobal, Option<DmabufFeedback>),
    full_redraw: u8,
    #[cfg(feature = "debug")]
    pub fps: fps_ticker::Fps,
}

impl DmabufHandler for AnvilState<WinitVulkanData> {
    fn dmabuf_state(&mut self) -> &mut DmabufState {
        &mut self.backend_data.dmabuf_state.0
    }

    fn dmabuf_imported(&mut self, _global: &DmabufGlobal, dmabuf: Dmabuf, notifier: ImportNotifier) {
        if self
            .backend_data
            .backend
            .renderer()
            .import_dmabuf(&dmabuf, None)
            .is_ok()
        {
            let _ = notifier.successful::<AnvilState<WinitVulkanData>>();
        } else {
            notifier.failed();
        }
    }
}

impl Backend for WinitVulkanData {
    fn seat_name(&self) -> String {
        String::from("winit")
    }
    fn reset_buffers(&mut self, _output: &Output) {
        self.full_redraw = 4;
    }
    fn early_import(&mut self, _surface: &wl_surface::WlSurface) {}
    fn retire_surface_tree_textures(&mut self, surface: &wl_surface::WlSurface) {
        if let Err(err) = smithay::backend::renderer::utils::retire_and_release_surface_tree_textures(
            self.backend.renderer(),
            surface,
        ) {
            warn!("Failed to release retired surface textures: {}", err);
        }
    }
    fn update_led_state(&mut self, _led_state: LedState) {}
}

pub fn run_winit_vulkan() {
    let mut event_loop = EventLoop::try_new().unwrap();
    let display = Display::new().unwrap();
    let mut display_handle = display.handle();

    let (mut backend, mut winit) = match winit::init_vulkan() {
        Ok(ret) => ret,
        Err(err) => {
            error!("Failed to initialize Vulkan Winit backend: {}", err);
            return;
        }
    };
    let size = backend.window_size();

    let mode = Mode {
        size,
        refresh: 60_000,
    };
    let output = Output::new(
        OUTPUT_NAME.to_string(),
        PhysicalProperties {
            size: (0, 0).into(),
            subpixel: Subpixel::Unknown,
            make: "Smithay".into(),
            model: "Winit".into(),
            serial_number: "Unknown".into(),
        },
    );
    let _global = output.create_global::<AnvilState<WinitVulkanData>>(&display.handle());
    output.change_current_state(Some(mode), Some(Transform::Flipped180), None, Some((0, 0).into()));
    output.set_preferred(mode);

    #[cfg(feature = "debug")]
    #[allow(deprecated)]
    let fps_image =
        image::io::Reader::with_format(std::io::Cursor::new(FPS_NUMBERS_PNG), image::ImageFormat::Png)
            .decode()
            .unwrap();
    #[cfg(feature = "debug")]
    let fps_texture = backend
        .renderer()
        .import_memory(
            &fps_image.to_rgba8(),
            Fourcc::Abgr8888,
            (fps_image.width() as i32, fps_image.height() as i32).into(),
            false,
        )
        .expect("Unable to upload FPS texture");
    #[cfg(feature = "debug")]
    let mut fps_element = FpsElement::new(fps_texture);

    let dmabuf_formats = backend.renderer().dmabuf_formats();
    let main_device = backend
        .renderer()
        .physical_device()
        .and_then(|device| {
            device
                .render_node()
                .ok()
                .flatten()
                .or_else(|| device.primary_node().ok().flatten())
        })
        .map(|node| node.dev_id())
        .unwrap_or(0);
    let dmabuf_default_feedback = DmabufFeedbackBuilder::new(main_device, dmabuf_formats.clone())
        .build()
        .ok();
    let dmabuf_state = if let Some(default_feedback) = dmabuf_default_feedback {
        let mut dmabuf_state = DmabufState::new();
        let dmabuf_global = dmabuf_state.create_global_with_default_feedback::<AnvilState<WinitVulkanData>>(
            &display.handle(),
            &default_feedback,
        );
        (dmabuf_state, dmabuf_global, Some(default_feedback))
    } else {
        let mut dmabuf_state = DmabufState::new();
        let dmabuf_global =
            dmabuf_state.create_global::<AnvilState<WinitVulkanData>>(&display.handle(), dmabuf_formats);
        (dmabuf_state, dmabuf_global, None)
    };

    let data = {
        let damage_tracker = OutputDamageTracker::from_output(&output);

        WinitVulkanData {
            backend,
            damage_tracker,
            dmabuf_state,
            full_redraw: 0,
            #[cfg(feature = "debug")]
            fps: fps_ticker::Fps::default(),
        }
    };
    let mut state = AnvilState::init(display, event_loop.handle(), data, true);
    state
        .shm_state
        .update_formats(state.backend_data.backend.renderer().shm_formats());
    state.space.map_output(&output, (0, 0));

    #[cfg(feature = "xwayland")]
    state.start_xwayland();

    info!("Initialization completed, starting the Vulkan winit main loop.");

    let mut pointer_element = PointerElement::default();

    while state.running.load(Ordering::SeqCst) {
        let status = winit.dispatch_new_events(|event| match event {
            WinitEvent::Resized { size, .. } => {
                let output = state.space.outputs().next().unwrap().clone();
                state.space.map_output(&output, (0, 0));
                let mode = Mode {
                    size,
                    refresh: 60_000,
                };
                output.change_current_state(Some(mode), None, None, None);
                output.set_preferred(mode);
                crate::shell::fixup_positions(&mut state.space, state.pointer.current_location());
            }
            WinitEvent::Input(event) => state.process_input_event_windowed(event, OUTPUT_NAME),
            _ => (),
        });

        if let PumpStatus::Exit(_) = status {
            state.running.store(false, Ordering::SeqCst);
            break;
        }

        {
            let now = state.clock.now();
            let frame_target = now
                + output
                    .current_mode()
                    .map(|mode| Duration::from_secs_f64(1_000f64 / mode.refresh as f64))
                    .unwrap_or_default();
            state.pre_repaint(&output, frame_target);

            let backend = &mut state.backend_data.backend;

            let mut reset = false;
            if let CursorImageStatus::Surface(ref surface) = state.cursor_status {
                reset = !surface.alive();
            }
            if reset {
                state.cursor_status = CursorImageStatus::default_named();
            }
            let cursor_visible = !matches!(state.cursor_status, CursorImageStatus::Surface(_));

            pointer_element.set_status(state.cursor_status.clone());

            #[cfg(feature = "debug")]
            let fps = state.backend_data.fps.avg().round() as u32;
            #[cfg(feature = "debug")]
            fps_element.update_fps(fps);

            let full_redraw = &mut state.backend_data.full_redraw;
            *full_redraw = full_redraw.saturating_sub(1);
            let space = &mut state.space;
            let damage_tracker = &mut state.backend_data.damage_tracker;
            let show_window_preview = state.show_window_preview;

            let dnd_icon = state.dnd_icon.as_ref();

            let scale = Scale::from(output.current_scale().fractional_scale());
            let cursor_hotspot = if let CursorImageStatus::Surface(ref surface) = state.cursor_status {
                compositor::with_states(surface, |states| {
                    states
                        .data_map
                        .get::<Mutex<CursorImageAttributes>>()
                        .unwrap()
                        .lock()
                        .unwrap()
                        .hotspot
                })
            } else {
                (0, 0).into()
            };
            let cursor_pos = state.pointer.current_location();

            let age = if *full_redraw > 0 {
                0
            } else {
                backend.buffer_age().unwrap_or(0)
            };
            let render_res = backend.bind().and_then(|(renderer, mut fb)| {
                let mut elements = Vec::<CustomRenderElements<VulkanRenderer>>::new();

                elements.extend(
                    pointer_element.render_elements(
                        renderer,
                        (cursor_pos - cursor_hotspot.to_f64())
                            .to_physical(scale)
                            .to_i32_round(),
                        scale,
                        1.0,
                    ),
                );

                if let Some(icon) = dnd_icon {
                    let dnd_icon_pos = (cursor_pos + icon.offset.to_f64())
                        .to_physical(scale)
                        .to_i32_round();
                    if icon.surface.alive() {
                        elements.extend(AsRenderElements::<VulkanRenderer>::render_elements(
                            &smithay::desktop::space::SurfaceTree::from_surface(&icon.surface),
                            renderer,
                            dnd_icon_pos,
                            scale,
                            1.0,
                        ));
                    }
                }

                #[cfg(feature = "debug")]
                elements.push(CustomRenderElements::Fps(fps_element.clone()));

                render_output(
                    &output,
                    space,
                    elements,
                    renderer,
                    &mut fb,
                    damage_tracker,
                    age,
                    show_window_preview,
                )
                .map_err(|err| match err {
                    OutputDamageTrackerError::Rendering(err) => err.into(),
                    _ => unreachable!(),
                })
            });

            match render_res {
                Ok(render_output_result) => {
                    let has_rendered = render_output_result.damage.is_some();
                    if let Some(damage) = render_output_result.damage {
                        if let Err(err) = backend.submit(Some(damage)) {
                            warn!("Failed to submit buffer: {}", err);
                        }
                    }

                    backend.window().set_cursor_visible(cursor_visible);

                    let states = render_output_result.states;

                    update_primary_scanout_output(
                        &state.space,
                        &output,
                        &state.dnd_icon,
                        &state.cursor_status,
                        &states,
                    );

                    if has_rendered {
                        let mut output_presentation_feedback =
                            take_presentation_feedback(&output, &state.space, &states);
                        output_presentation_feedback.presented(
                            frame_target,
                            output
                                .current_mode()
                                .map(|mode| {
                                    Refresh::fixed(Duration::from_secs_f64(1_000f64 / mode.refresh as f64))
                                })
                                .unwrap_or(Refresh::Unknown),
                            0,
                            wp_presentation_feedback::Kind::Vsync,
                        )
                    }

                    state.post_repaint(&output, frame_target, None, &states);
                }
                Err(SwapBuffersError::ContextLost(err)) => {
                    error!("Critical Rendering Error: {}", err);
                    state.running.store(false, Ordering::SeqCst);
                }
                Err(err) => warn!("Rendering error: {}", err),
            }
        }

        let result = event_loop.dispatch(Some(Duration::from_millis(1)), &mut state);
        if result.is_err() {
            state.running.store(false, Ordering::SeqCst);
        } else {
            state.space.refresh();
            state.popups.cleanup();
            display_handle.flush_clients().unwrap();
        }

        #[cfg(feature = "debug")]
        state.backend_data.fps.tick();
    }
}
