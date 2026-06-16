use std::{error::Error, ffi::OsString, process::Command, sync::Arc, time::Duration};

use smithay::{
    backend::{
        allocator::Fourcc,
        renderer::{
            Bind, Color32F, ExportMem, Frame, Offscreen, Renderer,
            element::{
                Kind,
                surface::{WaylandSurfaceRenderElement, render_elements_from_surface_tree},
            },
            utils::{draw_render_elements, on_commit_buffer_handler},
        },
        vulkan::{Instance, PhysicalDevice, version::Version},
    },
    reexports::{
        calloop::{EventLoop, Interest, Mode, PostAction, generic::Generic},
        wayland_server::{
            Client, Display, DisplayHandle,
            backend::{ClientData, ClientId, DisconnectReason},
            protocol::{
                wl_buffer,
                wl_surface::{self, WlSurface},
            },
        },
    },
    utils::{Physical, Rectangle, Serial, Size, Transform},
    wayland::{
        buffer::BufferHandler,
        compositor::{
            CompositorClientState, CompositorHandler, CompositorState, SurfaceAttributes, TraversalAction,
            with_surface_tree_downward,
        },
        shell::xdg::{PopupSurface, PositionerState, ToplevelSurface, XdgShellHandler, XdgShellState},
        shm::{ShmHandler, ShmState},
        socket::ListeningSocketSource,
    },
};
use wayland_protocols::xdg::shell::server::xdg_toplevel;

use smithay::backend::renderer::vulkan::{VulkanError, VulkanRenderTarget, VulkanRenderer};

const DEFAULT_WIDTH: i32 = 256;
const DEFAULT_HEIGHT: i32 = 256;
const DEFAULT_FRAMES: u32 = 120;
const CLEAR_PIXEL_ABGR8888: [u8; 4] = [255, 0, 0, 255];

struct App {
    display_handle: DisplayHandle,
    socket_name: OsString,
    compositor_state: CompositorState,
    xdg_shell_state: XdgShellState,
    shm_state: ShmState,
}

impl App {
    fn new(event_loop: &mut EventLoop<Self>, display: Display<Self>) -> Self {
        let display_handle = display.handle();
        let compositor_state = CompositorState::new::<Self>(&display_handle);
        let xdg_shell_state = XdgShellState::new::<Self>(&display_handle);
        let shm_state = ShmState::new::<Self>(&display_handle, vec![]);

        let socket = ListeningSocketSource::new_auto().expect("failed to create Wayland socket");
        let socket_name = socket.socket_name().to_os_string();

        event_loop
            .handle()
            .insert_source(socket, |client_stream, _, state| {
                state
                    .display_handle
                    .insert_client(client_stream, Arc::new(ClientState::default()))
                    .expect("failed to insert Wayland client");
            })
            .expect("failed to insert Wayland socket source");

        event_loop
            .handle()
            .insert_source(
                Generic::new(display, Interest::READ, Mode::Level),
                |_, display, state| {
                    // SAFETY: the display source owns the `Display` for the life of the event loop.
                    unsafe {
                        display.get_mut().dispatch_clients(state).unwrap();
                    }
                    Ok(PostAction::Continue)
                },
            )
            .expect("failed to insert Wayland display source");

        Self {
            display_handle,
            socket_name,
            compositor_state,
            xdg_shell_state,
            shm_state,
        }
    }
}

impl BufferHandler for App {
    fn buffer_destroyed(&mut self, _buffer: &wl_buffer::WlBuffer) {}
}

impl CompositorHandler for App {
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.compositor_state
    }

    fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState {
        &client.get_data::<ClientState>().unwrap().compositor_state
    }

    fn commit(&mut self, surface: &WlSurface) {
        on_commit_buffer_handler::<Self>(surface);
    }
}

impl ShmHandler for App {
    fn shm_state(&self) -> &ShmState {
        &self.shm_state
    }
}

impl XdgShellHandler for App {
    fn xdg_shell_state(&mut self) -> &mut XdgShellState {
        &mut self.xdg_shell_state
    }

    fn new_toplevel(&mut self, surface: ToplevelSurface) {
        surface.with_pending_state(|state| {
            state.states.set(xdg_toplevel::State::Activated);
        });
        surface.send_configure();
    }

    fn new_popup(&mut self, _surface: PopupSurface, _positioner: PositionerState) {}

    fn grab(
        &mut self,
        _surface: PopupSurface,
        _seat: smithay::reexports::wayland_server::protocol::wl_seat::WlSeat,
        _serial: Serial,
    ) {
    }

    fn reposition_request(&mut self, _surface: PopupSurface, _positioner: PositionerState, _token: u32) {}
}

#[derive(Default)]
struct ClientState {
    compositor_state: CompositorClientState,
}

impl ClientData for ClientState {
    fn initialized(&self, _client_id: ClientId) {}

    fn disconnected(&self, _client_id: ClientId, _reason: DisconnectReason) {}
}

fn main() -> Result<(), Box<dyn Error>> {
    if let Ok(env_filter) = tracing_subscriber::EnvFilter::try_from_default_env() {
        tracing_subscriber::fmt().with_env_filter(env_filter).init();
    } else {
        tracing_subscriber::fmt().init();
    }

    let mut event_loop = EventLoop::<App>::try_new()?;
    let display = Display::<App>::new()?;
    let mut app = App::new(&mut event_loop, display);
    let mut renderer = create_vulkan_renderer()?;
    let render_format =
        choose_render_format(&renderer).ok_or("Vulkan Abgr8888 offscreen rendering is unavailable")?;
    let output_size = Size::<i32, Physical>::from((DEFAULT_WIDTH, DEFAULT_HEIGHT));
    let mut target = Offscreen::<VulkanRenderTarget<'static>>::create_buffer(
        &mut renderer,
        render_format,
        (DEFAULT_WIDTH, DEFAULT_HEIGHT).into(),
    )?;

    tracing::info!(name = %app.socket_name.to_string_lossy(), "Listening on Wayland socket");
    let mut child = ChildGuard::new(spawn_client_if_requested(&app.socket_name)?);

    let full_damage = [Rectangle::from_size(output_size)];
    let mut imported_any_surface = false;
    for frame_idx in 0..DEFAULT_FRAMES {
        event_loop.dispatch(Duration::from_millis(16), &mut app)?;

        let elements =
            app.xdg_shell_state
                .toplevel_surfaces()
                .iter()
                .flat_map(|surface| {
                    render_elements_from_surface_tree::<
                        VulkanRenderer,
                        WaylandSurfaceRenderElement<VulkanRenderer>,
                    >(
                        &mut renderer,
                        surface.wl_surface(),
                        (0, 0),
                        1.0,
                        1.0,
                        Kind::Unspecified,
                    )
                })
                .collect::<Vec<_>>();

        imported_any_surface |= !elements.is_empty();

        {
            let mut framebuffer = Bind::bind(&mut renderer, &mut target)?;
            let mut frame = renderer.render(&mut framebuffer, output_size, Transform::Normal)?;
            frame.clear(Color32F::new(1.0, 0.0, 0.0, 1.0), &full_damage)?;
            draw_render_elements(&mut frame, 1.0, &elements, &full_damage)?;
            let sync = frame.finish()?;
            renderer.wait(&sync)?;

            let readback = renderer.copy_framebuffer(
                &framebuffer,
                Rectangle::from_size((DEFAULT_WIDTH, DEFAULT_HEIGHT).into()),
                render_format,
            )?;
            let bytes = renderer.map_texture(&readback)?;
            if imported_any_surface && frame_idx > 0 && has_non_clear_pixel(bytes) {
                tracing::info!("Vulkan wl_shm smoke imported and rendered a Wayland surface");
                child.terminate();
                return Ok(());
            }
        }

        send_frames(&app, frame_idx);
        app.display_handle.flush_clients()?;
    }

    child.terminate();
    if imported_any_surface {
        Err("Wayland surface imported but rendered output stayed clear".into())
    } else {
        Err("no Wayland wl_shm surface was imported; run with a client such as weston-simple-shm".into())
    }
}

fn create_vulkan_renderer() -> Result<VulkanRenderer, Box<dyn Error>> {
    let instance = Instance::new(Version::VERSION_1_3, None)?;
    let physical_device = PhysicalDevice::enumerate(&instance)?
        .next()
        .ok_or(VulkanError::VulkanUnavailable)?;
    Ok(VulkanRenderer::builder()
        .with_physical_device(physical_device)
        .build()?)
}

fn choose_render_format(renderer: &VulkanRenderer) -> Option<Fourcc> {
    let formats = <VulkanRenderer as Bind<VulkanRenderTarget<'static>>>::supported_formats(renderer)?;
    formats
        .iter()
        .any(|entry| entry.code == Fourcc::Abgr8888)
        .then_some(Fourcc::Abgr8888)
}

fn spawn_client_if_requested(socket_name: &OsString) -> Result<Option<std::process::Child>, Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        return Ok(None);
    }
    if args.first().is_some_and(|arg| arg == "--") {
        args.remove(0);
    }
    if args.is_empty() {
        return Ok(None);
    }

    let mut command = Command::new(&args[0]);
    command.args(&args[1..]);
    command.env("WAYLAND_DISPLAY", socket_name);
    Ok(Some(command.spawn()?))
}

struct ChildGuard(Option<std::process::Child>);

impl ChildGuard {
    fn new(child: Option<std::process::Child>) -> Self {
        Self(child)
    }

    fn terminate(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        self.terminate();
    }
}

fn has_non_clear_pixel(bytes: &[u8]) -> bool {
    bytes.chunks_exact(4).any(|pixel| pixel != CLEAR_PIXEL_ABGR8888)
}

fn send_frames(app: &App, time: u32) {
    for surface in app.xdg_shell_state.toplevel_surfaces() {
        send_frames_surface_tree(surface.wl_surface(), time);
    }
}

fn send_frames_surface_tree(surface: &wl_surface::WlSurface, time: u32) {
    with_surface_tree_downward(
        surface,
        (),
        |_, _, &()| TraversalAction::DoChildren(()),
        |_surface, states, &()| {
            for callback in states
                .cached_state
                .get::<SurfaceAttributes>()
                .current()
                .frame_callbacks
                .drain(..)
            {
                callback.done(time);
            }
        },
        |_, _, &()| true,
    );
}

smithay::delegate_dispatch2!(App);
