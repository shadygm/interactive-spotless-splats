import os
if os.getenv("XDG_SESSION_TYPE") == "wayland" and not os.getenv("PYOPENGL_PLATFORM"):
    os.environ["PYOPENGL_PLATFORM"] = "x11"

import sys
import time

import numpy as np
from loguru import logger

# When using a pure python backend, prefer to import glfw before imgui_bundle
# (so that you end up using the standard glfw, not the one provided by imgui_bundle)
import glfw

from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
import imgui_bundle.imgui.internal as internal

import OpenGL.GL as gl

# Workaround for PyOpenGL + GLFW on Wayland: PyOpenGL's GLX platform cannot
# detect the GLFW-created context via glXGetCurrentContext(), causing
# "Attempt to retrieve context when no valid context". We patch getContext
# to return a dummy object when the real context is not found.
# See: https://github.com/pthom/imgui_bundle/issues/321
import OpenGL.contextdata as _ctxdata
_orig_getContext = _ctxdata.getContext
class _DummyContext:
    pass
_dummy_ctx = _DummyContext()
def _patched_getContext(context=None):
    try:
        return _orig_getContext(context)
    except Exception:
        return _dummy_ctx
_ctxdata.getContext = _patched_getContext

from viewer.camera import OrbitCamera, FPSCamera, CameraState, create_camera
from viewer.scene import SceneState
from viewer.ui import UI, RenderSettings
from viewer.renderer import Renderer
from viewer.theme import apply_gruvbox_theme
from viewer.input import InputHandler


class App:
    def __init__(self, width=1280, height=720, colmap_path=None, ply_path=None):
        self.width = width
        self.height = height

        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.window = glfw.create_window(width, height, "Interactive Spotless Splats", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(0)  # Disable vsync to unlock frame rate

        # Remove any saved imgui.ini so docking layout is always fresh on startup
        import os as _os
        ini_path = _os.path.join(_os.getcwd(), "imgui.ini")
        if _os.path.exists(ini_path):
            _os.remove(ini_path)

        # ImGui setup with docking
        imgui.create_context()
        apply_gruvbox_theme()
        io = imgui.get_io()
        io.config_flags |= imgui.ConfigFlags_.docking_enable
        self.impl = GlfwRenderer(self.window)

        # Scene, renderer, UI, render settings
        self.camera = OrbitCamera(width, height)
        self.scene_state = SceneState()
        self.render_settings = RenderSettings()
        self.renderer = Renderer(width, height, device=self.scene_state._device, render_settings=self.render_settings)
        self.ui = UI(self.scene_state, self.renderer, self.render_settings, self.camera)
        self.input_handler = InputHandler(self.window, self.camera, self.render_settings)

        # Load initial data if provided
        if colmap_path:
            self.scene_state.load_colmap(colmap_path)
        if ply_path:
            self.scene_state.load_ply(ply_path)

        # Auto-fit camera to scene bounds so debug geometry is visible
        bmin, bmax = self.scene_state.get_scene_bounds()
        self.camera.fit_to_bounds(bmin, bmax)
        radius = getattr(self.camera, 'radius', 5.0)
        logger.info(f"Camera fitted to scene bounds: min={bmin}, max={bmax}, radius={radius:.2f}")

        # Build debug VBO caches for any pre-loaded data
        self.renderer.update_debug_cache(self.scene_state, True, True, True)
        self.renderer._cached_scene_version = self.scene_state.get_scene_version()
        self.renderer._cached_grid_version = self.render_settings.get_grid_version()
        self.renderer._cached_frustum_version = self.render_settings.get_frustum_version()

        # Callbacks
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        glfw.set_key_callback(self.window, self._on_key)

        # Docking setup state
        self._dockspace_setup_done = False

        # Camera mode tracking
        self._current_camera_mode = self.render_settings.camera_mode

        # Frame timing
        self._frame_count = 0
        self._last_time = glfw.get_time()

    def _on_resize(self, window, w, h):
        self.width = w
        self.height = h
        self.camera.resize(w, h)
        self.renderer.resize(w, h)
        gl.glViewport(0, 0, w, h)

    def _on_key(self, window, key, scancode, action, mods):
        self.input_handler.on_key(window, key, scancode, action, mods)

    def _setup_docking(self, dockspace_id):
        """One-time dockspace layout setup: split right, dock Scene panel."""
        if self._dockspace_setup_done:
            return

        # Always rebuild layout from scratch on first frame
        internal.dock_builder_remove_node(dockspace_id)
        internal.dock_builder_add_node(dockspace_id, internal.DockNodeFlagsPrivate_.dock_space)
        viewport = imgui.get_main_viewport()
        internal.dock_builder_set_node_size(dockspace_id, viewport.size)

        result = internal.dock_builder_split_node_py(dockspace_id, imgui.Dir_.right, 0.25)
        dock_id_right = result[0]
        internal.dock_builder_dock_window("Scene", dock_id_right)
        internal.dock_builder_dock_window("Render Settings", dock_id_right)
        internal.dock_builder_finish(dockspace_id)

        self._dockspace_setup_done = True
        logger.info("Docking layout initialized: panels docked on the right")

    def _switch_camera_mode(self, new_mode):
        """Switch between orbit and FPS camera, preserving view."""
        if new_mode == self._current_camera_mode:
            return

        old_cam = self.camera
        state = old_cam.to_state()
        width, height = old_cam.width, old_cam.height

        self.camera = create_camera(new_mode, state, width, height)

        if new_mode == "fps":
            self.input_handler.capture_fps_cursor()
        else:
            self.input_handler.release_fps_cursor()

        self.ui.camera = self.camera
        self.input_handler.camera = self.camera
        self._current_camera_mode = new_mode
        logger.info(f"Switched camera mode to {new_mode}")

    def run(self):
        while not glfw.window_should_close(self.window):
            current_time = glfw.get_time()
            dt = current_time - self._last_time
            self._last_time = current_time

            t0 = time.perf_counter()
            glfw.poll_events()
            self.impl.process_inputs()
            t1 = time.perf_counter()

            imgui.new_frame()
            t2 = time.perf_counter()

            # Full-viewport dockspace with passthru central node (OpenGL shows through)
            dockspace_flags = imgui.DockNodeFlags_.passthru_central_node
            dockspace_id = imgui.dock_space_over_viewport(
                dockspace_id=imgui.get_id("MainDockSpace"),
                viewport=imgui.get_main_viewport(),
                flags=dockspace_flags,
            )
            self._setup_docking(dockspace_id)

            # Scene panel (auto-docked to right split)
            imgui.begin("Scene")
            self.ui.draw_content()
            imgui.end()

            # Render Settings panel (auto-docked to right split as a tab)
            imgui.begin("Render Settings")
            self.ui.draw_render_settings()
            imgui.end()

            # Process all input
            self.input_handler.process(dt)

            # Check for pending scene loads from UI
            pending_colmap = self.ui.pending_colmap_path
            pending_ply = self.ui.pending_ply_path
            if pending_colmap is not None:
                self.scene_state.load_colmap(pending_colmap)
                self.renderer.update_debug_cache(self.scene_state, True, True, True)
            if pending_ply is not None:
                self.scene_state.load_ply(pending_ply)
            if pending_colmap is not None or pending_ply is not None:
                self.ui.clear_pending_actions()

            # React to camera mode changes from UI
            if self.render_settings.camera_mode != self._current_camera_mode:
                self._switch_camera_mode(self.render_settings.camera_mode)

            imgui.render()
            t3 = time.perf_counter()

            # OpenGL background: clear, gsplat, debug overlays
            gl.glViewport(0, 0, self.width, self.height)
            gl.glClearColor(0.0, 0.0, 0.0, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            gaussians = self.scene_state.snapshot_gaussians()
            gsplat_ok = self.renderer.render_gsplat(self.camera, gaussians, self.width, self.height)
            if gsplat_ok:
                self.renderer.render_texture_to_screen()
            else:
                self.renderer.render_debug(self.camera, self.scene_state, self.width, self.height)
            t4 = time.perf_counter()

            # ImGui draw on top
            self.impl.render(imgui.get_draw_data())
            t5 = time.perf_counter()

            glfw.swap_buffers(self.window)
            t6 = time.perf_counter()

            # Update FPS
            self.renderer.update_fps(dt)
            self._frame_count += 1


        self.impl.shutdown()
        glfw.destroy_window(self.window)
        glfw.terminate()
