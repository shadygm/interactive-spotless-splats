import time
from OpenGL.GL import *
from loguru import logger

from viewer.rendering.gl_resources import Texture2D
from viewer.rendering.quad_pass import QuadPass
from viewer.rendering.debug_pass import DebugPass
from viewer.rendering.gsplat_pass import GsplatPass


class Renderer:
    def __init__(self, width, height, device="cuda", render_settings=None):
        self.width = width
        self.height = height
        self.device = device
        self.render_settings = render_settings

        self._quad_pass = QuadPass()
        self._debug_pass = DebugPass(render_settings=render_settings)
        self._gsplat_pass = GsplatPass(device=device)

        # Create OpenGL texture for gsplat output
        self._texture = Texture2D(width, height)
        self._has_texture_content = False

        # FPS tracking
        self._frame_times = []
        self._last_fps_update = 0
        self.fps = 0.0

    # ------------------------------------------------------------------
    # Backward-compatible cache-version attribute proxies
    # ------------------------------------------------------------------
    @property
    def _cached_scene_version(self):
        return self._debug_pass._cached_scene_version

    @_cached_scene_version.setter
    def _cached_scene_version(self, value):
        self._debug_pass._cached_scene_version = value

    @property
    def _cached_frustum_version(self):
        return self._debug_pass._cached_frustum_version

    @_cached_frustum_version.setter
    def _cached_frustum_version(self, value):
        self._debug_pass._cached_frustum_version = value

    # ------------------------------------------------------------------
    # Pass delegation (backward-compatible public API)
    # ------------------------------------------------------------------
    def render_gsplat(self, camera, gaussians, width, height):
        """Render Gaussians via gsplat and upload to OpenGL texture."""
        ok = self._gsplat_pass.render(
            camera, gaussians, width, height,
            self.render_settings, self._texture,
        )
        self._has_texture_content = ok
        return ok

    def render_debug(self, camera, scene_state, width, height):
        """Render debug overlays using cached VAOs."""
        self._debug_pass.render(camera, scene_state, width, height)

    def render_texture_to_screen(self):
        """Draw a fullscreen quad with the gsplat texture."""
        if not self._has_texture_content:
            logger.warning("[VIEWPORT] No texture content, skipping quad draw")
            return
        self._quad_pass.render_texture_to_screen(self._texture)

    def update_debug_cache(self, scene_state, rebuild_frustums=True, rebuild_points=True):
        """Rebuild specified VBOs."""
        self._debug_pass.update_debug_cache(scene_state, rebuild_frustums, rebuild_points)

    def resize(self, width, height):
        """Update texture size."""
        self.width = width
        self.height = height
        self._texture.resize(width, height)
        self._has_texture_content = False

    def update_fps(self, dt):
        """Update FPS counter."""
        self._frame_times.append(dt)
        if len(self._frame_times) > 60:
            self._frame_times.pop(0)
        if len(self._frame_times) >= 10:
            avg_dt = sum(self._frame_times) / len(self._frame_times)
            self.fps = 1.0 / avg_dt if avg_dt > 0 else 0.0

    # ------------------------------------------------------------------
    # New orchestrator method
    # ------------------------------------------------------------------
    def render_frame(self, camera, scene_state, gaussians, width, height):
        """Render a complete frame: gsplat -> quad blit -> debug overlays."""
        ok = self.render_gsplat(camera, gaussians, width, height)
        if ok:
            self.render_texture_to_screen()
        self.render_debug(camera, scene_state, width, height)
