import ctypes
import numpy as np
from OpenGL.GL import *
from loguru import logger

from viewer.rendering.gl_resources import ShaderProgram, VBO
from viewer.rendering.shaders import DEBUG_VERTEX, DEBUG_FRAGMENT, FRUSTUM_VERTEX, FRUSTUM_FRAGMENT

# Maximum points to draw for performance
_MAX_POINTS = 50000

# Frustum template: 8 lines = 16 vertices
# Each vertex stores an index 0..4 selecting center or one of 4 corners
_FRUSTUM_TEMPLATE = np.array([
    0, 1,  # center -> corner0
    0, 2,  # center -> corner1
    0, 3,  # center -> corner2
    0, 4,  # center -> corner3
    1, 2,  # corner0 -> corner1
    2, 3,  # corner1 -> corner2
    3, 4,  # corner2 -> corner3
    4, 1,  # corner3 -> corner0
], dtype=np.float32)


class DebugPass:
    """Owns frustums and points rendering."""

    def __init__(self, render_settings=None):
        self.render_settings = render_settings

        # Compile debug shader
        self._debug_program = ShaderProgram(DEBUG_VERTEX, DEBUG_FRAGMENT, validate=False)

        # Compile instanced frustum shader
        self._frustum_program = ShaderProgram(FRUSTUM_VERTEX, FRUSTUM_FRAGMENT, validate=False)

        # Frustum VAO + template VBO + instance VBO
        self._frustum_vao = glGenVertexArrays(1)
        self._frustum_template_vbo = VBO()
        self._frustum_instance_vbo = VBO()
        self._build_frustum_template_vbo()
        self._frustum_instance_count = 0

        # Point VAO + VBO
        self._point_vao = glGenVertexArrays(1)
        self._point_vbo = VBO()
        self._point_count = 0

        # Cached versions to detect changes
        self._cached_scene_version = -1
        self._cached_frustum_version = -1

    def _build_frustum_template_vbo(self):
        """Build static frustum template VBO (one frustum topology)."""
        glBindVertexArray(self._frustum_vao)
        self._frustum_template_vbo.upload(_FRUSTUM_TEMPLATE)
        self._frustum_template_vbo.set_attrib(0, 1, 0, 0)
        glBindVertexArray(0)
        logger.debug("Frustum template VBO built")

    def update_debug_cache(self, scene_state, rebuild_frustums=True, rebuild_points=True):
        """Rebuild specified VBOs."""
        if rebuild_frustums:
            frustums = scene_state.get_camera_frustums(self.render_settings)
            if frustums:
                instance_data = np.zeros((len(frustums), 15), dtype=np.float32)
                for i, (C, corners) in enumerate(frustums):
                    instance_data[i, 0:3] = C
                    instance_data[i, 3:6] = corners[0]
                    instance_data[i, 6:9] = corners[1]
                    instance_data[i, 9:12] = corners[2]
                    instance_data[i, 12:15] = corners[3]

                glBindVertexArray(self._frustum_vao)
                self._frustum_instance_vbo.upload(instance_data)

                self._frustum_instance_vbo.set_attrib(1, 3, 15 * 4, 0, divisor=1)
                self._frustum_instance_vbo.set_attrib(2, 3, 15 * 4, 3 * 4, divisor=1)
                self._frustum_instance_vbo.set_attrib(3, 3, 15 * 4, 6 * 4, divisor=1)
                self._frustum_instance_vbo.set_attrib(4, 3, 15 * 4, 9 * 4, divisor=1)
                self._frustum_instance_vbo.set_attrib(5, 3, 15 * 4, 12 * 4, divisor=1)

                glBindVertexArray(0)
                self._frustum_instance_count = len(frustums)
                logger.debug(f"Frustums rebuilt: {len(frustums)} cameras")
            else:
                self._frustum_instance_count = 0

        if rebuild_points:
            xyz, rgb = scene_state.get_point_cloud()
            if xyz is not None and len(xyz) > 0:
                n = len(xyz)
                stride = max(1, n // _MAX_POINTS)

                point_verts = []
                for i in range(0, n, stride):
                    point_verts.append(list(xyz[i]) + list(rgb[i]))
                point_data = np.array(point_verts, dtype=np.float32)

                glBindVertexArray(self._point_vao)
                self._point_vbo.upload(point_data)
                self._point_vbo.set_attrib(0, 3, 6 * 4, 0)
                self._point_vbo.set_attrib(1, 3, 6 * 4, 3 * 4)
                glBindVertexArray(0)

                self._point_count = len(point_verts)
                logger.debug(f"Points rebuilt: {self._point_count} vertices")
            else:
                self._point_count = 0

    def render(self, camera, scene_state, width, height):
        """Render debug overlays using cached VAOs."""
        import time
        t0 = time.perf_counter()

        # Check what needs rebuilding
        current_scene_version = scene_state.get_scene_version()
        current_frustum_version = self.render_settings.get_frustum_version() if self.render_settings else 0

        rebuild_frustums = current_frustum_version != self._cached_frustum_version or current_scene_version != self._cached_scene_version
        rebuild_points = current_scene_version != self._cached_scene_version

        if rebuild_frustums or rebuild_points:
            self.update_debug_cache(scene_state, rebuild_frustums, rebuild_points)
            self._cached_scene_version = current_scene_version
            self._cached_frustum_version = current_frustum_version
        t1 = time.perf_counter()

        # Compute MVP
        proj = camera.get_projection_matrix()
        c2w = camera.get_view_matrix()
        view = np.linalg.inv(c2w).astype(np.float32)
        mvp = (proj @ view).astype(np.float32)

        glDisable(GL_DEPTH_TEST)

        # Draw frustums (instanced)
        if self._frustum_instance_count > 0:
            self._frustum_program.use()
            self._frustum_program.set_mat4("mvp", mvp)
            color = self.render_settings.frustum_color if self.render_settings else [1.0, 1.0, 1.0]
            self._frustum_program.set_vec3("uColor", color[0], color[1], color[2])
            glBindVertexArray(self._frustum_vao)
            glDrawArraysInstanced(GL_LINES, 0, 16, self._frustum_instance_count)

        # Draw points
        if self._point_count > 0:
            glEnable(GL_PROGRAM_POINT_SIZE)
            self._debug_program.use()
            self._debug_program.set_mat4("mvp", mvp)
            point_size = self.render_settings.point_size if self.render_settings else 4.0
            self._debug_program.set_float("uPointSize", point_size)
            glBindVertexArray(self._point_vao)
            glDrawArrays(GL_POINTS, 0, self._point_count)
            glDisable(GL_PROGRAM_POINT_SIZE)

        glBindVertexArray(0)
        glUseProgram(0)
        t2 = time.perf_counter()

        logger.debug(
            f"debug breakdown (ms): cache_check={(t1-t0)*1000:.2f} draw={(t2-t1)*1000:.2f} total={(t2-t0)*1000:.2f}"
        )
