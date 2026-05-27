import ctypes
import numpy as np
from OpenGL.GL import *

from viewer.rendering.gl_resources import ShaderProgram, VBO
from viewer.rendering.shaders import QUAD_VERTEX, QUAD_FRAGMENT


class QuadPass:
    """Owns the fullscreen quad shader, VAO/VBO, and texture blit logic."""

    def __init__(self):
        self._program = ShaderProgram(QUAD_VERTEX, QUAD_FRAGMENT, validate=False)

        self._vao = glGenVertexArrays(1)
        self._vbo = VBO()
        # Flip V texture coordinates so OpenGL bottom-left origin matches
        # image top-left origin without a CPU np.flip per frame.
        quad_vertices = np.array([
            -1.0, -1.0, 0.0, 1.0,
             1.0, -1.0, 1.0, 1.0,
             1.0,  1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 0.0,
        ], dtype=np.float32)
        glBindVertexArray(self._vao)
        self._vbo.upload(quad_vertices)
        self._vbo.set_attrib(0, 2, 4 * 4, 0)
        self._vbo.set_attrib(1, 2, 4 * 4, 2 * 4)
        glBindVertexArray(0)

    def render_texture_to_screen(self, texture):
        """Draw a fullscreen quad with the given texture."""
        glDisable(GL_DEPTH_TEST)
        self._program.use()
        texture.bind(0)
        self._program.set_int("uTex", 0)
        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
