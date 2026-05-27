import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram


class ShaderProgram:
    """Wraps compileShader/compileProgram and caches uniform locations."""

    def __init__(self, vertex_src, fragment_src, validate=False):
        vs = compileShader(vertex_src, GL_VERTEX_SHADER)
        fs = compileShader(fragment_src, GL_FRAGMENT_SHADER)
        self.program = compileProgram(vs, fs, validate=validate)
        self._uniforms = {}

    def use(self):
        glUseProgram(self.program)

    def loc(self, name):
        if name not in self._uniforms:
            self._uniforms[name] = glGetUniformLocation(self.program, name)
        return self._uniforms[name]

    def set_mat4(self, name, value, transpose=GL_TRUE):
        glUniformMatrix4fv(self.loc(name), 1, transpose, value)

    def set_float(self, name, value):
        glUniform1f(self.loc(name), value)

    def set_int(self, name, value):
        glUniform1i(self.loc(name), value)

    def set_vec3(self, name, x, y, z):
        glUniform3f(self.loc(name), x, y, z)


class VBO:
    """Wraps glGenBuffers/glBindBuffer/glBufferData/glEnableVertexAttribArray/glVertexAttribPointer."""

    def __init__(self, data=None, usage=GL_STATIC_DRAW):
        self.vbo = glGenBuffers(1)
        if data is not None:
            self.upload(data, usage)

    def upload(self, data, usage=GL_STATIC_DRAW):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, usage)

    def bind(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

    def set_attrib(self, index, size, stride, offset, divisor=0):
        glEnableVertexAttribArray(index)
        glVertexAttribPointer(index, size, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        if divisor:
            glVertexAttribDivisor(index, divisor)


class Texture2D:
    """Wraps glGenTextures/glTexParameteri/glTexImage2D."""

    def __init__(self, width, height, internal_format=GL_RGBA8, format=GL_RGBA, type=GL_UNSIGNED_BYTE):
        self.texture = glGenTextures(1)
        self.width = width
        self.height = height
        self.internal_format = internal_format
        self.format = format
        self.type = type
        self.bind()
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, format, type, None)
        self.unbind()

    def bind(self, unit=0):
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_2D, self.texture)

    def unbind(self):
        glBindTexture(GL_TEXTURE_2D, 0)

    def resize(self, width, height):
        self.width = width
        self.height = height
        self.bind()
        glTexImage2D(GL_TEXTURE_2D, 0, self.internal_format, width, height, 0, self.format, self.type, None)
        self.unbind()

    def sub_image(self, width, height, data):
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, self.format, self.type, data)

    def image(self, width, height, data):
        glTexImage2D(GL_TEXTURE_2D, 0, self.internal_format, width, height, 0, self.format, self.type, data)
