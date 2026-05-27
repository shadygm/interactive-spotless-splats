import numpy as np
from loguru import logger

# When using a pure python backend, prefer to import glfw before imgui_bundle
# (so that you end up using the standard glfw, not the one provided by imgui_bundle)
import glfw

from imgui_bundle import imgui


class InputHandler:
    def __init__(self, window, camera, render_settings):
        self.window = window
        self.camera = camera
        self.render_settings = render_settings
        self._fps_cursor_captured = False

    def on_key(self, window, key, scancode, action, mods):
        """GLFW key callback for one-shot key events."""
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            if self.render_settings.camera_mode == "fps" and self._fps_cursor_captured:
                # In FPS mode, first ESC press releases the cursor
                glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)
                self._fps_cursor_captured = False
                logger.debug("FPS cursor released")
            else:
                glfw.set_window_should_close(self.window, True)

    def process(self, dt):
        """Process all input for this frame."""
        io = imgui.get_io()
        actions = {}

        if not io.want_capture_mouse:
            # Mouse input
            if self.render_settings.camera_mode == "fps":
                if self._fps_cursor_captured:
                    # FPS: mouse movement controls look when cursor is captured
                    self.camera.look(io.mouse_delta.x, io.mouse_delta.y)
                elif imgui.is_mouse_clicked(imgui.MouseButton_.left):
                    # Click in viewport to re-capture cursor
                    glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
                    self._fps_cursor_captured = True
                    logger.debug("FPS cursor captured")
            else:
                # Orbit: drag to rotate/pan
                if imgui.is_mouse_down(imgui.MouseButton_.left):
                    self.camera.rotate(io.mouse_delta.x, io.mouse_delta.y)
                if imgui.is_mouse_down(imgui.MouseButton_.right):
                    self.camera.pan(io.mouse_delta.x, io.mouse_delta.y)

            # Scroll
            if io.mouse_wheel != 0.0:
                if self.render_settings.camera_mode == "fps" and self._fps_cursor_captured:
                    # In FPS mode, scroll adjusts move speed
                    self.camera.move_speed *= 1.0 + io.mouse_wheel * 0.1
                    self.camera.move_speed = max(self.camera.move_speed, 0.1)
                elif self.render_settings.camera_mode != "fps":
                    if glfw.get_key(self.window, glfw.KEY_R) == glfw.PRESS:
                        # R + scroll = roll camera (tilt head left/right)
                        self.camera.roll += io.mouse_wheel * 0.05
                    else:
                        self.camera.zoom(io.mouse_wheel)

        # Keyboard camera controls when ImGui doesn't want keyboard
        if not io.want_capture_keyboard:
            c2w = self.camera.get_view_matrix()
            right = c2w[:3, 0]
            up = c2w[:3, 1]
            forward = c2w[:3, 2]

            if self.render_settings.camera_mode == "fps":
                # FPS: WASD moves camera position
                speed_boost = 5.0 if (glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(self.window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS) else 1.0

                move_forward = 0
                move_right = 0
                move_up = 0

                if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
                    move_forward += 1
                if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
                    move_forward -= 1
                if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
                    move_right -= 1
                if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
                    move_right += 1
                if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
                    move_up += 1
                if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
                    move_up -= 1

                self.camera.move(move_forward, move_right, move_up, dt, speed_boost)
            else:
                # Orbit: WASD moves center
                speed = 0.5 * self.camera.radius * dt
                if glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(self.window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS:
                    speed *= 5.0  # speed boost

                if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
                    self.camera.center -= forward * speed
                if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
                    self.camera.center += forward * speed
                if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
                    self.camera.center -= right * speed
                if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
                    self.camera.center += right * speed
                if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
                    self.camera.center += up * speed
                if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
                    self.camera.center -= up * speed
                if glfw.get_key(self.window, glfw.KEY_UP) == glfw.PRESS:
                    self.camera.elevation += 1.0 * dt
                if glfw.get_key(self.window, glfw.KEY_DOWN) == glfw.PRESS:
                    self.camera.elevation -= 1.0 * dt
                if glfw.get_key(self.window, glfw.KEY_LEFT) == glfw.PRESS:
                    self.camera.azimuth -= 1.0 * dt
                if glfw.get_key(self.window, glfw.KEY_RIGHT) == glfw.PRESS:
                    self.camera.azimuth += 1.0 * dt
                self.camera.elevation = np.clip(self.camera.elevation, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)

        return actions

    def is_fps_cursor_captured(self):
        return self._fps_cursor_captured

    def capture_fps_cursor(self):
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        self._fps_cursor_captured = True
        logger.debug("FPS cursor captured")

    def release_fps_cursor(self):
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)
        self._fps_cursor_captured = False
        logger.debug("FPS cursor released")
