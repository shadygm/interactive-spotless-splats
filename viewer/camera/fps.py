import numpy as np
from viewer.camera.base import Camera
from viewer.camera.state import CameraState


class FPSCamera(Camera):
    """First-person camera with yaw/pitch (no gimbal lock via direct basis)."""

    def __init__(self, width, height, fov=45.0, near=0.1, far=1000.0):
        super().__init__(width, height, fov, near, far)
        self.position = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        self.yaw = -np.pi / 2  # facing -Z
        self.pitch = 0.0
        self.move_speed = 5.0
        # Preserve original behavior: __init__ uses raw height, resize uses max(height, 1)
        self.aspect = width / float(height)

    def _get_basis(self):
        """Return (right, up, forward) using direct formulas — no cross-product singularities."""
        cp = np.cos(self.pitch)
        sp = np.sin(self.pitch)
        cy = np.cos(self.yaw)
        sy = np.sin(self.yaw)

        forward = np.array([cp * cy, sp, cp * sy], dtype=np.float32)
        right = np.array([sy, 0.0, -cy], dtype=np.float32)
        up = np.array([-sp * cy, cp, -sp * sy], dtype=np.float32)

        return right, up, forward

    def look(self, dx, dy):
        """Update yaw/pitch from mouse delta."""
        sensitivity = 0.002
        self.yaw -= dx * sensitivity
        self.pitch -= dy * sensitivity
        self.pitch = np.clip(self.pitch, -np.pi / 2 + 0.1, np.pi / 2 - 0.1)

    def move(self, forward, right, up, dt, speed_boost=1.0):
        """Move camera relative to its orientation."""
        speed = self.move_speed * dt * speed_boost
        direction = self.get_direction()
        right_vec = self.get_right()
        up_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        self.position += direction * forward * speed
        self.position += right_vec * right * speed
        self.position += up_vec * up * speed

    def get_direction(self):
        """Return normalized look direction."""
        _, _, forward = self._get_basis()
        return forward

    def get_right(self):
        """Return normalized right vector."""
        right, _, _ = self._get_basis()
        return right

    def get_view_matrix(self):
        """Return 4x4 camera-to-world (c2w) numpy float32 array."""
        right, up, forward = self._get_basis()

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward  # OpenGL: camera looks down negative Z
        c2w[:3, 3] = self.position
        return c2w

    def get_position(self):
        """Return camera world position."""
        return self.position.copy()

    def reset(self):
        """Reset camera to default home position."""
        self.position = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        self.yaw = -np.pi / 2
        self.pitch = 0.0

    def fit_to_bounds(self, min_bound, max_bound):
        """Set camera position based on scene bounds."""
        center = ((min_bound + max_bound) / 2.0).astype(np.float32)
        scene_size = float(np.linalg.norm(max_bound - min_bound))
        self.position = center + np.array([0.0, 0.0, scene_size * 1.5], dtype=np.float32)
        self.yaw = -np.pi / 2
        self.pitch = 0.0

    def resize(self, width, height):
        """Update aspect ratio."""
        super().resize(width, height)

    def to_state(self):
        return CameraState(
            position=self.position.copy(),
            yaw=self.yaw,
            pitch=self.pitch,
            fov=self.fov,
            move_speed=self.move_speed,
            source_mode="fps",
        )

    def from_state(self, state):
        if state.position is not None:
            self.position = state.position.copy()
        self.yaw = state.yaw
        self.pitch = state.pitch
        self.fov = state.fov
        self.move_speed = state.move_speed
