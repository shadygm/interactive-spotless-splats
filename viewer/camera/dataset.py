from __future__ import annotations

import numpy as np

from viewer.camera.base import Camera
from viewer.camera.state import CameraState


class DatasetCamera(Camera):
    """Locked camera that exactly reproduces a dataset frustum."""

    def __init__(self, width: int, height: int, c2w: np.ndarray, K: np.ndarray, fov: float = 45.0):
        super().__init__(width, height, fov=fov)
        self._c2w = np.asarray(c2w, dtype=np.float32)
        self._K = np.asarray(K, dtype=np.float32)
        self.center = self._c2w[:3, 3].copy()
        self.radius = 1.0
        self.azimuth = 0.0
        self.elevation = 0.0
        self.roll = 0.0
        self.position = self.center.copy()
        self.yaw = 0.0
        self.pitch = 0.0
        self.move_speed = 5.0

    def get_view_matrix(self) -> np.ndarray:
        return self._c2w.copy()

    def get_K(self) -> np.ndarray:
        return self._K.copy()

    def get_position(self) -> np.ndarray:
        return self._c2w[:3, 3].copy()

    def to_state(self) -> CameraState:
        """Export the locked dataset frustum as a generic camera state.

        The app may switch camera modes after a frustum jump; returning a
        regular CameraState keeps that path working while preserving the
        current exact pose as closely as possible.
        """
        position = self.get_position()
        forward = -self._c2w[:3, 2].copy()
        forward /= np.linalg.norm(forward) + 1e-8
        yaw = float(np.arctan2(forward[2], forward[0]))
        pitch = float(np.arcsin(np.clip(forward[1], -1.0, 1.0)))
        return CameraState(
            position=position,
            yaw=yaw,
            pitch=pitch,
            fov=self.fov,
            move_speed=self.move_speed,
            source_mode="fps",
        )

    def reset(self):
        return None

    def fit_to_bounds(self, min_bound, max_bound):
        return None

    def resize(self, width, height):
        super().resize(width, height)

    def rotate(self, dx, dy):
        return None

    def pan(self, dx, dy):
        return None

    def zoom(self, delta):
        return None

    def look(self, dx, dy):
        return None

    def move(self, forward, right, up, dt, speed_boost=1.0):
        return None
