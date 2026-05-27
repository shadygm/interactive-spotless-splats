from abc import ABC, abstractmethod
import numpy as np


class Camera(ABC):
    def __init__(self, width, height, fov=45.0, near=0.1, far=1000.0):
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far
        self.aspect = width / float(max(height, 1))

    def get_projection_matrix(self):
        """Return 4x4 perspective projection matrix."""
        f = 1.0 / np.tan(np.radians(self.fov) / 2.0)
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / self.aspect
        proj[1, 1] = f
        proj[2, 2] = (self.far + self.near) / (self.near - self.far)
        proj[2, 3] = (2.0 * self.far * self.near) / (self.near - self.far)
        proj[3, 2] = -1.0
        return proj

    def get_K(self):
        """Return 3x3 intrinsics matrix matching the projection."""
        f = self.width / (2.0 * np.tan(np.radians(self.fov) / 2.0))
        K = np.array([
            [f, 0.0, self.width / 2.0],
            [0.0, f, self.height / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        return K

    def resize(self, width, height):
        """Update aspect ratio."""
        self.width = width
        self.height = height
        self.aspect = width / float(max(height, 1))

    @abstractmethod
    def get_view_matrix(self) -> np.ndarray:
        """Return 4x4 camera-to-world (c2w) numpy float32 array."""
        ...

    @abstractmethod
    def get_position(self) -> np.ndarray:
        ...

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def fit_to_bounds(self, min_bound, max_bound):
        ...


def compute_camera_basis(forward: np.ndarray, roll: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute right, up, forward basis vectors from a forward direction.

    Handles gimbal lock by falling back to Z-up when forward is parallel to Y-up.
    Applies roll rotation around the forward axis if roll != 0.

    Returns (right, up, forward) as normalized float32 arrays.
    """
    forward = np.asarray(forward, dtype=np.float32)
    forward /= np.linalg.norm(forward) + 1e-8

    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(world_up, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(world_up, forward)
        right_norm = np.linalg.norm(right)
    right /= right_norm + 1e-8

    up = np.cross(forward, right)
    up /= np.linalg.norm(up) + 1e-8

    if abs(roll) > 1e-6:
        cr = np.cos(roll)
        sr = np.sin(roll)
        right, up = right * cr + up * sr, -right * sr + up * cr

    return right, up, forward
