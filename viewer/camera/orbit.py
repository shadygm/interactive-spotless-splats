import numpy as np
from viewer.camera.base import Camera, compute_camera_basis
from viewer.camera.state import CameraState


class OrbitCamera(Camera):
    def __init__(self, width, height, fov=45.0, near=0.1, far=1000.0):
        super().__init__(width, height, fov, near, far)
        self.center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.radius = 5.0
        self.azimuth = 0.0
        self.elevation = 0.3
        self.roll = 0.0
        # Preserve original behavior: __init__ uses raw height, resize uses max(height, 1)
        self.aspect = width / float(height)

    def rotate(self, dx, dy):
        """Update azimuth/elevation from mouse delta (left drag)."""
        sensitivity = 0.005
        self.azimuth += dx * sensitivity
        self.elevation += dy * sensitivity
        self.elevation = np.clip(self.elevation, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)

    def pan(self, dx, dy):
        """Move look_at point in camera plane (right drag)."""
        sensitivity = 0.002 * self.radius
        # Compute camera basis vectors
        c2w = self.get_view_matrix()
        right = c2w[:3, 0]
        up = c2w[:3, 1]
        self.center -= right * dx * sensitivity
        self.center += up * dy * sensitivity

    def zoom(self, delta):
        """Change radius (scroll)."""
        sensitivity = 0.1
        self.radius *= 1.0 + delta * sensitivity
        self.radius = max(self.radius, 0.1)

    def get_view_matrix(self):
        """Return 4x4 camera-to-world (c2w) numpy float32 array."""
        # Spherical coordinates to Cartesian
        x = self.radius * np.cos(self.elevation) * np.sin(self.azimuth)
        y = self.radius * np.sin(self.elevation)
        z = self.radius * np.cos(self.elevation) * np.cos(self.azimuth)
        eye = self.center + np.array([x, y, z], dtype=np.float32)

        # Camera looks at center from eye
        forward = self.center - eye

        right, up, forward = compute_camera_basis(forward, self.roll)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward  # OpenGL: camera looks down negative Z
        c2w[:3, 3] = eye

        return c2w

    def get_position(self):
        """Return camera world position."""
        return self.get_view_matrix()[:3, 3]

    def reset(self):
        """Reset camera to default home position."""
        self.center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.radius = 20.0
        self.azimuth = 0.0
        self.elevation = 0.3
        self.roll = 0.0

    def fit_to_bounds(self, min_bound, max_bound):
        """Set orbit center to scene center and radius based on scene size."""
        self.center = ((min_bound + max_bound) / 2.0).astype(np.float32)
        scene_size = float(np.linalg.norm(max_bound - min_bound))
        # Set radius so the entire scene fits in view (approximate)
        self.radius = max(scene_size * 1.5, 1.0)
        # Keep existing angles (default 0.0, 0.3) - facing same direction

    def resize(self, width, height):
        """Update aspect ratio."""
        super().resize(width, height)

    def to_state(self):
        return CameraState(
            position=self.get_position(),
            azimuth=self.azimuth,
            elevation=self.elevation,
            radius=self.radius,
            center=self.center.copy(),
            roll=self.roll,
            fov=self.fov,
            source_mode="orbit",
        )

    def from_state(self, state):
        if state.center is not None:
            self.center = state.center.copy()
        self.radius = state.radius
        self.azimuth = state.azimuth
        self.elevation = state.elevation
        self.roll = state.roll
        self.fov = state.fov
