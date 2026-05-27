import numpy as np

from viewer.camera.orbit import OrbitCamera
from viewer.camera.fps import FPSCamera
from viewer.camera.state import CameraState


def create_camera(mode: str, state: CameraState, width: int, height: int):
    """Create a camera of the given mode, initialized from a CameraState.

    Handles cross-mode conversion (e.g., orbit -> fps or fps -> orbit)
    by mapping equivalent parameters.
    """
    if mode == "orbit":
        cam = OrbitCamera(width, height, fov=state.fov)
        if state.source_mode == "fps":
            # Convert FPS yaw/pitch to orbit azimuth/elevation
            cam.azimuth = -state.yaw - np.pi / 2
            cam.elevation = state.pitch
            # Compute forward direction from azimuth/elevation to place center in front
            forward = np.array([
                np.cos(cam.elevation) * np.sin(cam.azimuth),
                np.sin(cam.elevation),
                np.cos(cam.elevation) * np.cos(cam.azimuth),
            ], dtype=np.float32)
            cam.center = state.position + forward * 5.0
            cam.radius = 5.0
        else:
            # Same-mode or default: restore state directly
            cam.from_state(state)
        return cam
    elif mode == "fps":
        cam = FPSCamera(width, height, fov=state.fov)
        if state.source_mode == "orbit":
            # Convert orbit azimuth/elevation to FPS yaw/pitch
            cam.position = state.position.copy()
            cam.yaw = -state.azimuth - np.pi / 2
            cam.pitch = state.elevation
            cam.move_speed = state.move_speed
        else:
            # Same-mode or default: restore state directly
            cam.from_state(state)
        return cam
    else:
        raise ValueError(f"Unknown camera mode: {mode}")
