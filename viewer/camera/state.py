from dataclasses import dataclass
import numpy as np


@dataclass
class CameraState:
    position: np.ndarray
    yaw: float = 0.0
    pitch: float = 0.0
    azimuth: float = 0.0
    elevation: float = 0.0
    radius: float = 5.0
    center: np.ndarray = None
    roll: float = 0.0
    fov: float = 45.0
    move_speed: float = 5.0
    source_mode: str = ""
