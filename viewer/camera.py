import importlib.util
import os
import sys

# viewer/camera.py shadows the viewer.camera package. To preserve backward
# compatibility, we bootstrap the package into sys.modules so that
# submodules can be imported and all public names remain available.
_camera_dir = os.path.join(os.path.dirname(__file__), 'camera')
_spec = importlib.util.spec_from_file_location('viewer.camera', os.path.join(_camera_dir, '__init__.py'))
_package = importlib.util.module_from_spec(_spec)
sys.modules['viewer.camera'] = _package
_spec.loader.exec_module(_package)

# Re-export everything from the package for backward compatibility
Camera = _package.Camera
OrbitCamera = _package.OrbitCamera
FPSCamera = _package.FPSCamera
CameraState = _package.CameraState
compute_camera_basis = _package.compute_camera_basis
create_camera = _package.create_camera
DatasetCamera = _package.DatasetCamera

__all__ = [
    "Camera",
    "OrbitCamera",
    "FPSCamera",
    "DatasetCamera",
    "CameraState",
    "compute_camera_basis",
    "create_camera",
]
