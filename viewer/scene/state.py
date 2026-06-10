import os
import threading

import numpy as np
import torch
from loguru import logger

from viewer.camera.state import CameraState
from viewer.scene.bounds import compute_scene_bounds
from viewer.scene.frustums import build_camera_frustums
from viewer.scene.coordinate_system import qvec2rotmat
from viewer.scene.point_cloud import extract_point_cloud
from viewer.scene.loaders import ColmapLoader, PlyLoader, TransformsJsonLoader
from viewer.scene.gaussian_data import build_axis_gaussians


class SceneState:
    def __init__(self):
        self._lock = threading.RLock()
        self.colmap_cameras = {}
        self.colmap_images = {}
        self.colmap_points3D = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gaussians = build_axis_gaussians(self._device)
        self._default_gaussians = True
        self.has_colmap = False
        self.has_gaussians = True
        self._version = 0

    def _bump_version(self):
        with self._lock:
            self._version += 1

    def get_scene_version(self):
        with self._lock:
            return self._version

    def load_dataset(self, path):
        """Parse a COLMAP or transforms.json dataset directory."""
        with self._lock:
            try:
                # Try COLMAP first
                result = ColmapLoader().load(path)
                self.colmap_cameras = result["cameras"]
                self.colmap_images = result["images"]
                self.colmap_points3D = result["points3D"]
                self.has_colmap = bool(
                    self.colmap_cameras or self.colmap_images or self.colmap_points3D
                )
                self._last_colmap_path = path
                if self._default_gaussians:
                    self.gaussians = None
                    self.has_gaussians = False
                    self._default_gaussians = False
                self._bump_version()
            except Exception as colmap_err:
                # Fallback to transforms.json if present
                import os
                if os.path.exists(os.path.join(path, "transforms.json")):
                    try:
                        result = TransformsJsonLoader().load(path)
                        self.colmap_cameras = result["cameras"]
                        self.colmap_images = result["images"]
                        self.colmap_points3D = result["points3D"]
                        self.has_colmap = bool(
                            self.colmap_cameras or self.colmap_images or self.colmap_points3D
                        )
                        self._last_colmap_path = path
                        if self._default_gaussians:
                            self.gaussians = None
                            self.has_gaussians = False
                            self._default_gaussians = False
                        self._bump_version()
                        logger.info(f"Loaded transforms.json dataset from {path}")
                    except Exception as e:
                        logger.error(f"Failed to load transforms.json from {path}: {e}")
                        self.has_colmap = False
                else:
                    logger.error(f"Failed to load COLMAP from {path}: {colmap_err}")
                    self.has_colmap = False

    def load_colmap(self, path):
        """Backward-compatible alias for load_dataset."""
        self.load_dataset(path)

    def load_ply(self, path):
        """Parse a 3DGS PLY file and merge with axis Gaussians."""
        with self._lock:
            try:
                result = PlyLoader().load(path)
                ply_gaussians = result["gaussians"]
                if ply_gaussians is not None:
                    self.gaussians = ply_gaussians
                    self.has_gaussians = True
                    self._default_gaussians = False
                    self._bump_version()

                    # Debug: log gaussian bounds and stats
                    means_np = self.gaussians["means"].cpu().numpy()
                    N = means_np.shape[0]
                    scales_np = self.gaussians["scales"].cpu().numpy()
                    opacities_np = self.gaussians["opacities"].cpu().numpy()
                    colors_np = self.gaussians["colors"].cpu().numpy()
                    logger.info(f"Device: {self._device}")
                    logger.info(f"Gaussians count: {N}")
                    logger.info(f"Means bounds: min={means_np.min(axis=0)}, max={means_np.max(axis=0)}")
                    logger.info(f"Means mean: {means_np.mean(axis=0)}")
                    logger.info(f"Scales min/max/mean: {scales_np.min():.4f}/{scales_np.max():.4f}/{scales_np.mean():.4f}")
                    logger.info(f"Opacities min/max/mean: {opacities_np.min():.4f}/{opacities_np.max():.4f}/{opacities_np.mean():.4f}")
                    logger.info(f"Colors min/max/mean: {colors_np.min():.4f}/{colors_np.max():.4f}/{colors_np.mean():.4f}")
                    logger.info(f"SH degree: {self.gaussians['sh_degree']}")
                else:
                    self.has_gaussians = True  # axis gaussians still present
                    self._default_gaussians = True
            except Exception as e:
                logger.error(f"Failed to load PLY from {path}: {e}")
                self.has_gaussians = True  # axis gaussians still present
                self._default_gaussians = True

    def snapshot_gaussians(self):
        """Return a shallow copy of gaussian tensors under lock (for render thread)."""
        with self._lock:
            if self.gaussians is None:
                return None
            return {
                "means": self.gaussians["means"],
                "quats": self.gaussians["quats"],
                "scales": self.gaussians["scales"],
                "opacities": self.gaussians["opacities"],
                "colors": self.gaussians["colors"],
                "sh_degree": self.gaussians["sh_degree"],
            }

    def clear_gaussians(self):
        """Remove all gaussians from the scene.

        The renderer already treats a None gaussian buffer as empty, so this is a
        safe way to clear the learned splat state after stopping training.
        """
        with self._lock:
            self.gaussians = None
            self.has_gaussians = False
            self._default_gaussians = False
            self._bump_version()

    def has_learned_gaussians(self):
        """Return True when a non-default splat set is present.

        This is used to suppress the initial point-cloud overlay once the scene
        has a real splat model, either loaded from disk or produced by training.
        """
        with self._lock:
            return self.gaussians is not None and not self._default_gaussians

    def get_scene_bounds(self):
        """Return axis-aligned bounding box (min, max) of camera positions and points."""
        with self._lock:
            return compute_scene_bounds(self.colmap_images, self.colmap_points3D)

    def get_camera_frustums(self, render_settings=None):
        """Return list of (position, corners) for each COLMAP image."""
        with self._lock:
            if not self.has_colmap or not self.colmap_images or not self.colmap_cameras:
                return []
            bmin, bmax = compute_scene_bounds(self.colmap_images, self.colmap_points3D)
            scene_scale = float(np.linalg.norm(bmax - bmin))
            return build_camera_frustums(
                self.colmap_images, self.colmap_cameras, render_settings, scene_scale
            )

    def get_ordered_camera_ids(self) -> list[int]:
        with self._lock:
            return sorted(self.colmap_images.keys()) if self.colmap_images else []

    def get_camera_count(self) -> int:
        with self._lock:
            return len(self.colmap_images) if self.colmap_images else 0

    def get_camera_state_by_index(self, index: int) -> CameraState | None:
        """Return the Nth dataset camera as a viewport camera state."""
        with self._lock:
            if not self.has_colmap or not self.colmap_images or not self.colmap_cameras:
                return None

            image_ids = sorted(self.colmap_images.keys())
            if not image_ids:
                return None

            index = int(np.clip(index, 0, len(image_ids) - 1))
            image = self.colmap_images[image_ids[index]]
            camera = self.colmap_cameras.get(image["camera_id"])
            if camera is None:
                return None

            R = qvec2rotmat(image["qvec"])
            tvec = np.asarray(image["tvec"], dtype=np.float64)
            position = (-R.T @ tvec).astype(np.float32)
            forward = (-R.T[:, 2]).astype(np.float32)
            forward /= np.linalg.norm(forward) + 1e-8

            yaw = float(np.arctan2(forward[2], forward[0]))
            pitch = float(np.arcsin(np.clip(forward[1], -1.0, 1.0)))

            params = np.asarray(camera.get("params", []), dtype=np.float64)
            width = float(camera.get("width", 1))
            fov = 45.0
            if params.size >= 1:
                if camera.get("model") == 1 and params.size >= 2:
                    focal = 0.5 * (float(params[0]) + float(params[1]))
                else:
                    focal = float(params[0])
                if focal > 1e-8:
                    fov = float(np.degrees(2.0 * np.arctan(width / (2.0 * focal))))

            return CameraState(
                position=position,
                yaw=yaw,
                pitch=pitch,
                fov=fov,
                source_mode="fps",
            )

    def get_camera_pose_by_index(self, index: int) -> dict | None:
        """Return the exact dataset pose and intrinsics for the Nth camera."""
        with self._lock:
            if not self.has_colmap or not self.colmap_images or not self.colmap_cameras:
                return None

            image_ids = sorted(self.colmap_images.keys())
            if not image_ids:
                return None

            index = int(np.clip(index, 0, len(image_ids) - 1))
            image = self.colmap_images[image_ids[index]]
            camera = self.colmap_cameras.get(image["camera_id"])
            if camera is None:
                return None

            R = qvec2rotmat(image["qvec"])
            tvec = np.asarray(image["tvec"], dtype=np.float64)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = R.T.astype(np.float32)
            c2w[:3, 3] = (-R.T @ tvec).astype(np.float32)

            params = np.asarray(camera.get("params", []), dtype=np.float64)
            width = int(camera.get("width", 1))
            height = int(camera.get("height", 1))

            if camera.get("model") == 1 and params.size >= 4:
                fx, fy, cx, cy = map(float, params[:4])
            elif camera.get("model") == 2 and params.size >= 4:
                f, cx, cy = map(float, params[:3])
                fx = fy = f
            else:
                fx = fy = float(params[0]) if params.size >= 1 else float(width)
                cx = float(params[1]) if params.size >= 2 else width / 2.0
                cy = float(params[2]) if params.size >= 3 else height / 2.0

            K = np.array(
                [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

            return {
                "c2w": c2w,
                "K": K,
                "width": width,
                "height": height,
                "name": image.get("name", ""),
                "camera_id": image.get("camera_id", -1),
            }

    def get_point_cloud(self):
        """Return (xyz, rgb) numpy arrays from colmap_points3D."""
        with self._lock:
            if self.has_learned_gaussians():
                return None, None
            if not self.has_colmap or not self.colmap_points3D:
                return None, None
            return extract_point_cloud(self.colmap_points3D)
