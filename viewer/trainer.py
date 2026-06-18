import sys
import os
import threading
import math
import time
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
from loguru import logger

from gsplat.rendering import rasterization
from gsplat.strategy import MCMCStrategy, DefaultStrategy
from viewer.scene.loaders import _load_point_cloud_records
from trainer.spotless.dataset import build_semantic_feature_manifest, load_semantic_features
from trainer.spotless.encoding import get_positional_encodings, get_positional_encodings_from_coords
from trainer.spotless.mask import robust_cluster_mask, robust_mask
from trainer.spotless.mlp import SpotLessModule
from trainer.spotless.stats import RunningStats


# ---------------------------------------------------------------------------
# Simple COLMAP parser using pycolmap 4.x
# ---------------------------------------------------------------------------


def _flip_c2w_to_gsplat(c2w: np.ndarray) -> np.ndarray:
    """Convert OpenGL-style c2w to the OpenCV/COLMAP convention used by gsplat."""
    c2w_gsplat = c2w.copy()
    c2w_gsplat[:3, 1] *= -1.0
    c2w_gsplat[:3, 2] *= -1.0
    return c2w_gsplat


def _similarity_from_cameras(c2w: np.ndarray, strict_scaling: bool = False, center_method: str = "focus") -> np.ndarray:
    """Match the reference gsplat normalization transform for camera poses."""
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # Estimate the world up axis from average camera up axes.
    ups = np.sum(R * np.array([0.0, -1.0, 0.0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = float((up_camspace * world_up).sum())
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1.0:
        R_align = np.eye(3) + skew + (skew @ skew) * 1.0 / (1.0 + c)
    else:
        R_align = np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    R = R_align @ R
    fwds = np.sum(R * np.array([0.0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    if center_method == "focus":
        nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
        translate = -np.median(nearest, axis=0)
    elif center_method == "poses":
        translate = -np.median(t, axis=0)
    else:
        raise ValueError(f"Unknown center_method {center_method}")

    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))
    transform[:3, :] *= scale
    return transform


def _align_principal_axes(point_cloud: np.ndarray) -> np.ndarray:
    """Return a PCA alignment transform matching the reference implementation."""
    centroid = np.median(point_cloud, axis=0)
    translated = point_cloud - centroid
    covariance_matrix = np.cov(translated, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sort_indices = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, sort_indices]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1

    rotation_matrix = eigenvectors.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = -rotation_matrix @ centroid
    return transform


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply an SE(3)/similarity transform to points."""
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _transform_cameras(matrix: np.ndarray, camtoworlds: np.ndarray) -> np.ndarray:
    """Apply an SE(3)/similarity transform to camera-to-world matrices."""
    camtoworlds = np.einsum("nij,ki->nkj", camtoworlds, matrix)
    scaling = np.linalg.norm(camtoworlds[:, 0, :3], axis=1)
    camtoworlds[:, :3, :3] = camtoworlds[:, :3, :3] / scaling[:, None, None]
    return camtoworlds


def _normalize_world_space(camtoworlds: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize cameras and points using the reference gsplat procedure."""
    t1 = _similarity_from_cameras(camtoworlds)
    camtoworlds = _transform_cameras(t1, camtoworlds)
    points = _transform_points(t1, points)

    t2 = _align_principal_axes(points)
    camtoworlds = _transform_cameras(t2, camtoworlds)
    points = _transform_points(t2, points)

    transform = t2 @ t1

    # Reference upside-down fix.
    if np.median(points[:, 2]) > np.mean(points[:, 2]):
        t3 = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        camtoworlds = _transform_cameras(t3, camtoworlds)
        points = _transform_points(t3, points)
        transform = t3 @ transform

    return camtoworlds, points, transform


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for quaternions in [w, x, y, z] format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a quaternion [w, x, y, z]."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float32)


def _discover_semantic_shape(
    image_paths: list[str],
    dataset_root: str | None = None,
    feature_kind: str = "raw",
) -> tuple[int, ...] | None:
    requested_kind = feature_kind
    if dataset_root is not None:
        images_dir = os.path.join(dataset_root, "images")
        if os.path.isdir(images_dir):
            suffix = "_sdfeats_clustered.npy" if requested_kind == "clustered" else "_sdfeats.npy"
            candidates = sorted(Path(images_dir).glob(f"*{suffix}"))
            if candidates:
                logger.info(
                    f"Spotless semantic discovery: probing {candidates[0]} "
                    f"({len(candidates)} candidates under {images_dir})"
                )
            for candidate in candidates:
                try:
                    features = np.load(candidate)
                    logger.info(
                        f"Spotless semantic discovery: loaded {candidate} "
                        f"with shape={tuple(features.shape)} dtype={features.dtype}"
                    )
                    return tuple(features.shape)
                except Exception as exc:
                    logger.warning(f"Spotless semantic discovery: failed to load {candidate}: {exc}")

    for image_path in image_paths:
        try:
            features, candidate, loaded_kind = load_semantic_features(
                image_path,
                dataset_root=dataset_root,
                feature_kind=requested_kind,
            )
            if features is not None:
                logger.info(
                    f"Spotless semantic discovery: loaded {candidate} "
                    f"for {image_path} (kind={loaded_kind}, shape={tuple(features.shape)})"
                )
                return tuple(features.shape)
        except Exception as exc:
                logger.warning(f"Spotless semantic discovery: failed to load features for {image_path}: {exc}")
    return None


def _unwrap_singleton_string(value: Any) -> Any:
    """Normalize DataLoader-collated string fields back to a scalar when uniform."""
    if isinstance(value, (list, tuple)) and value and all(item == value[0] for item in value):
        return value[0]
    return value


def _resize_image_if_needed(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target (width, height) if it differs from the current size."""
    target_w, target_h = target_size
    if image.shape[1] == target_w and image.shape[0] == target_h:
        return image
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

class _SimpleColmapParser:
    """Minimal COLMAP parser using pycolmap 4.x API."""

    def __init__(self, data_dir: str, factor: int = 1, test_every: int = 8):
        import pycolmap

        self.data_dir = os.path.abspath(data_dir)
        self.factor = factor
        self.test_every = test_every

        # Find sparse reconstruction directory
        sparse_dir = os.path.join(data_dir, "sparse", "0")
        if not os.path.exists(sparse_dir):
            sparse_dir = os.path.join(data_dir, "sparse")
        if not os.path.exists(sparse_dir):
            raise ValueError(f"COLMAP sparse directory not found in {data_dir}")

        # Read reconstruction
        rec = pycolmap.Reconstruction()
        rec.read(sparse_dir)

        if rec.num_images == 0:
            raise ValueError("No images found in COLMAP reconstruction.")

        # Collect camera data
        self.Ks_dict: Dict[int, np.ndarray] = {}
        self.imsize_dict: Dict[int, Tuple[int, int]] = {}
        for cam_id, cam in rec.cameras.items():
            params = np.array(cam.params, dtype=np.float64)
            model = cam.model_name
            if model == "PINHOLE":
                fx, fy, cx, cy = params
            elif model == "SIMPLE_PINHOLE":
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model == "SIMPLE_RADIAL":
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:
                # Default to first 4 params for other models
                fx, fy, cx, cy = params[:4]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            if factor > 1:
                K[:2, :] /= factor
            self.Ks_dict[cam_id] = K
            w = cam.width // factor
            h = cam.height // factor
            self.imsize_dict[cam_id] = (w, h)

        # Collect image data
        w2c_mats = []
        camera_ids = []
        image_names = []

        for img_id in sorted(rec.images.keys()):
            img = rec.images[img_id]
            if not img.has_pose:
                continue
            image_names.append(img.name)
            camera_ids.append(img.camera_id)

            # World-to-camera
            rigid = img.cam_from_world()
            R = np.array(rigid.rotation.matrix(), dtype=np.float64)
            t = np.array(rigid.translation, dtype=np.float64).reshape(3, 1)
            w2c = np.concatenate([np.concatenate([R, t], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
            w2c_mats.append(w2c)

        w2c_mats = np.stack(w2c_mats, axis=0)
        camtoworlds = np.linalg.inv(w2c_mats)

        # Sort by image name for consistent train/val splits
        inds = np.argsort(image_names)
        self.image_names = [image_names[i] for i in inds]
        self.camtoworlds = camtoworlds[inds].astype(np.float32)
        self.camera_ids = [camera_ids[i] for i in inds]

        # Create 0-based contiguous camera indices
        unique_camera_ids = sorted(set(self.camera_ids))
        self.camera_id_to_idx = {cid: idx for idx, cid in enumerate(unique_camera_ids)}
        self.camera_indices = [self.camera_id_to_idx[cid] for cid in self.camera_ids]
        self.num_cameras = len(unique_camera_ids)

        # Image paths: look for pre-resized folder first (e.g. images_4)
        # This matches the simple_trainer.py behavior.
        colmap_image_dir = os.path.join(data_dir, "images")
        if factor > 1:
            image_dir = os.path.join(data_dir, f"images_{factor}")
            if not os.path.exists(image_dir):
                image_dir = colmap_image_dir
        else:
            image_dir = colmap_image_dir
        if not os.path.exists(image_dir):
            raise ValueError(f"Image folder {image_dir} does not exist.")
        self.image_paths = [os.path.join(image_dir, name) for name in self.image_names]

        # Point cloud
        points_list = []
        colors_list = []
        for pt_id in rec.points3D:
            pt = rec.points3D[pt_id]
            points_list.append(pt.xyz)
            colors_list.append(pt.color)
        if len(points_list) == 0:
            logger.warning(
                f"No COLMAP point cloud was found in {sparse_dir}; creating 1000 fallback points in a 5x5x5 box"
            )
            rng = np.random.default_rng(42)
            self.points = rng.uniform(-2.5, 2.5, size=(1000, 3)).astype(np.float32)
            self.points_rgb = rng.integers(96, 256, size=(1000, 3), dtype=np.uint8)
        else:
            self.points = np.array(points_list, dtype=np.float32)
            self.points_rgb = np.array(colors_list, dtype=np.uint8)

        if len(self.camtoworlds) > 0 and len(self.points) > 0:
            self.camtoworlds, self.points, self.transform = _normalize_world_space(self.camtoworlds, self.points)
            self.transform_inv = np.linalg.inv(self.transform)
        else:
            self.transform = np.eye(4, dtype=np.float64)
            self.transform_inv = np.eye(4, dtype=np.float64)

        # Scene scale
        camera_locations = self.camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = float(np.max(dists))


class _SimpleColmapDataset(torch.utils.data.Dataset):
    """Minimal COLMAP dataset for training.

    Loads images fresh from disk every iteration (no caching).
    The DataLoader workers handle prefetching. This matches simple_trainer.py.
    """

    def __init__(
        self,
        parser: _SimpleColmapParser,
        split: str = "train",
        feature_kind: str = "raw",
        use_eval_split: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.feature_kind = feature_kind
        indices = np.arange(len(self.parser.image_names))
        if use_eval_split:
            if split == "train":
                self.indices = indices[indices % self.parser.test_every != 0]
            else:
                self.indices = indices[indices % self.parser.test_every == 0]
        else:
            self.indices = indices if split == "train" else indices[:0]
        self.semantic_feature_paths, self.semantic_preview_paths, self.semantic_shape = build_semantic_feature_manifest(
            self.parser.image_paths,
            dataset_root=self.parser.data_dir,
            feature_kind=self.feature_kind,
        )
        self.has_semantics = self.semantic_shape is not None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image_path = self.parser.image_paths[index]

        # Load image with imageio (fast, no PIL overhead).
        # If a pre-resized folder was found, this is already downsampled.
        image = imageio.imread(image_path)[..., :3]

        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy().astype(np.float32)
        camtoworld = self.parser.camtoworlds[index].copy()
        image = _resize_image_if_needed(image, self.parser.imsize_dict[camera_id])

        data = {
            "K": torch.from_numpy(K),
            "camtoworld": torch.from_numpy(camtoworld),
            "image": torch.from_numpy(image),
            "image_id": item,
            "camera_idx": self.parser.camera_indices[index],
            # Keep the collate path tensor/string-only; PyTorch default_collate
            # rejects None values inside dict samples.
            "semantic_preview_path": str(self.semantic_preview_paths[index]) if self.semantic_preview_paths[index] is not None else "",
            "semantic_kind": self.feature_kind,
        }
        if self.has_semantics:
            feature_path = self.semantic_feature_paths[index]
            if feature_path is None:
                features = torch.zeros(self.semantic_shape, dtype=torch.float32)
                semantics_valid = False
            else:
                features = torch.from_numpy(np.load(feature_path).copy())
                semantics_valid = True
            data["semantics"] = features.float()
            data["semantics_valid"] = torch.tensor(semantics_valid, dtype=torch.bool)
        return data


# ---------------------------------------------------------------------------
# Transforms.json (nerfstudio / instant-ngp) parser
# ---------------------------------------------------------------------------

class _TransformsJsonParser:
    """Parse transforms.json dataset for training."""

    def __init__(self, data_dir: str, factor: int = 1, test_every: int = 8, feature_kind: str = "raw"):
        import json

        self.data_dir = os.path.abspath(data_dir)
        self.factor = factor
        self.test_every = test_every
        self.feature_kind = feature_kind

        transforms_path = os.path.join(data_dir, "transforms.json")
        with open(transforms_path, "r") as f:
            meta = json.load(f)

        # Distortion is tracked per frame because on-the-go exports can vary
        # focal length and radial distortion slightly across frames.
        self.dist_params: Dict[int, np.ndarray] = {}
        self.undist_mapx: Dict[int, np.ndarray] = {}
        self.undist_mapy: Dict[int, np.ndarray] = {}
        self.undist_roi: Dict[int, Tuple[int, int, int, int]] = {}
        self.Ks_dict = {}
        self.imsize_dict = {}
        self.camera_ids = []
        self.camera_indices = []
        self.camera_id_to_idx = {}
        self.transform = np.eye(4, dtype=np.float64)
        self.transform_inv = np.eye(4, dtype=np.float64)
        self.has_semantics = False
        self.semantic_shape: tuple[int, ...] | None = None
        self.semantic_feature_paths: list[Optional[Path]] = []
        self.semantic_preview_paths: list[Optional[Path]] = []

        # Collect image data
        image_names = []
        image_paths = []
        camtoworlds = []
        points_records = _load_point_cloud_records(data_dir)

        frames = meta.get("frames", [])
        for camera_id, frame in enumerate(frames):
            file_path = frame.get("file_path", "")
            if not os.path.isabs(file_path):
                file_path = os.path.normpath(os.path.join(data_dir, file_path))
            name = os.path.basename(file_path)

            w = int(frame.get("w", meta.get("w", 1920)))
            h = int(frame.get("h", meta.get("h", 1080)))
            fl_x = float(frame.get("fl_x", meta.get("fl_x", 0.0)))
            fl_y = float(frame.get("fl_y", meta.get("fl_y", fl_x)))
            cx = float(frame.get("cx", meta.get("cx", w / 2.0)))
            cy = float(frame.get("cy", meta.get("cy", h / 2.0)))
            k1 = float(frame.get("k1", meta.get("k1", 0.0)))
            is_fisheye = bool(frame.get("is_fisheye", meta.get("is_fisheye", False)))

            if factor > 1:
                w //= factor
                h //= factor
                fl_x /= factor
                fl_y /= factor
                cx /= factor
                cy /= factor

            K = np.array([[fl_x, 0.0, cx], [0.0, fl_y, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
            if abs(k1) > 1e-8 and not is_fisheye:
                dist_params = np.array([k1, 0.0, 0.0, 0.0], dtype=np.float64)
                K_undist, roi = cv2.getOptimalNewCameraMatrix(K, dist_params, (w, h), 0)
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, dist_params, None, K_undist, (w, h), cv2.CV_32FC1
                )
                self.dist_params[camera_id] = dist_params
                self.undist_mapx[camera_id] = mapx
                self.undist_mapy[camera_id] = mapy
                self.undist_roi[camera_id] = roi
                K = K_undist
                w, h = int(roi[2]), int(roi[3])
                logger.info(
                    f"Frame {camera_id} SIMPLE_RADIAL undistortion: k1={k1:.6f}, roi={roi}"
                )

            self.Ks_dict[camera_id] = K
            self.imsize_dict[camera_id] = (w, h)

            transform_matrix = np.array(frame["transform_matrix"], dtype=np.float32)
            # transforms.json uses OpenGL-style c2w. Convert to the OpenCV/COLMAP
            # convention that gsplat expects by flipping camera Y/Z axes.
            camtoworld = _flip_c2w_to_gsplat(transform_matrix)

            camtoworlds.append(camtoworld)
            self.camera_ids.append(camera_id)
            image_names.append(name)
            image_paths.append(file_path)

        camtoworlds = np.stack(camtoworlds, axis=0)
        points = np.zeros((0, 3), dtype=np.float32)
        points_rgb = np.zeros((0, 3), dtype=np.uint8)
        if points_records:
            points = np.stack([p["xyz"] for p in points_records.values()], axis=0).astype(np.float32)
            points_rgb = np.stack([p["rgb"] for p in points_records.values()], axis=0).astype(np.uint8)

        if len(camtoworlds) > 0 and len(points) > 0:
            camtoworlds, points, transform = _normalize_world_space(camtoworlds, points)
            self.transform = transform
            self.transform_inv = np.linalg.inv(transform)

        # Sort by image name for consistent train/val splits
        inds = np.argsort(image_names)
        self.image_names = [image_names[i] for i in inds]
        self.image_paths = [image_paths[i] for i in inds]
        self.camtoworlds = camtoworlds[inds].astype(np.float32)
        self.camera_ids = [self.camera_ids[i] for i in inds]
        self.semantic_feature_paths, self.semantic_preview_paths, self.semantic_shape = build_semantic_feature_manifest(
            self.image_paths,
            dataset_root=self.data_dir,
            feature_kind=self.feature_kind,
        )
        self.has_semantics = self.semantic_shape is not None

        # Create 0-based contiguous camera indices
        unique_camera_ids = sorted(set(self.camera_ids))
        self.camera_id_to_idx = {cid: idx for idx, cid in enumerate(unique_camera_ids)}
        self.camera_indices = [self.camera_id_to_idx[cid] for cid in self.camera_ids]
        self.num_cameras = len(unique_camera_ids)

        # Point cloud: prefer the on-the-go dataset-root PointCloud.ply and fall
        # back to COLMAP sparse points if present.
        self.points = np.zeros((0, 3), dtype=np.float32)
        self.points_rgb = np.zeros((0, 3), dtype=np.uint8)
        if len(points) > 0:
            self.points = points.astype(np.float32)
            self.points_rgb = points_rgb.astype(np.uint8)

        # Scene scale
        if len(self.camtoworlds) > 0:
            camera_locations = self.camtoworlds[:, :3, 3]
            scene_center = np.mean(camera_locations, axis=0)
            dists = np.linalg.norm(camera_locations - scene_center, axis=1)
            self.scene_scale = float(np.max(dists))
        else:
            self.scene_scale = 1.0


class _TransformsJsonDataset(torch.utils.data.Dataset):
    """Dataset for transforms.json format."""

    def __init__(self, parser: _TransformsJsonParser, split: str = "train", use_eval_split: bool = False):
        self.parser = parser
        self.split = split
        indices = np.arange(len(self.parser.image_names))
        if use_eval_split:
            if split == "train":
                self.indices = indices[indices % self.parser.test_every != 0]
            else:
                self.indices = indices[indices % self.parser.test_every == 0]
        else:
            self.indices = indices if split == "train" else indices[:0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image_path = self.parser.image_paths[index]

        image = imageio.imread(image_path)[..., :3]

        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy().astype(np.float32)
        camtoworld = self.parser.camtoworlds[index].copy()
        image = _resize_image_if_needed(image, self.parser.imsize_dict[camera_id])

        # Undistort if distortion maps are present
        if camera_id in self.parser.undist_mapx:
            import cv2
            mapx = self.parser.undist_mapx[camera_id]
            mapy = self.parser.undist_mapy[camera_id]
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.undist_roi[camera_id]
            image = image[y:y + h, x:x + w]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K),
            "camtoworld": torch.from_numpy(camtoworld),
            "image": torch.from_numpy(image),
            "image_id": item,
            "camera_idx": self.parser.camera_indices[index],
            "semantic_preview_path": str(self.parser.semantic_preview_paths[index]) if self.parser.semantic_preview_paths[index] is not None else "",
            "semantic_kind": self.parser.feature_kind,
        }
        if self.parser.has_semantics:
            feature_path = self.parser.semantic_feature_paths[index]
            if feature_path is None:
                features = torch.zeros(self.parser.semantic_shape, dtype=torch.float32)
                semantics_valid = False
            else:
                features = torch.from_numpy(np.load(feature_path).copy())
                semantics_valid = True
            data["semantics"] = features.float()
            data["semantics_valid"] = torch.tensor(semantics_valid, dtype=torch.bool)
        return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _knn(x: torch.Tensor, K: int = 4) -> torch.Tensor:
    """kNN using scikit-learn KD-tree (fast, low memory)."""
    from sklearn.neighbors import NearestNeighbors

    if x.shape[0] == 0:
        return torch.empty((0, K), dtype=x.dtype, device=x.device)
    K = min(K, x.shape[0])
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Simple PyTorch SSIM implementation."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, window_size, 1, window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, 1, window_size // 2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, 1, window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, 1, window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


_compiled_ssim = None


def _get_compiled_ssim():
    global _compiled_ssim
    if _compiled_ssim is None:
        # Keep SSIM uncompiled so repeated stop/reset/start cycles do not trip
        # over stale torch.compile / cudagraph state.
        _compiled_ssim = _ssim
    return _compiled_ssim


def _reset_compiled_ssim():
    global _compiled_ssim
    _compiled_ssim = None
    try:
        torch._dynamo.reset()
    except Exception:
        pass


def _rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB to SH DC coefficients."""
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def _create_splats_and_optimizers(
    parser: _SimpleColmapParser,
    device: str,
    sh_degree: int = 3,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
):
    """Create splats and optimizers from COLMAP parser point cloud."""
    points = torch.from_numpy(parser.points).float()
    rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()

    N = points.shape[0]
    if N == 0:
        logger.warning(
            "Empty point cloud reached splat initialization; creating 1000 fallback points in a 5x5x5 box"
        )
        rng = np.random.default_rng(42)
        points = torch.from_numpy(rng.uniform(-2.5, 2.5, size=(1000, 3)).astype(np.float32)).float()
        rgbs = torch.from_numpy(rng.integers(96, 256, size=(1000, 3), dtype=np.uint8) / 255.0).float()
        N = 1000

    if N >= 2:
        # Initialize scales from kNN distances
        dist2_avg = (_knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(torch.clamp(dist_avg * init_scale, min=1e-6)).unsqueeze(-1).repeat(1, 3)
    else:
        scales = torch.full((N, 3), math.log(max(init_scale, 1e-6)), dtype=torch.float32)

    quats = torch.rand((N, 4))
    opacities = torch.logit(torch.full((N,), init_opacity))

    # SH colors
    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))
    colors[:, 0, :] = _rgb_to_sh(rgbs)

    params = [
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
        ("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3),
        ("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20),
    ]

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    optimizers = {
        name: torch.optim.Adam(
            [{"params": splats[name], "lr": lr, "name": name}],
            eps=1e-15,
            betas=(0.9, 0.999),
            fused=True,
        )
        for name, _, lr in params
    }

    return splats, optimizers


# ---------------------------------------------------------------------------
# Config and Trainer
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    max_splats: int = 1_000_000
    num_iterations: int = 30_000
    ssim_lambda: float = 0.0
    loss_type: str = "robust"
    evaluations: bool = False
    semantics: bool = False
    cluster: bool = False
    robust_percentile: float = 0.7
    schedule: bool = True
    schedule_beta: float = -3e-3
    lower_bound: float = 0.5
    upper_bound: float = 0.9
    bin_size: int = 10000
    sh_degree: int = 3
    batch_size: int = 1
    data_factor: int = 4
    test_every: int = 8
    device: str = "cuda"
    strategy: str = "default"  # "mcmc" or "default"
    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    reset_every: int = 3000
    refine_every: int = 200
    pause_refine_after_reset: int = 0
    absgrad: bool = False
    revised_opacity: bool = False
    strategy_verbose: bool = False
    opacity_reg: float = 0.0
    scale_reg: float = 0.0
    headless: bool = False
    output_dir: str = "./output"


class Trainer:
    """Background-thread 3DGS trainer using MCMC strategy."""

    def __init__(self, scene_state, config: Optional[TrainerConfig] = None):
        self.scene_state = scene_state
        self.config = config or TrainerConfig()

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Loss history: list of (iteration, loss_value)
        self.loss_history: List[Tuple[int, float]] = []
        self.eval_history: List[Tuple[int, float]] = []
        self.current_iteration = 0
        self.current_splats = 0
        self.is_running = False
        self.status_message = "Idle"
        self._train_start_time = 0.0

        # Training state (owned by training thread)
        self._splats = None
        self._optimizers = None
        self._strategy = None
        self._strategy_state = None
        self._trainloader = None
        self._trainloader_iter = None
        self._evalloader = None
        self._scheduler = None
        self._parser = None
        self._spotless_module = None
        self._spotless_optimizer = None
        self._spotless_chunk_size = 8192
        self._spotless_snapshot: Optional[Dict[str, Any]] = None
        self._spotless_snapshot_step = -1
        self._step_condition = threading.Condition()
        self._step_mode_enabled = False
        self._step_requests = 0
        self._semantics_available = False
        self._spotless_active = False
        self.running_stats = RunningStats(
            bin_size=self.config.bin_size,
            robust_percentile=self.config.robust_percentile,
            lower_bound=self.config.lower_bound,
            upper_bound=self.config.upper_bound,
        )

    def start(self, colmap_path: str):
        """Start training in a background thread."""
        if self.is_running:
            logger.warning("Trainer already running")
            return

        _reset_compiled_ssim()
        self._stop_event.clear()
        self.loss_history.clear()
        self.eval_history.clear()
        self.current_iteration = 0
        self.status_message = "Initializing..."
        self._spotless_snapshot = None
        self._spotless_snapshot_step = -1
        self._spotless_module = None
        self._spotless_optimizer = None
        self._semantics_available = False
        self._spotless_active = False
        self.running_stats = RunningStats(
            bin_size=self.config.bin_size,
            robust_percentile=self.config.robust_percentile,
            lower_bound=self.config.lower_bound,
            upper_bound=self.config.upper_bound,
        )
        with self._step_condition:
            self._step_requests = 0

        self._thread = threading.Thread(
            target=self._train_loop, args=(colmap_path,), daemon=True
        )
        self._thread.start()

    def stop(self):
        """Signal the training thread to stop and wait for it."""
        if not self.is_running:
            return
        self._stop_event.set()
        with self._step_condition:
            self._step_condition.notify_all()
        self.status_message = "Stopping..."
        if self._thread:
            self._thread.join(timeout=5.0)
        self.is_running = False
        self.status_message = "Stopped"

    def reset(self):
        """Clear all training state and remove learned gaussians from the scene."""
        if self.is_running:
            logger.warning("Cannot reset training while it is running; stop it first")
            return

        _reset_compiled_ssim()
        with self._lock:
            self._splats = None
            self._optimizers = None
            self._strategy = None
            self._strategy_state = None
            self._trainloader = None
            self._trainloader_iter = None
            self._evalloader = None
            self._scheduler = None
            self._parser = None
            self._spotless_module = None
            self._spotless_optimizer = None
            self._spotless_snapshot = None
            self._spotless_snapshot_step = -1
            self.loss_history.clear()
            self.eval_history.clear()
            self.current_iteration = 0
            self.current_splats = 0
            self.status_message = "Reset"
            self._step_mode_enabled = False
            self._step_requests = 0
        self.running_stats = RunningStats(
            bin_size=self.config.bin_size,
            robust_percentile=self.config.robust_percentile,
            lower_bound=self.config.lower_bound,
            upper_bound=self.config.upper_bound,
        )
        self.scene_state.clear_gaussians()
        logger.info("Training state reset; learned gaussians cleared")

    def set_step_mode(self, enabled: bool):
        with self._step_condition:
            self._step_mode_enabled = enabled
            if not enabled:
                self._step_condition.notify_all()

    def request_next_step(self):
        with self._step_condition:
            self._step_requests += 1
            self._step_condition.notify_all()

    def get_spotless_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._spotless_snapshot is None:
                return None
            snapshot = dict(self._spotless_snapshot)
        return snapshot

    def _wait_for_step_request(self):
        with self._step_condition:
            while self._step_mode_enabled and self._step_requests <= 0 and not self._stop_event.is_set():
                self.status_message = "Paused"
                self._step_condition.wait(timeout=0.1)
            if self._stop_event.is_set():
                return False
            if self._step_requests > 0:
                self._step_requests -= 1
        return True

    def _train_loop(self, colmap_path: str):
        try:
            self.is_running = True
            cfg = self.config
            device = cfg.device
            evaluations_enabled = bool(cfg.evaluations)

            logger.info(f"[TRAINER] CUDA available: {torch.cuda.is_available()}")
            logger.info(f"[TRAINER] Using device: {device}")
            if torch.cuda.is_available():
                logger.info(f"[TRAINER] CUDA device: {torch.cuda.get_device_name(0)}")
                # Enable TF32 for faster matmul on Ampere+ GPUs
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision('high')
                # Allow cuDNN to benchmark optimal algorithms for consistent input sizes
                torch.backends.cudnn.benchmark = True

            # Auto-detect dataset format
            import os
            transforms_path = os.path.join(colmap_path, "transforms.json")
            feature_kind = "clustered" if cfg.cluster else "raw"
            if os.path.exists(transforms_path):
                logger.info(f"Detected transforms.json dataset at {colmap_path}")
                parser = _TransformsJsonParser(
                    data_dir=colmap_path,
                    factor=cfg.data_factor,
                    test_every=cfg.test_every,
                    feature_kind=feature_kind,
                )
                trainset = _TransformsJsonDataset(
                    parser,
                    split="train",
                    use_eval_split=evaluations_enabled,
                )
                evalset = _TransformsJsonDataset(
                    parser,
                    split="eval",
                    use_eval_split=evaluations_enabled,
                )
            else:
                logger.info(f"Detected COLMAP dataset at {colmap_path}")
                parser = _SimpleColmapParser(
                    data_dir=colmap_path,
                    factor=cfg.data_factor,
                    test_every=cfg.test_every,
                )
                trainset = _SimpleColmapDataset(
                    parser,
                    split="train",
                    feature_kind=feature_kind,
                    use_eval_split=evaluations_enabled,
                )
                evalset = _SimpleColmapDataset(
                    parser,
                    split="eval",
                    feature_kind=feature_kind,
                    use_eval_split=evaluations_enabled,
                )
            self._parser = parser
            images_dir = os.path.join(parser.data_dir, "images")
            raw_count = 0
            clustered_count = 0
            if os.path.isdir(images_dir):
                raw_count = len(list(Path(images_dir).glob("*_sdfeats.npy")))
                clustered_count = len(list(Path(images_dir).glob("*_sdfeats_clustered.npy")))
            desired_count = clustered_count if feature_kind == "clustered" else raw_count
            opposite_count = raw_count if feature_kind == "clustered" else clustered_count
            self._semantics_available = desired_count > 0
            if self._semantics_available and not cfg.semantics:
                cfg.semantics = True
                logger.info(f"Auto-enabled spotless semantics because {feature_kind} feature files were found")
            if cfg.semantics and not self._semantics_available:
                logger.warning(
                    "Spotless semantics requested but the expected feature kind was not found "
                    f"under {images_dir} (kind={feature_kind}, raw={raw_count}, clustered={clustered_count})"
                )
                if opposite_count > 0:
                    logger.warning(
                        f"Found the opposite semantic kind instead; not falling back automatically "
                        f"(opposite_kind={'raw' if feature_kind == 'clustered' else 'clustered'})"
                    )
                cfg.semantics = False

            scene_scale = parser.scene_scale * 1.1

            # Create splats from point cloud
            self._splats, self._optimizers = _create_splats_and_optimizers(
                parser,
                device,
                sh_degree=cfg.sh_degree,
                scene_scale=scene_scale,
            )
            logger.info(f"Initialized {len(self._splats['means'])} splats from point cloud")
            logger.info(f"[TRAINER] Splats device: {self._splats['means'].device}")
            self._spotless_active = cfg.semantics and self._semantics_available
            if self._spotless_active and not cfg.cluster:
                semantic_shape = getattr(trainset, "semantic_shape", None)
                if semantic_shape is None:
                    semantic_shape = _discover_semantic_shape(
                        getattr(parser, "image_paths", []),
                        dataset_root=getattr(parser, "data_dir", None),
                        feature_kind=feature_kind,
                    )
                if semantic_shape is None:
                    logger.warning(
                        "Spotless semantics are enabled but no feature shape could be inferred; "
                        "the SpotLess MLP will initialize lazily from the first valid batch"
                    )
                else:
                    if len(semantic_shape) < 3:
                        logger.warning(
                            f"Invalid semantic feature shape: {semantic_shape}; "
                            "the SpotLess MLP will initialize lazily from the first valid batch"
                        )
                    else:
                        semantic_channels = int(semantic_shape[0])
                        self._spotless_module = SpotLessModule(
                            num_features=semantic_channels + 80,
                            num_classes=1,
                        ).to(device)
                        self._spotless_optimizer = torch.optim.Adam(
                            self._spotless_module.parameters(),
                            lr=1e-3,
                        )
                        logger.info(
                            f"Initialized SpotLessModule with {semantic_channels} feature channels "
                            f"({semantic_channels + 80} with positional encoding)"
                        )

            # Setup strategy
            if cfg.strategy == "mcmc":
                self._strategy = MCMCStrategy(
                    cap_max=cfg.max_splats,
                    refine_start_iter=cfg.refine_start_iter,
                    refine_stop_iter=cfg.refine_stop_iter,
                    refine_every=cfg.refine_every,
                    min_opacity=cfg.prune_opa,
                    verbose=cfg.strategy_verbose,
                )
            else:
                self._strategy = DefaultStrategy(
                    prune_opa=cfg.prune_opa,
                    grow_grad2d=cfg.grow_grad2d,
                    grow_scale3d=cfg.grow_scale3d,
                    grow_scale2d=cfg.grow_scale2d,
                    prune_scale3d=cfg.prune_scale3d,
                    prune_scale2d=cfg.prune_scale2d,
                    refine_scale2d_stop_iter=cfg.refine_scale2d_stop_iter,
                    refine_start_iter=cfg.refine_start_iter,
                    refine_stop_iter=cfg.refine_stop_iter,
                    reset_every=cfg.reset_every,
                    refine_every=cfg.refine_every,
                    pause_refine_after_reset=cfg.pause_refine_after_reset,
                    absgrad=cfg.absgrad,
                    revised_opacity=cfg.revised_opacity,
                    verbose=cfg.strategy_verbose,
                )
            self._strategy.check_sanity(self._splats, self._optimizers)
            if isinstance(self._strategy, DefaultStrategy):
                self._strategy_state = self._strategy.initialize_state(scene_scale=scene_scale)
            else:
                self._strategy_state = self._strategy.initialize_state()

            # Learning rate scheduler for means
            self._scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self._optimizers["means"], gamma=0.01 ** (1.0 / cfg.num_iterations)
            )

            # DataLoader: keep a few workers alive for prefetching and to
            # avoid serializing image/feature loads on the training thread.
            num_workers = 4
            loader_kwargs = {
                "batch_size": cfg.batch_size,
                "shuffle": True,
                "num_workers": num_workers,
                "persistent_workers": num_workers > 0,
                "pin_memory": num_workers > 0,
            }
            self._trainloader = torch.utils.data.DataLoader(trainset, **loader_kwargs)
            self._trainloader_iter = iter(self._trainloader)
            self._evalloader = None
            if evaluations_enabled and len(evalset) > 0:
                eval_loader_kwargs = {
                    "batch_size": 1,
                    "shuffle": False,
                    "num_workers": num_workers,
                    "persistent_workers": num_workers > 0,
                    "pin_memory": num_workers > 0,
                }
                self._evalloader = torch.utils.data.DataLoader(evalset, **eval_loader_kwargs)

            # Training loop
            self._train_start_time = time.time()
            for step in range(cfg.num_iterations):
                if self._stop_event.is_set():
                    break
                if self._step_mode_enabled and not self._wait_for_step_request():
                    break

                t_data_start = time.time()
                try:
                    data = next(self._trainloader_iter)
                except StopIteration:
                    self._trainloader_iter = iter(self._trainloader)
                    data = next(self._trainloader_iter)
                t_data = time.time() - t_data_start

                t_gpu_start = time.time()
                camtoworlds = data["camtoworld"].to(device, non_blocking=True)
                Ks = data["K"].to(device, non_blocking=True)
                pixels = data["image"].to(device, non_blocking=True) / 255.0
                sample_semantic_kind = _unwrap_singleton_string(data.get("semantic_kind"))

                height, width = pixels.shape[1:3]

                # Use the SH degree selected in the UI
                sh_degree_to_use = cfg.sh_degree

                # Forward
                means = self._splats["means"]
                quats = self._splats["quats"]
                scales = torch.exp(self._splats["scales"])
                opacities = torch.sigmoid(self._splats["opacities"])
                colors = torch.cat([self._splats["sh0"], self._splats["shN"]], 1)

                if step == 0:
                    logger.info(f"[TRAINER] means device: {means.device}")
                    logger.info(f"[TRAINER] pixels device: {pixels.device}")
                    logger.info(f"[TRAINER] image size: {height}x{width}")

                renders, alphas, info = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=torch.linalg.inv(camtoworlds),
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_to_use,
                    near_plane=0.01,
                    far_plane=1e10,
                    packed=False,
                )

                colors_render = renders[..., :3].clamp(0.0, 1.0)
                error_per_pixel = torch.abs(colors_render - pixels)
                use_spotless = bool(cfg.semantics and self._semantics_available)
                semantics_valid = data.get("semantics_valid", None)
                if semantics_valid is not None:
                    if torch.is_tensor(semantics_valid):
                        if not bool(torch.all(semantics_valid).item()):
                            use_spotless = False
                    elif not bool(semantics_valid):
                        use_spotless = False
                if sample_semantic_kind is not None and sample_semantic_kind != feature_kind:
                    logger.warning(
                        f"Semantic kind mismatch for training sample: expected {feature_kind}, got {sample_semantic_kind}"
                    )
                    use_spotless = False

                pred_mask = None
                pred_mask_up = None
                lower_mask = None
                upper_mask = None
                loss_map = None
                semantic_features = None
                semantic_features_snapshot = None
                semantic_preview_path = data.get("semantic_preview_path")
                semantic_kind = None
                raw_spotless_chunked = False
                publish_snapshot = (step == 0) or self._step_mode_enabled or ((step + 1) % 25 == 0)

                if cfg.loss_type == "robust" or use_spotless:
                    pred_mask = robust_mask(error_per_pixel, self.running_stats.avg_err)
                    if use_spotless:
                        semantic_features = data.get("semantics")
                        if semantic_features is not None:
                            semantic_features_snapshot = semantic_features
                            semantic_features = semantic_features.to(device, non_blocking=True).float()
                            if semantic_features.ndim == 3:
                                semantic_features = semantic_features.unsqueeze(0)
                            if cfg.cluster:
                                semantic_kind = "clustered"
                                semantic_features = F.interpolate(
                                    semantic_features,
                                    size=(height, width),
                                    mode="nearest",
                                )
                                pred_mask = robust_cluster_mask(pred_mask, semantic_features)
                            else:
                                semantic_kind = "raw"
                                if self._spotless_module is None:
                                    semantic_channels = int(semantic_features.shape[1])
                                    self._spotless_module = SpotLessModule(
                                        num_features=semantic_channels + 80,
                                        num_classes=1,
                                    ).to(device)
                                    self._spotless_optimizer = torch.optim.Adam(
                                        self._spotless_module.parameters(),
                                        lr=1e-3,
                                    )
                                    logger.info(
                                        f"Lazily initialized SpotLessModule with {semantic_channels} feature channels"
                                    )
                                if self._spotless_module is not None:
                                    raw_spotless_chunked = True
                                    lower_mask = robust_mask(
                                        error_per_pixel, self.running_stats.lower_err
                                    )
                                    upper_mask = robust_mask(
                                        error_per_pixel, self.running_stats.upper_err
                                    )
                                    pixel_count = height * width
                                    chunk_size = min(self._spotless_chunk_size, pixel_count)
                                    flat_error = error_per_pixel.reshape(-1, error_per_pixel.shape[-1])
                                    flat_lower = lower_mask.reshape(-1, 1)
                                    flat_upper = upper_mask.reshape(-1, 1)
                                    rgb_loss_sum = torch.zeros((), device=device)
                                    spot_loss_sum = torch.zeros((), device=device)
                                    loss_map_chunks: list[torch.Tensor] = []
                                    pred_mask_chunks: list[torch.Tensor] = []

                                    self._spotless_module.eval()
                                    for start in range(0, pixel_count, chunk_size):
                                        end = min(start + chunk_size, pixel_count)
                                        idx = torch.arange(start, end, device=device)
                                        ys = torch.div(idx, width, rounding_mode="floor")
                                        xs = idx % width
                                        grid_x = ((xs.to(torch.float32) + 0.5) / float(width)) * 2.0 - 1.0
                                        grid_y = ((ys.to(torch.float32) + 0.5) / float(height)) * 2.0 - 1.0
                                        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)

                                        sampled_features = F.grid_sample(
                                            semantic_features,
                                            grid,
                                            mode="bilinear",
                                            align_corners=False,
                                        )
                                        sampled_features = sampled_features.squeeze(0).squeeze(-1).transpose(0, 1)
                                        pos_enc = get_positional_encodings_from_coords(
                                            xs,
                                            ys,
                                            height,
                                            width,
                                            20,
                                        )
                                        spotless_input = torch.cat([sampled_features, pos_enc], dim=-1)
                                        pred_chunk = self._spotless_module(spotless_input)

                                        if publish_snapshot:
                                            pred_mask_chunks.append(pred_chunk.detach().cpu())

                                        sampled_mask = pred_chunk.detach()
                                        if cfg.schedule:
                                            alpha = np.exp(cfg.schedule_beta * np.floor((1 + step) / 1.5))
                                            sampled_mask = torch.bernoulli(
                                                torch.clip(
                                                    alpha + (1 - alpha) * sampled_mask,
                                                    min=0.0,
                                                    max=1.0,
                                                )
                                            )

                                        lower_chunk = flat_lower[start:end]
                                        upper_chunk = flat_upper[start:end]
                                        err_chunk = flat_error[start:end]
                                        if publish_snapshot:
                                            loss_map_chunks.append(
                                                torch.mean((sampled_mask * err_chunk).detach().cpu(), dim=-1, keepdim=True)
                                            )
                                        spot_loss_sum = spot_loss_sum + (
                                            F.relu(pred_chunk - upper_chunk) + F.relu(lower_chunk - pred_chunk)
                                        ).sum()

                                        rgb_loss_sum = rgb_loss_sum + (sampled_mask * err_chunk).sum()

                                    pred_mask = torch.cat(pred_mask_chunks, dim=0).reshape(1, height, width, 1) if pred_mask_chunks else None
                                    loss_map = (
                                        torch.cat(loss_map_chunks, dim=0).reshape(1, height, width, 1)
                                        if loss_map_chunks
                                        else None
                                    )
                                    rgbloss = rgb_loss_sum / float(pixel_count * error_per_pixel.shape[-1])
                                    spot_loss = spot_loss_sum / float(pixel_count)
                                    spot_loss = spot_loss + 0.5 * self._spotless_module.get_regularizer()
                    if not raw_spotless_chunked:
                        if cfg.schedule:
                            alpha = np.exp(cfg.schedule_beta * np.floor((1 + step) / 1.5))
                            pred_mask = torch.bernoulli(
                                torch.clip(
                                    alpha + (1 - alpha) * pred_mask.detach(),
                                    min=0.0,
                                    max=1.0,
                                )
                            )
                        if pred_mask is not None:
                            masked_error = pred_mask.detach() * error_per_pixel
                            rgbloss = masked_error.mean()
                            loss_map = torch.mean(masked_error, dim=-1, keepdim=True)
                        else:
                            rgbloss = error_per_pixel.mean()
                            loss_map = torch.mean(error_per_pixel, dim=-1, keepdim=True)
                    elif loss_map is None:
                        if pred_mask is not None:
                            loss_map = torch.mean(pred_mask.detach() * error_per_pixel, dim=-1, keepdim=True)
                        else:
                            loss_map = torch.mean(error_per_pixel, dim=-1, keepdim=True)
                else:
                    # Loss: SSIM + L1
                    l1loss = F.l1_loss(colors_render, pixels)
                    ssimloss = 1.0 - _get_compiled_ssim()(
                        colors_render.permute(0, 3, 1, 2),
                        pixels.permute(0, 3, 1, 2),
                    )
                    rgbloss = l1loss
                    loss = torch.lerp(l1loss, ssimloss, cfg.ssim_lambda)
                    loss_map = torch.mean(error_per_pixel, dim=-1, keepdim=True)

                if cfg.loss_type == "robust" or use_spotless:
                    ssimloss = 1.0 - _get_compiled_ssim()(
                        colors_render.permute(0, 3, 1, 2),
                        pixels.permute(0, 3, 1, 2),
                    )
                    loss = rgbloss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

                if cfg.opacity_reg > 0.0:
                    loss = loss + cfg.opacity_reg * torch.sigmoid(self._splats["opacities"]).mean()
                if cfg.scale_reg > 0.0:
                    loss = loss + cfg.scale_reg * torch.exp(self._splats["scales"]).mean()

                if raw_spotless_chunked:
                    self._spotless_module.train()
                else:
                    spot_loss = None

                if publish_snapshot:
                    # Snapshot visualization should reflect the true per-pixel
                    # reconstruction error, not the mask-weighted training target.
                    error_map = torch.mean(error_per_pixel, dim=-1, keepdim=True)
                    semantic_preview_path = data.get("semantic_preview_path")
                    if isinstance(semantic_preview_path, (list, tuple)) and semantic_preview_path:
                        semantic_preview_path = semantic_preview_path[0]
                    self._publish_spotless_snapshot(
                        step=step + 1,
                        pixels=pixels,
                        colors_render=colors_render,
                        pred_mask=pred_mask,
                        error_map=error_map,
                        semantic_features=semantic_features_snapshot,
                        semantic_preview_path=semantic_preview_path,
                        semantic_kind=semantic_kind,
                        loss=float(loss.item()),
                        spot_loss=float(spot_loss.item()) if spot_loss is not None else None,
                    )

                # Strategy pre-backward (DefaultStrategy only)
                if isinstance(self._strategy, DefaultStrategy):
                    self._strategy.step_pre_backward(
                        params=self._splats,
                        optimizers=self._optimizers,
                        state=self._strategy_state,
                        step=step,
                        info=info,
                    )

                # Backward
                loss.backward()
                if spot_loss is not None and self._spotless_optimizer is not None:
                    spot_loss.backward()

                # Optimize
                for optimizer in self._optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if self._spotless_optimizer is not None:
                    self._spotless_optimizer.step()
                    self._spotless_optimizer.zero_grad(set_to_none=True)
                self._scheduler.step()

                # Strategy post-backward
                if isinstance(self._strategy, MCMCStrategy):
                    self._strategy.step_post_backward(
                        params=self._splats,
                        optimizers=self._optimizers,
                        state=self._strategy_state,
                        step=step,
                        info=info,
                        lr=self._scheduler.get_last_lr()[0],
                    )
                else:
                    self._strategy.step_post_backward(
                        params=self._splats,
                        optimizers=self._optimizers,
                        state=self._strategy_state,
                        step=step,
                        info=info,
                        packed=False,
                    )

                if cfg.loss_type == "robust" or use_spotless:
                    info["err"] = torch.histogram(
                        torch.mean(error_per_pixel, dim=-1).detach().cpu(),
                        bins=cfg.bin_size,
                        range=(0.0, 1.0),
                    )[0]
                    self.running_stats.update(info["err"])

                self.current_iteration = step + 1
                self.current_splats = len(self._splats['means'])

                # Every 100 iterations: update viewer and log loss
                if (step + 1) % 100 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_gpu = time.time() - t_gpu_start
                    loss_val = loss.item()
                    with self._lock:
                        self.loss_history.append((step + 1, loss_val))

                    # Export gaussians to scene_state (skip in headless mode)
                    if not cfg.headless:
                        self._export_gaussians()

                    elapsed = time.time() - self._train_start_time
                    it_per_sec = (step + 1) / elapsed if elapsed > 0 else 0.0
                    self.status_message = (
                        f"{step + 1}/{cfg.num_iterations} ({it_per_sec:.1f} it/s)"
                    )
                    logger.info(
                        f"Training... {self.status_message}, "
                        f"loss={loss_val:.4f}, splats={self.current_splats}"
                    )
                    logger.info(f"[TRAINER] Timing: data={t_data:.3f}s, gpu={t_gpu:.3f}s")

                if evaluations_enabled and self._evalloader is not None and (step + 1) % 1000 == 0:
                    eval_psnr = self._evaluate_psnr(device=device)
                    if eval_psnr is not None:
                        with self._lock:
                            self.eval_history.append((step + 1, eval_psnr))
                        logger.info(f"[TRAINER] Eval PSNR at step {step + 1}: {eval_psnr:.3f}")

            self.status_message = "Training complete"
            logger.info("Training complete")

            # Save final PLY in headless mode
            if cfg.headless and self._splats is not None:
                self._save_ply(colmap_path)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.status_message = f"Error: {e}"
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False

    def _export_gaussians(self):
        """Export current splats to scene_state.gaussians in viewer format."""
        if self._splats is None:
            return

        with torch.no_grad():
            means = self._splats["means"].detach().clone()
            quats = F.normalize(self._splats["quats"], dim=-1).detach().clone()
            scales = torch.exp(self._splats["scales"]).detach().clone()
            opacities = torch.sigmoid(self._splats["opacities"]).detach().clone()
            colors = torch.cat(
                [self._splats["sh0"], self._splats["shN"]], 1
            ).detach().clone()

            # If the dataset was normalized for training, transform the Gaussians
            # back into the original scene frame before exposing them to the viewer.
            if self._parser is not None and hasattr(self._parser, "transform_inv"):
                inv_transform = self._parser.transform_inv.astype(np.float32)
                means_np = means.cpu().numpy()
                scales_np = scales.cpu().numpy()
                quats_np = quats.cpu().numpy()
                means_np = _transform_points(inv_transform, means_np)
                scale_factor = float(np.linalg.norm(inv_transform[0, :3]))
                scales_np = scales_np * scale_factor
                rot = inv_transform[:3, :3]
                rot_scale = float(np.linalg.norm(rot[0]))
                if rot_scale > 0:
                    rot = rot / rot_scale
                q_rot = _quat_from_matrix(rot)
                quats_np = np.stack([_quat_mul(q_rot, q) for q in quats_np], axis=0)
                means = torch.from_numpy(means_np).to(means.device)
                scales = torch.from_numpy(scales_np).to(scales.device)
                quats = torch.from_numpy(quats_np).to(quats.device)

            # COLMAP/Ply scenes are stored in the viewer with a Y flip; the
            # transforms.json path already lives in the viewer's raw frame.
            y_flipped = not isinstance(self._parser, _TransformsJsonParser)
            if y_flipped:
                means[:, 1] *= -1
                # Quaternions: [w, x, y, z] -> [w, -x, y, -z] for Y flip
                quats[:, 1] *= -1
                quats[:, 3] *= -1

            gaussians = {
                "means": means,
                "quats": quats,
                "scales": scales,
                "opacities": opacities,
                "colors": colors,
                "sh_degree": self.config.sh_degree,
            }

            # Update scene_state under its lock
            with self.scene_state._lock:
                self.scene_state.gaussians = gaussians
                self.scene_state.gaussian_y_flipped = y_flipped
                self.scene_state.has_gaussians = True
                self.scene_state._default_gaussians = False
                self.scene_state._bump_version()

    def _save_ply(self, colmap_path: str):
        """Export current splats to a PLY file."""
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)
        scene_name = Path(colmap_path).name or "scene"
        ply_path = os.path.join(cfg.output_dir, f"{scene_name}.ply")

        # Reuse the same scene export path as the UI button so the written PLY
        # matches the current viewer/training coordinate convention.
        self._export_gaussians()
        self.scene_state.export_ply(ply_path)
        logger.info(f"Saved PLY to {ply_path}")

    @torch.no_grad()
    def _evaluate_psnr(self, device: str) -> Optional[float]:
        """Evaluate PSNR on the held-out split when evaluations are enabled."""
        if self._evalloader is None or self._splats is None:
            return None

        try:
            from torchmetrics.image import PeakSignalNoiseRatio

            psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
            use_torchmetrics = True
        except ImportError:
            logger.warning(
                "torchmetrics is not installed; falling back to a local PSNR computation"
            )
            psnr_metric = None
            use_torchmetrics = False

        sample_count = 0
        sq_error_sum = 0.0
        pixel_count = 0
        for data in self._evalloader:
            camtoworlds = data["camtoworld"].to(device, non_blocking=True)
            Ks = data["K"].to(device, non_blocking=True)
            pixels = data["image"].to(device, non_blocking=True) / 255.0
            height, width = pixels.shape[1:3]

            means = self._splats["means"]
            quats = self._splats["quats"]
            scales = torch.exp(self._splats["scales"])
            opacities = torch.sigmoid(self._splats["opacities"])
            colors = torch.cat([self._splats["sh0"], self._splats["shN"]], 1)

            renders, _, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(camtoworlds),
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=self.config.sh_degree,
                near_plane=0.01,
                far_plane=1e10,
                packed=False,
            )
            colors_render = renders[..., :3].clamp(0.0, 1.0).permute(0, 3, 1, 2)
            pixels = pixels.permute(0, 3, 1, 2)
            if use_torchmetrics:
                psnr_metric.update(colors_render, pixels)
            else:
                diff = colors_render - pixels
                sq_error_sum += float(torch.sum(diff * diff).item())
                pixel_count += int(diff.numel())
            sample_count += 1

        if sample_count == 0:
            return None

        if use_torchmetrics:
            return float(psnr_metric.compute().item())

        if pixel_count == 0:
            return None
        mse = sq_error_sum / pixel_count
        if mse <= 0.0:
            return float("inf")
        return float(-10.0 * math.log10(mse))

    def _publish_spotless_snapshot(
        self,
        *,
        step: int,
        pixels: torch.Tensor,
        colors_render: torch.Tensor,
        pred_mask: Optional[torch.Tensor],
        error_map: Optional[torch.Tensor],
        semantic_features: Optional[torch.Tensor],
        semantic_preview_path: Optional[str],
        semantic_kind: Optional[str],
        loss: float,
        spot_loss: Optional[float],
    ) -> None:
        """Store the latest training batch for the Spotless inspection panel."""

        def _to_numpy(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
            if t is None:
                return None
            return t.detach().float().cpu().numpy()

        with self._lock:
            self._spotless_snapshot = {
                "step": step,
                "loss": loss,
                "spot_loss": spot_loss,
                "gt": _to_numpy(pixels[0].clamp(0.0, 1.0)),
                "render": _to_numpy(colors_render[0].clamp(0.0, 1.0)),
                "mask": _to_numpy(pred_mask[0].clamp(0.0, 1.0)) if pred_mask is not None else None,
                "error_map": _to_numpy(error_map[0]) if error_map is not None else None,
                "semantics": _to_numpy(semantic_features[0]) if semantic_features is not None else None,
                "semantic_preview_path": semantic_preview_path,
                "semantic_kind": semantic_kind,
            }
            self._spotless_snapshot_step = step
