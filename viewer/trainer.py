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
from gsplat import export_splats
from viewer.scene.loaders import _load_point_cloud_records


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

class _SimpleColmapParser:
    """Minimal COLMAP parser using pycolmap 4.x API."""

    def __init__(self, data_dir: str, factor: int = 1, test_every: int = 8):
        import pycolmap

        self.data_dir = data_dir
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
        self.points = np.array(points_list, dtype=np.float32)
        self.points_rgb = np.array(colors_list, dtype=np.uint8)

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

    def __init__(self, parser: _SimpleColmapParser, split: str = "train"):
        self.parser = parser
        self.split = split
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

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

        data = {
            "K": torch.from_numpy(K),
            "camtoworld": torch.from_numpy(camtoworld),
            "image": torch.from_numpy(image),
            "image_id": item,
            "camera_idx": self.parser.camera_indices[index],
        }
        return data


# ---------------------------------------------------------------------------
# Transforms.json (nerfstudio / instant-ngp) parser
# ---------------------------------------------------------------------------

class _TransformsJsonParser:
    """Parse transforms.json dataset for training."""

    def __init__(self, data_dir: str, factor: int = 1, test_every: int = 8):
        import json

        self.data_dir = data_dir
        self.factor = factor
        self.test_every = test_every

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

    def __init__(self, parser: _TransformsJsonParser, split: str = "train"):
        self.parser = parser
        self.split = split
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image_path = self.parser.image_paths[index]

        image = imageio.imread(image_path)[..., :3]

        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy().astype(np.float32)
        camtoworld = self.parser.camtoworlds[index].copy()

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
        }
        return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _knn(x: torch.Tensor, K: int = 4) -> torch.Tensor:
    """kNN using scikit-learn KD-tree (fast, low memory)."""
    from sklearn.neighbors import NearestNeighbors

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


# Compile SSIM for speed (low-overhead mode for small graphs)
_compiled_ssim = torch.compile(_ssim, mode="reduce-overhead", dynamic=False, fullgraph=False)


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
):
    """Create splats and optimizers from COLMAP parser point cloud."""
    points = torch.from_numpy(parser.points).float()
    rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()

    # Initialize scales from kNN distances
    dist2_avg = (_knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)

    N = points.shape[0]
    quats = torch.rand((N, 4))
    opacities = torch.logit(torch.full((N,), init_opacity))

    # SH colors
    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))
    colors[:, 0, :] = _rgb_to_sh(rgbs)

    params = [
        ("means", torch.nn.Parameter(points), 1.6e-4),
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
    ssim_lambda: float = 0.2
    sh_degree: int = 3
    batch_size: int = 1
    data_factor: int = 4
    test_every: int = 8
    device: str = "cuda"
    strategy: str = "mcmc"  # "mcmc" or "default"
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
        self._scheduler = None
        self._parser = None

    def start(self, colmap_path: str):
        """Start training in a background thread."""
        if self.is_running:
            logger.warning("Trainer already running")
            return

        self._stop_event.clear()
        self.loss_history.clear()
        self.current_iteration = 0
        self.status_message = "Initializing..."

        self._thread = threading.Thread(
            target=self._train_loop, args=(colmap_path,), daemon=True
        )
        self._thread.start()

    def stop(self):
        """Signal the training thread to stop and wait for it."""
        if not self.is_running:
            return
        self._stop_event.set()
        self.status_message = "Stopping..."
        if self._thread:
            self._thread.join(timeout=5.0)
        self.is_running = False
        self.status_message = "Stopped"

    def _train_loop(self, colmap_path: str):
        try:
            self.is_running = True
            cfg = self.config
            device = cfg.device

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
            if os.path.exists(transforms_path):
                logger.info(f"Detected transforms.json dataset at {colmap_path}")
                parser = _TransformsJsonParser(
                    data_dir=colmap_path,
                    factor=cfg.data_factor,
                    test_every=cfg.test_every,
                )
                trainset = _TransformsJsonDataset(parser, split="train")
            else:
                logger.info(f"Detected COLMAP dataset at {colmap_path}")
                parser = _SimpleColmapParser(
                    data_dir=colmap_path,
                    factor=cfg.data_factor,
                    test_every=cfg.test_every,
                )
                trainset = _SimpleColmapDataset(parser, split="train")
            self._parser = parser

            scene_scale = parser.scene_scale * 1.1

            # Create splats from point cloud
            self._splats, self._optimizers = _create_splats_and_optimizers(
                parser, device, sh_degree=cfg.sh_degree
            )
            logger.info(f"Initialized {len(self._splats['means'])} splats from point cloud")
            logger.info(f"[TRAINER] Splats device: {self._splats['means'].device}")

            # Setup strategy
            if cfg.strategy == "mcmc":
                self._strategy = MCMCStrategy(
                    cap_max=cfg.max_splats,
                    refine_start_iter=500,
                    refine_stop_iter=max(cfg.num_iterations - 5000, 1000),
                    refine_every=100,
                )
            else:
                self._strategy = DefaultStrategy(
                    refine_start_iter=500,
                    refine_stop_iter=max(cfg.num_iterations - 5000, 1000),
                    refine_every=100,
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

            # DataLoader: use num_workers>0 for prefetching and pin_memory
            # for faster async CPU->GPU transfers. persistent_workers avoids
            # process spawn overhead across epochs.
            loader_kwargs = {
                "batch_size": cfg.batch_size,
                "shuffle": True,
                "num_workers": 4,
                "persistent_workers": True,
                "pin_memory": True,
            }
            self._trainloader = torch.utils.data.DataLoader(trainset, **loader_kwargs)
            self._trainloader_iter = iter(self._trainloader)

            # Training loop
            self._train_start_time = time.time()
            for step in range(cfg.num_iterations):
                if self._stop_event.is_set():
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

                colors_render = renders[..., :3]

                # Loss: SSIM + L1
                l1loss = F.l1_loss(colors_render, pixels)
                ssimloss = 1.0 - _compiled_ssim(
                    colors_render.permute(0, 3, 1, 2),
                    pixels.permute(0, 3, 1, 2),
                )
                loss = torch.lerp(l1loss, ssimloss, cfg.ssim_lambda)

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

                # Optimize
                for optimizer in self._optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
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
            if not isinstance(self._parser, _TransformsJsonParser):
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
                self.scene_state.has_gaussians = True
                self.scene_state._default_gaussians = False
                self.scene_state._bump_version()

    def _save_ply(self, colmap_path: str):
        """Export current splats to a PLY file."""
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)
        scene_name = Path(colmap_path).name or "scene"
        ply_path = os.path.join(cfg.output_dir, f"{scene_name}.ply")

        with torch.no_grad():
            means = self._splats["means"].detach().clone()
            scales = self._splats["scales"].detach().clone()
            quats = self._splats["quats"].detach().clone()
            opacities = self._splats["opacities"].detach().clone()
            sh0 = self._splats["sh0"].detach().clone()
            shN = self._splats["shN"].detach().clone()

        export_splats(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            format="ply",
            save_to=ply_path,
        )
        logger.info(f"Saved PLY to {ply_path}")
