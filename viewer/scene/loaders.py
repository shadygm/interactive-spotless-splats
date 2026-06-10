import json
import os
from typing import Protocol, runtime_checkable

import numpy as np
from loguru import logger

from viewer.loaders.colmap import (
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    read_cameras_text,
    read_images_text,
    read_points3D_text,
)
from viewer.loaders.ply import load_ply_file
from viewer.scene.coordinate_system import flip_y_axis, flip_quaternion_y
from viewer.scene.gaussian_data import build_gaussian_tensors


@runtime_checkable
class SceneLoader(Protocol):
    def load(self, path: str) -> dict:
        """Load scene data from path and return a dict with keys like 'images', 'cameras', 'points3D', 'gaussians'."""
        ...


def _load_point_cloud_records(path: str) -> dict[int, dict]:
    """Load a point cloud from the common on-the-go / COLMAP locations.

    Prefer the dataset-root `PointCloud.ply` used by on-the-go exports, then
    fall back to COLMAP-style sparse point clouds.
    """
    candidates = [
        os.path.join(path, "PointCloud.ply"),
        os.path.join(path, "pointcloud.ply"),
        os.path.join(path, "sparse", "0", "points3D.ply"),
        os.path.join(path, "sparse", "points3D.ply"),
    ]

    for ply_path in candidates:
        if not os.path.exists(ply_path):
            continue
        data = load_ply_file(ply_path)
        required = {"x", "y", "z", "red", "green", "blue"}
        if not required.issubset(data.keys()):
            continue

        xyz = np.stack([data["x"], data["y"], data["z"]], axis=-1).astype(np.float32)
        rgb = np.stack([data["red"], data["green"], data["blue"]], axis=-1).astype(np.uint8)
        points3D = {}
        for i in range(len(xyz)):
            points3D[i] = {"xyz": xyz[i], "rgb": rgb[i]}
        return points3D

    return {}


class ColmapLoader:
    def load(self, path: str) -> dict:
        """Load a COLMAP dataset directory.

        Returns a dict with keys: cameras, images, points3D, gaussians.
        """
        path = os.path.abspath(path)
        sparse_dir = os.path.join(path, "sparse", "0")
        if not os.path.exists(sparse_dir):
            sparse_dir = os.path.join(path, "sparse")

        # Try binary first, then text
        cameras_bin = os.path.join(sparse_dir, "cameras.bin")
        cameras_txt = os.path.join(sparse_dir, "cameras.txt")
        images_bin = os.path.join(sparse_dir, "images.bin")
        images_txt = os.path.join(sparse_dir, "images.txt")
        points3D_bin = os.path.join(sparse_dir, "points3D.bin")
        points3D_txt = os.path.join(sparse_dir, "points3D.txt")

        cameras = {}
        images = {}
        points3D = {}

        if os.path.exists(cameras_bin):
            cameras = read_cameras_binary(cameras_bin)
        elif os.path.exists(cameras_txt):
            cameras = read_cameras_text(cameras_txt)

        if os.path.exists(images_bin):
            images = read_images_binary(images_bin)
        elif os.path.exists(images_txt):
            images = read_images_text(images_txt)

        if os.path.exists(points3D_bin):
            points3D = read_points3D_binary(points3D_bin)
        elif os.path.exists(points3D_txt):
            points3D = read_points3D_text(points3D_txt)

        if not (cameras or images or points3D):
            raise ValueError(f"No COLMAP data found in {sparse_dir}")

        # COLMAP cameras look down positive Z (forward_sign = 1.0)
        for cam in cameras.values():
            cam.setdefault("forward_sign", 1.0)

        # Flip Y axis in memory (common coordinate system conversion).
        # Quaternions must be transformed with [w, x, y, z] -> [w, -x, y, -z]
        # to keep the rotation consistent with the reflected coordinate system.
        if images:
            for img in images.values():
                img["tvec"] = img["tvec"].copy()
                flip_y_axis(img["tvec"])
                img["qvec"] = img["qvec"].copy()
                flip_quaternion_y(img["qvec"])
        if points3D:
            for p in points3D.values():
                p["xyz"] = p["xyz"].copy()
                flip_y_axis(p["xyz"])

        return {
            "cameras": cameras,
            "images": images,
            "points3D": points3D,
            "gaussians": None,
        }


class PlyLoader:
    def load(self, path: str) -> dict:
        """Load a 3DGS PLY file.

        Returns a dict with keys: cameras, images, points3D, gaussians.
        """
        path = os.path.abspath(path)
        data = load_ply_file(path)

        # Flip Y so PLY matches our COLMAP Y-up convention
        means = np.stack([data["x"], data["y"], data["z"]], axis=-1).astype(np.float32)
        flip_y_axis(means)

        # Quaternions: rot_0,1,2,3 -> w,x,y,z (PLY convention is usually w,x,y,z)
        # Transform with [w, x, y, z] -> [w, -x, y, -z] to match the Y flip.
        quats = np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=-1).astype(np.float32)
        flip_quaternion_y(quats)

        # Update data dict with flipped arrays so build_gaussian_tensors uses them
        data["x"] = means[:, 0]
        data["y"] = means[:, 1]
        data["z"] = means[:, 2]
        data["rot_0"] = quats[:, 0]
        data["rot_1"] = quats[:, 1]
        data["rot_2"] = quats[:, 2]
        data["rot_3"] = quats[:, 3]

        # Determine device from availability (same default as SceneState)
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gaussians = build_gaussian_tensors(data, device)

        return {
            "cameras": {},
            "images": {},
            "points3D": {},
            "gaussians": gaussians,
        }


class TransformsJsonLoader:
    def load(self, path: str) -> dict:
        """Load a transforms.json dataset (e.g. nerfstudio / instant-ngp format).

        Returns a dict with keys: cameras, images, points3D, gaussians.
        """
        path = os.path.abspath(path)
        transforms_path = os.path.join(path, "transforms.json")
        with open(transforms_path, "r") as f:
            meta = json.load(f)

        # Global intrinsics
        w = int(meta.get("w", 1920))
        h = int(meta.get("h", 1080))
        fl_x = float(meta.get("fl_x", meta.get("camera_angle_x", 0.0)))
        fl_y = float(meta.get("fl_y", meta.get("camera_angle_y", 0.0)))
        cx = float(meta.get("cx", w / 2.0))
        cy = float(meta.get("cy", h / 2.0))
        k1 = float(meta.get("k1", 0.0))
        is_fisheye = bool(meta.get("is_fisheye", False))

        # Determine camera model based on distortion
        if abs(k1) > 1e-8 and not is_fisheye:
            # SIMPLE_RADIAL: [f, cx, cy, k]
            # Use average of fl_x and fl_y as f for SIMPLE_RADIAL
            f_avg = (fl_x + fl_y) / 2.0
            cameras = {
                0: {
                    "model": 2,  # SIMPLE_RADIAL
                    "width": w,
                    "height": h,
                    "params": np.array([f_avg, cx, cy, k1], dtype=np.float64),
                    "forward_sign": -1.0,  # OpenGL: camera looks down negative Z
                }
            }
        else:
            # Default camera (PINHOLE since we have 4 distinct params)
            cameras = {
                0: {
                    "model": 1,  # PINHOLE
                    "width": w,
                    "height": h,
                    "params": np.array([fl_x, fl_y, cx, cy], dtype=np.float64),
                    "forward_sign": -1.0,  # OpenGL: camera looks down negative Z
                }
            }

        images = {}
        for i, frame in enumerate(meta.get("frames", [])):
            file_path = frame.get("file_path", "")
            # Resolve relative to dataset directory
            if not os.path.isabs(file_path):
                file_path = os.path.normpath(os.path.join(path, file_path))
            name = os.path.basename(file_path)

            transform_matrix = np.array(frame["transform_matrix"], dtype=np.float32)
            # transform_matrix is camera-to-world
            camtoworld = transform_matrix

            # Convert to world-to-camera for COLMAP-like format
            w2c = np.linalg.inv(camtoworld)
            R = w2c[:3, :3]
            t = w2c[:3, 3]

            # Rotation matrix to quaternion [w, x, y, z]
            qvec = _rotation_matrix_to_quaternion(R)

            images[i] = {
                "name": name,
                "camera_id": 0,
                "qvec": qvec,
                "tvec": t.astype(np.float64),
            }

        # Load points3D from the dataset-root point cloud when available.
        points3D = _load_point_cloud_records(path)

        # transforms.json / on-the-go exports are already in the viewer's world
        # coordinate system, so the point cloud should be used as-is.

        return {
            "cameras": cameras,
            "images": images,
            "points3D": points3D,
            "gaussians": None,
        }


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to quaternion [w, x, y, z]."""
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

    return np.array([w, x, y, z], dtype=np.float64)


LOADERS: dict[str, SceneLoader] = {
    "colmap": ColmapLoader(),
    "ply": PlyLoader(),
    "transforms": TransformsJsonLoader(),
}


def load_scene(path: str, format: str | None = None) -> dict:
    """Auto-detect format and load scene data.

    Args:
        path: Path to the scene file or directory.
        format: Optional format hint ("colmap", "ply", or "transforms"). If None, auto-detect.

    Returns:
        A dict with keys: cameras, images, points3D, gaussians.
    """
    path = os.path.abspath(path)

    if format is None:
        if os.path.isdir(path):
            # Check for transforms.json first (common nerfstudio format)
            if os.path.exists(os.path.join(path, "transforms.json")):
                format = "transforms"
            else:
                format = "colmap"
        elif path.lower().endswith(".ply"):
            format = "ply"
        else:
            raise ValueError(f"Cannot auto-detect scene format for path: {path}")

    loader = LOADERS.get(format)
    if loader is None:
        raise ValueError(f"Unknown scene format: {format}")

    return loader.load(path)
