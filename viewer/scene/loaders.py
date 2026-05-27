import os
from typing import Protocol, runtime_checkable

import numpy as np
from loguru import logger

from viewer.loaders.colmap import read_cameras_binary, read_images_binary, read_points3D_binary
from viewer.loaders.ply import load_ply_file
from viewer.scene.coordinate_system import flip_y_axis, flip_quaternion_y
from viewer.scene.gaussian_data import build_gaussian_tensors


@runtime_checkable
class SceneLoader(Protocol):
    def load(self, path: str) -> dict:
        """Load scene data from path and return a dict with keys like 'images', 'cameras', 'points3D', 'gaussians'."""
        ...


class ColmapLoader:
    def load(self, path: str) -> dict:
        """Load a COLMAP dataset directory.

        Returns a dict with keys: cameras, images, points3D, gaussians.
        """
        path = os.path.abspath(path)
        sparse_dir = os.path.join(path, "sparse", "0")
        cameras_path = os.path.join(sparse_dir, "cameras.bin")
        images_path = os.path.join(sparse_dir, "images.bin")
        points3D_path = os.path.join(sparse_dir, "points3D.bin")

        cameras = {}
        images = {}
        points3D = {}

        if os.path.exists(cameras_path):
            cameras = read_cameras_binary(cameras_path)
        if os.path.exists(images_path):
            images = read_images_binary(images_path)
        if os.path.exists(points3D_path):
            points3D = read_points3D_binary(points3D_path)

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


LOADERS: dict[str, SceneLoader] = {
    "colmap": ColmapLoader(),
    "ply": PlyLoader(),
}


def load_scene(path: str, format: str | None = None) -> dict:
    """Auto-detect format and load scene data.

    Args:
        path: Path to the scene file or directory.
        format: Optional format hint ("colmap" or "ply"). If None, auto-detect.

    Returns:
        A dict with keys: cameras, images, points3D, gaussians.
    """
    path = os.path.abspath(path)

    if format is None:
        if os.path.isdir(path):
            format = "colmap"
        elif path.lower().endswith(".ply"):
            format = "ply"
        else:
            raise ValueError(f"Cannot auto-detect scene format for path: {path}")

    loader = LOADERS.get(format)
    if loader is None:
        raise ValueError(f"Unknown scene format: {format}")

    return loader.load(path)
