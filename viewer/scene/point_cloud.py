import numpy as np


def extract_point_cloud(points3D):
    """Return (xyz, rgb) numpy arrays from colmap_points3D.

    Args:
        points3D: dict of COLMAP point records (each with 'xyz' and 'rgb').

    Returns:
        (xyz, rgb) as float32 arrays, or (None, None) if empty.
    """
    if not points3D:
        return None, None
    xyz = np.stack([p["xyz"] for p in points3D.values()], axis=0).astype(np.float32)
    rgb = np.stack([p["rgb"] for p in points3D.values()], axis=0).astype(np.float32) / 255.0
    return xyz, rgb
