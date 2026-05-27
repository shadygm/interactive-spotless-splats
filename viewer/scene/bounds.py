import numpy as np

from viewer.scene.coordinate_system import qvec2rotmat


def compute_scene_bounds(images, points3D):
    """Return axis-aligned bounding box (min, max) of camera positions and points.

    Args:
        images: dict of COLMAP image records (each with 'qvec' and 'tvec').
        points3D: dict of COLMAP point records (each with 'xyz').

    Returns:
        (bmin, bmax) as float32 arrays.
    """
    pts = []
    if images:
        for img in images.values():
            qvec = img["qvec"]
            tvec = img["tvec"]
            R = qvec2rotmat(qvec)
            C = -R.T @ tvec
            pts.append(C)
    if points3D:
        for p in points3D.values():
            pts.append(p["xyz"])
    if not pts:
        return np.array([-1, -1, -1], dtype=np.float32), np.array([1, 1, 1], dtype=np.float32)
    pts = np.stack(pts, axis=0)
    return pts.min(axis=0).astype(np.float32), pts.max(axis=0).astype(np.float32)
