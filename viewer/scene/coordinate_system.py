import numpy as np


def flip_y_axis(points: np.ndarray) -> np.ndarray:
    """Flip Y axis in-place for coordinate system conversion."""
    points[..., 1] *= -1.0
    return points


def flip_quaternion_y(qvec: np.ndarray) -> np.ndarray:
    """Transform a quaternion to match a Y-axis reflection.

    A Y-axis reflection (y -> -y) is an improper transformation.  For a
    rotation represented by quaternion [w, x, y, z], the correctly
    transformed rotation is given by [w, -x, y, -z].  This follows from
    conjugating the rotation matrix by diag(1, -1, 1).
    """
    qvec[..., 1] *= -1.0  # negate x
    qvec[..., 3] *= -1.0  # negate z
    return qvec


def qvec2rotmat(qvec):
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    w, x, y, z = qvec
    R = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
    ], dtype=np.float64)
    return R
