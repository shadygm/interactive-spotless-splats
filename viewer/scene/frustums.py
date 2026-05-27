import numpy as np

from viewer.scene.coordinate_system import qvec2rotmat


def build_camera_frustums(images, cameras, render_settings=None, scene_scale=None):
    """Return list of (position, corners) for each COLMAP image.

    Args:
        images: dict of COLMAP image records.
        cameras: dict of COLMAP camera records.
        render_settings: optional object with a ``frustum_size`` attribute.
        scene_scale: optional pre-computed scene diagonal. If None, computed from images.

    Returns:
        List of (position, corners_world) tuples.
    """
    if not images or not cameras:
        return []

    # Compute scene scale for consistent frustum size
    if scene_scale is None:
        from viewer.scene.bounds import compute_scene_bounds

        bmin, bmax = compute_scene_bounds(images, {})
        scene_scale = float(np.linalg.norm(bmax - bmin))
    if scene_scale < 1e-6:
        scene_scale = 1.0
    # Frustum depth = fraction of scene diagonal from render settings
    size = render_settings.frustum_size if render_settings else 0.01
    frustum_depth = scene_scale * size

    frustums = []
    for img in images.values():
        cam = cameras.get(img["camera_id"])
        if cam is None:
            continue

        qvec = img["qvec"]
        tvec = img["tvec"]
        R = qvec2rotmat(qvec)
        # Camera center C = -R^T * t
        C = -R.T @ tvec

        # Build intrinsics from camera params
        model = cam["model"]
        width = cam["width"]
        height = cam["height"]
        params = cam["params"]

        if model == 0:  # SIMPLE_PINHOLE
            fx, cx, cy = params[0], params[1], params[2]
            fy = fx
        elif model == 1:  # PINHOLE
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        else:
            # Default to simple pinhole with first param
            fx = params[0] if len(params) > 0 else width
            fy = fx
            cx = width / 2.0
            cy = height / 2.0

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ], dtype=np.float64)

        # Image corners in pixel coords
        corners_px = np.array([
            [0, 0, 1],
            [width, 0, 1],
            [width, height, 1],
            [0, height, 1],
        ], dtype=np.float64).T

        # Unproject to camera space (z=1 plane)
        K_inv = np.linalg.inv(K)
        corners_cam = K_inv @ corners_px
        # Scale to a fixed visualization depth relative to scene size
        corners_cam = corners_cam * frustum_depth
        corners_cam = np.vstack([corners_cam, np.ones((1, 4))])

        # Transform to world space
        # World-to-cam: [R | t], so cam-to-world: [R^T | -R^T t]
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = C
        corners_world = c2w @ corners_cam
        corners_world = corners_world[:3, :].T

        frustums.append((C.astype(np.float32), corners_world.astype(np.float32)))
    return frustums
