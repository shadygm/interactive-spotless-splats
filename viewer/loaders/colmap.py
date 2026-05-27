import struct
import numpy as np
from loguru import logger


# Camera model param counts (from COLMAP)
_CAMERA_MODEL_PARAMS = {
    0: 3,   # SIMPLE_PINHOLE: f, cx, cy
    1: 4,   # PINHOLE: fx, fy, cx, cy
    2: 4,   # SIMPLE_RADIAL: f, cx, cy, k
    3: 5,   # RADIAL: f, cx, cy, k1, k2
    4: 8,   # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
    5: 8,   # OPENCV_FISHEYE: fx, fy, cx, cy, k1, k2, k3, k4
    6: 12,  # FULL_OPENCV: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
    7: 5,   # FOV: fx, fy, cx, cy, omega
    8: 4,   # SIMPLE_RADIAL_FISHEYE: f, cx, cy, k
    9: 5,   # RADIAL_FISHEYE: f, cx, cy, k1, k2
    10: 12, # THIN_PRISM_FISHEYE: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
}


def _read_string(f):
    """Read a null-terminated string from a binary file."""
    name = b""
    while True:
        ch = f.read(1)
        if ch == b"\x00" or ch == b"":
            break
        name += ch
    return name.decode("utf-8", errors="replace")


def read_cameras_binary(path):
    """Read COLMAP cameras.bin file."""
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = int(struct.unpack("<Q", f.read(8))[0])
        logger.debug(f"Loading {num_cameras} cameras from {path}")
        for i in range(num_cameras):
            camera_id = int(struct.unpack("<I", f.read(4))[0])
            model = int(struct.unpack("<i", f.read(4))[0])  # int32, not uint32
            width = int(struct.unpack("<Q", f.read(8))[0])
            height = int(struct.unpack("<Q", f.read(8))[0])
            
            # Determine number of params from camera model
            num_params = _CAMERA_MODEL_PARAMS.get(model)
            if num_params is None:
                logger.warning(f"Unknown camera model {model} for camera {camera_id}, trying to read num_params field")
                # Fallback: try reading explicit num_params (newer COLMAP format)
                num_params = int(struct.unpack("<Q", f.read(8))[0])
            
            params = np.frombuffer(f.read(num_params * 8), dtype=np.float64)
            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
            logger.debug(f"Camera {camera_id}: model={model}, {width}x{height}, {num_params} params")
    logger.info(f"Loaded {len(cameras)} cameras from {path}")
    return cameras


def read_images_binary(path):
    """Read COLMAP images.bin file."""
    images = {}
    with open(path, "rb") as f:
        num_images = int(struct.unpack("<Q", f.read(8))[0])
        logger.debug(f"Loading {num_images} images from {path}")
        for i in range(num_images):
            image_id = int(struct.unpack("<I", f.read(4))[0])
            qvec = np.frombuffer(f.read(4 * 8), dtype=np.float64)
            tvec = np.frombuffer(f.read(3 * 8), dtype=np.float64)
            camera_id = int(struct.unpack("<I", f.read(4))[0])
            name = _read_string(f)
            num_points2D = int(struct.unpack("<Q", f.read(8))[0])
            xys = np.frombuffer(f.read(num_points2D * 2 * 8), dtype=np.float64).reshape(-1, 2)
            point3D_ids = np.frombuffer(f.read(num_points2D * 8), dtype=np.int64)
            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
                "xys": xys,
                "point3D_ids": point3D_ids,
            }
            logger.debug(f"Image {image_id}: {name}, camera={camera_id}, {num_points2D} points2D")
    logger.info(f"Loaded {len(images)} images from {path}")
    return images


def read_points3D_binary(path):
    """Read COLMAP points3D.bin file."""
    points3D = {}
    with open(path, "rb") as f:
        num_points = int(struct.unpack("<Q", f.read(8))[0])
        logger.debug(f"Loading {num_points} points3D from {path}")
        for i in range(num_points):
            point3D_id = int(struct.unpack("<Q", f.read(8))[0])
            xyz = np.frombuffer(f.read(3 * 8), dtype=np.float64)
            rgb = np.frombuffer(f.read(3), dtype=np.uint8)
            error = struct.unpack("<d", f.read(8))[0]
            track_length = int(struct.unpack("<Q", f.read(8))[0])
            track = np.frombuffer(f.read(track_length * 8), dtype=np.uint32).reshape(-1, 2)
            points3D[point3D_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track": track,
            }
    logger.info(f"Loaded {len(points3D)} points3D from {path}")
    return points3D
