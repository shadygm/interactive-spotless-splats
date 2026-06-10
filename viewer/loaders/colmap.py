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


# ---------------------------------------------------------------------------
# Text format loaders
# ---------------------------------------------------------------------------

_CAMERA_MODEL_NAME_TO_ID = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
    "OPENCV_FISHEYE": 5,
    "FULL_OPENCV": 6,
    "FOV": 7,
    "SIMPLE_RADIAL_FISHEYE": 8,
    "RADIAL_FISHEYE": 9,
    "THIN_PRISM_FISHEYE": 10,
}


def read_cameras_text(path):
    """Read COLMAP cameras.txt file."""
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model_name = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array([float(x) for x in parts[4:]], dtype=np.float64)
            model = _CAMERA_MODEL_NAME_TO_ID.get(model_name, -1)
            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
    logger.info(f"Loaded {len(cameras)} cameras from {path}")
    return cameras


def read_images_text(path):
    """Read COLMAP images.txt file."""
    images = {}
    with open(path, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        image_id = int(parts[0])
        qvec = np.array([float(x) for x in parts[1:5]], dtype=np.float64)
        tvec = np.array([float(x) for x in parts[5:8]], dtype=np.float64)
        camera_id = int(parts[8])
        name = parts[9]
        # Read the next line (2D points, if present)
        xys = []
        point3D_ids = []
        if i < len(lines):
            next_line = lines[i].strip()
            if next_line and not next_line.startswith("#"):
                i += 1
                point_parts = next_line.split()
                # Format: (X, Y, POINT3D_ID) triplets
                for j in range(0, len(point_parts), 3):
                    if j + 2 < len(point_parts):
                        xys.append([float(point_parts[j]), float(point_parts[j + 1])])
                        point3D_ids.append(int(point_parts[j + 2]))
        images[image_id] = {
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": camera_id,
            "name": name,
            "xys": np.array(xys, dtype=np.float64) if xys else np.zeros((0, 2), dtype=np.float64),
            "point3D_ids": np.array(point3D_ids, dtype=np.int64) if point3D_ids else np.zeros(0, dtype=np.int64),
        }
    logger.info(f"Loaded {len(images)} images from {path}")
    return images


def read_points3D_text(path):
    """Read COLMAP points3D.txt file."""
    points3D = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            point3D_id = int(parts[0])
            xyz = np.array([float(x) for x in parts[1:4]], dtype=np.float64)
            rgb = np.array([int(x) for x in parts[4:7]], dtype=np.uint8)
            error = float(parts[7])
            track_len = int(parts[8])
            # Parse remaining elements as (IMAGE_ID, POINT2D_IDX) pairs
            track_parts = parts[9:]
            track = []
            for j in range(0, len(track_parts), 2):
                if j + 1 < len(track_parts):
                    track.append([int(track_parts[j]), int(track_parts[j + 1])])
            track = np.array(track, dtype=np.int64).reshape(-1, 2) if track else np.zeros((0, 2), dtype=np.int64)
            points3D[point3D_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track": track,
            }
    logger.info(f"Loaded {len(points3D)} points3D from {path}")
    return points3D
