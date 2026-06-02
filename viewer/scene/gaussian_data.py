import numpy as np
import torch


def build_axis_gaussians(device: str) -> dict:
    """Build 4 canonical axis-aligned Gaussians (origin + X/Y/Z directions).

    Returns a dict with keys: means, quats, scales, opacities, colors, sh_degree.
    """
    means = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.float32)

    quats = np.array([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
    ], dtype=np.float32)

    scales = np.array([
        [0.1, 0.1, 0.1],
        [0.5, 0.1, 0.1],
        [0.1, 0.5, 0.1],
        [0.1, 0.1, 0.5],
    ], dtype=np.float32)

    opacities = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    colors = np.array([
        [1.0, 1.0, 1.0],  # origin - white
        [1.0, 0.0, 0.0],  # X - red
        [0.0, 1.0, 0.0],  # Y - green
        [0.0, 0.0, 1.0],  # Z - blue
    ], dtype=np.float32)[:, None, :]  # (4, 1, 3)

    return {
        "means": torch.from_numpy(means).to(device),
        "quats": torch.from_numpy(quats).to(device),
        "scales": torch.from_numpy(scales).to(device),
        "opacities": torch.from_numpy(opacities).to(device),
        "colors": torch.from_numpy(colors).to(device),
        "sh_degree": 0,
    }


def _determine_sh_degree(data: dict) -> int:
    """Determine SH degree from f_rest_* properties."""
    rest_keys = [k for k in data.keys() if k.startswith("f_rest_")]
    if not rest_keys:
        return 0
    num_rest = len(rest_keys)
    # SH bands: (sh_degree+1)^2 - 1 = num_rest / 3
    # So (sh_degree+1)^2 = num_rest / 3 + 1
    total_bases = num_rest / 3.0 + 1
    sh_degree = int(np.round(np.sqrt(total_bases))) - 1
    sh_degree = max(0, min(3, sh_degree))  # clamp to valid levels 0..3
    return sh_degree


def _build_sh_coeffs(data: dict, colors: np.ndarray, sh_degree: int) -> np.ndarray:
    """Build SH coefficients array [N, K, 3] from PLY data."""
    N = colors.shape[0]
    K = (sh_degree + 1) ** 2
    sh_coeffs = np.zeros((N, K, 3), dtype=np.float32)
    sh_coeffs[:, 0, :] = colors
    for k in range(1, K):
        for c in range(3):
            key = f"f_rest_{(k - 1) * 3 + c}"
            if key in data:
                sh_coeffs[:, k, c] = data[key]
    return sh_coeffs


def build_gaussian_tensors(data: dict, device: str) -> dict:
    """Parse PLY data dict and build gaussian tensors on the given device.

    Returns a dict with keys: means, quats, scales, opacities, colors, sh_degree.
    """
    N = len(data["x"])

    means = np.stack([data["x"], data["y"], data["z"]], axis=-1).astype(np.float32)

    # Quaternions: rot_0,1,2,3 -> w,x,y,z (PLY convention is usually w,x,y,z)
    quats = np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=-1).astype(np.float32)
    quats_norm = np.linalg.norm(quats, axis=-1, keepdims=True)
    quats = quats / (quats_norm + 1e-8)

    scales = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=-1)
    scales = np.exp(scales).astype(np.float32)

    opacities = data["opacity"].astype(np.float32)
    opacities = 1.0 / (1.0 + np.exp(-opacities))

    # Colors: try SH DC first, fallback to RGB
    if "f_dc_0" in data and "f_dc_1" in data and "f_dc_2" in data:
        colors = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=-1).astype(np.float32)
        sh_degree = _determine_sh_degree(data)
        if sh_degree > 0:
            colors = _build_sh_coeffs(data, colors, sh_degree)
        else:
            # No rest coefficients: DC only, reshape to (N, 1, 3)
            colors = colors[:, None, :]
            sh_degree = 0
    elif "red" in data and "green" in data and "blue" in data:
        colors = np.stack([data["red"], data["green"], data["blue"]], axis=-1).astype(np.float32) / 255.0
        sh_degree = 0
    else:
        colors = np.ones((N, 3), dtype=np.float32) * 0.5
        sh_degree = 0

    return {
        "means": torch.from_numpy(means).to(device),
        "quats": torch.from_numpy(quats).to(device),
        "scales": torch.from_numpy(scales).to(device),
        "opacities": torch.from_numpy(opacities).to(device),
        "colors": torch.from_numpy(colors).to(device),
        "sh_degree": sh_degree,
    }
