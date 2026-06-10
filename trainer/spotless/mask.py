from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(x.shape)}")
    return x


def robust_mask(error_per_pixel: torch.Tensor, loss_threshold: float) -> torch.Tensor:
    """Return an inlier mask using the reference 3x3 neighborhood smoothing."""
    error_per_pixel = _as_bchw(error_per_pixel)
    if error_per_pixel.shape[1] == 1 and error_per_pixel.shape[-1] != 1:
        error_per_pixel = error_per_pixel.permute(0, 2, 3, 1)
    if error_per_pixel.shape[-1] != 1:
        error_per_pixel = error_per_pixel.mean(dim=-1, keepdim=True)

    is_inlier_pixel = (error_per_pixel < loss_threshold).float()
    is_inlier_pixel = is_inlier_pixel.permute(0, 3, 1, 2)

    window = torch.ones((1, 1, 3, 3), device=error_per_pixel.device, dtype=error_per_pixel.dtype) / 9.0
    has_inlier_neighbors = F.conv2d(is_inlier_pixel, window, padding=1)
    has_inlier_neighbors = (has_inlier_neighbors > 0.5).float()
    is_inlier_pixel = ((has_inlier_neighbors + is_inlier_pixel) > 1e-3).float()
    return is_inlier_pixel.permute(0, 2, 3, 1)


def robust_cluster_mask(inlier_sf: torch.Tensor, semantics: torch.Tensor) -> torch.Tensor:
    """Vote per semantic cluster and keep clusters with >50% inlier pixels."""
    inlier_sf = _as_bchw(inlier_sf)
    semantics = _as_bchw(semantics)

    if inlier_sf.shape[-1] != 1:
        inlier_sf = inlier_sf.mean(dim=-1, keepdim=True)
    if semantics.shape[1] == 1 and semantics.shape[-1] != 1:
        semantics = semantics.permute(0, 3, 1, 2)

    inlier_sf = inlier_sf.permute(0, 3, 1, 2)
    cluster_size = semantics.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-8)
    inlier_cluster_size = (inlier_sf * semantics).sum(dim=(-1, -2), keepdim=True)
    cluster_inlier_percentage = inlier_cluster_size / cluster_size
    is_inlier_cluster = (cluster_inlier_percentage > 0.5).float()
    pred_mask = (semantics * is_inlier_cluster).sum(dim=1, keepdim=True).clamp_max(1.0)
    return pred_mask.permute(0, 2, 3, 1)

