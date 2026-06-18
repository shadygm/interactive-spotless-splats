from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch

SemanticKind = Literal["raw", "clustered"]

def _feature_path_for_kind(
    image_path: str | Path,
    dataset_root: str | Path | None = None,
) -> tuple[Path, Path]:
    image_path = Path(image_path)
    raw = image_path.parent / f"{image_path.stem}_sdfeats.npy"
    clustered = image_path.parent / f"{image_path.stem}_sdfeats_clustered.npy"

    # Prefer the exact sidecar layout next to the image. If a dataset root is
    # provided, also allow the common `root/images/` layout.
    if dataset_root is not None:
        dataset_root = Path(dataset_root)
        root_images = dataset_root / "images"
        if root_images != image_path.parent:
            raw_root = root_images / raw.name
            clustered_root = root_images / clustered.name
            if raw.exists() or clustered.exists():
                return raw, clustered
            return raw_root, clustered_root

    return raw, clustered


def load_semantic_features(
    image_path: str | Path,
    dataset_root: str | Path | None = None,
    feature_kind: SemanticKind = "raw",
) -> tuple[Optional[torch.Tensor], Optional[Path], Optional[str]]:
    """Load raw or clustered semantic features that live next to an image."""
    raw_path, clustered_path = _feature_path_for_kind(image_path, dataset_root=dataset_root)
    candidate = raw_path if feature_kind == "raw" else clustered_path
    if candidate.exists():
        features = np.load(candidate)
        tensor = torch.from_numpy(features.astype(np.float32, copy=False))
        feature_kind = "clustered" if "clustered" in candidate.stem else "raw"
        return tensor, candidate, feature_kind
    return None, None, None


def build_semantic_feature_manifest(
    image_paths: list[str] | list[Path],
    dataset_root: str | Path | None = None,
    feature_kind: SemanticKind = "raw",
) -> tuple[list[Optional[Path]], list[Optional[Path]], Optional[Tuple[int, ...]]]:
    """Precompute feature-file paths for a dataset.

    This avoids repeating filesystem discovery in every __getitem__ call.
    """
    feature_paths: list[Optional[Path]] = []
    preview_paths: list[Optional[Path]] = []
    semantic_shape: Optional[Tuple[int, ...]] = None
    dataset_root = Path(dataset_root) if dataset_root is not None else None

    for image_path in image_paths:
        raw_path, clustered_path = _feature_path_for_kind(image_path, dataset_root=dataset_root)
        chosen = raw_path if feature_kind == "raw" else clustered_path
        if not chosen.exists():
            chosen = None
        if chosen is None:
            feature_paths.append(None)
            preview_paths.append(None)
            continue

        feature_paths.append(chosen)
        preview_paths.append(chosen.with_name(f"{chosen.stem}_preview.png"))
        if semantic_shape is None:
            try:
                semantic_shape = tuple(np.load(chosen, mmap_mode="r").shape)
            except Exception:
                try:
                    semantic_shape = tuple(np.load(chosen).shape)
                except Exception:
                    semantic_shape = None

    return feature_paths, preview_paths, semantic_shape


class FeatureMixin:
    """Mixin that loads semantic feature files next to images."""

    semantic_shape: Optional[Tuple[int, ...]] = None

    def _load_features(
        self,
        image_path: str | Path,
        dataset_root: str | Path | None = None,
        feature_kind: SemanticKind = "raw",
    ) -> Optional[torch.Tensor]:
        features, _, _ = load_semantic_features(
            image_path,
            dataset_root=dataset_root,
            feature_kind=feature_kind,
        )
        return features
