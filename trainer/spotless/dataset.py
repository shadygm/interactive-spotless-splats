from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch


def _candidate_feature_paths(
    image_path: str | Path,
    dataset_root: str | Path | None = None,
    prefer_clustered: bool = False,
) -> list[Path]:
    image_path = Path(image_path)
    candidates: list[Path] = []

    def add_dir(base_dir: Path) -> None:
        raw = base_dir / f"{image_path.stem}_sdfeats.npy"
        clustered = base_dir / f"{image_path.stem}_sdfeats_clustered.npy"
        if prefer_clustered:
            candidates.extend([clustered, raw])
        else:
            candidates.extend([raw, clustered])

    # Most common layout: feature files live next to the image.
    add_dir(image_path.parent)

    if dataset_root is not None:
        dataset_root = Path(dataset_root)
        add_dir(dataset_root / "images")

    # If the image path itself is not the one the dataset loader sees
    # (e.g. relative `./images/...` vs absolute path), also scan the nearest
    # `images/` directory in the ancestry.
    for parent in image_path.parents:
        if parent.name == "images":
            add_dir(parent)
            break

    # Also allow discovery from the dataset root if the caller handed us an
    # image path under `root/images/...`.
    if image_path.parent.name == "images":
        add_dir(image_path.parent)

    # Deduplicate while preserving order.
    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return ordered


def load_semantic_features(
    image_path: str | Path,
    dataset_root: str | Path | None = None,
    prefer_clustered: bool = False,
) -> tuple[Optional[torch.Tensor], Optional[Path], Optional[str]]:
    """Load raw or clustered semantic features that live next to an image."""
    for candidate in _candidate_feature_paths(
        image_path,
        dataset_root=dataset_root,
        prefer_clustered=prefer_clustered,
    ):
        if not candidate.exists():
            # Fall back to stem-based globbing when the exact filename is not
            # what the extractor used.
            matches = sorted(candidate.parent.glob(f"{Path(image_path).stem}_sdfeats*.npy"))
            if prefer_clustered:
                matches = sorted(matches, key=lambda p: (0 if "clustered" in p.stem else 1, p.name))
            if not matches:
                continue
            candidate = matches[0]

        features = np.load(candidate)
        tensor = torch.from_numpy(features.astype(np.float32, copy=False))
        feature_kind = "clustered" if "clustered" in candidate.stem else "raw"
        return tensor, candidate, feature_kind
    return None, None, None


def build_semantic_feature_manifest(
    image_paths: list[str] | list[Path],
    dataset_root: str | Path | None = None,
    prefer_clustered: bool = False,
) -> tuple[list[Optional[Path]], list[Optional[Path]], Optional[Tuple[int, ...]]]:
    """Precompute feature-file paths for a dataset.

    This avoids repeating filesystem discovery in every __getitem__ call.
    """
    feature_paths: list[Optional[Path]] = []
    preview_paths: list[Optional[Path]] = []
    semantic_shape: Optional[Tuple[int, ...]] = None
    dataset_root = Path(dataset_root) if dataset_root is not None else None

    for image_path in image_paths:
        candidates = _candidate_feature_paths(
            image_path,
            dataset_root=dataset_root,
            prefer_clustered=prefer_clustered,
        )
        chosen: Optional[Path] = None
        for candidate in candidates:
            if candidate.exists():
                chosen = candidate
                break
        if chosen is None:
            feature_paths.append(None)
            preview_paths.append(None)
            continue

        feature_paths.append(chosen)
        if "clustered" in chosen.stem:
            preview_paths.append(chosen.with_name(f"{chosen.stem}_preview.png"))
        else:
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

    def _load_features(self, image_path: str | Path, dataset_root: str | Path | None = None) -> Optional[torch.Tensor]:
        features, _, _ = load_semantic_features(image_path, dataset_root=dataset_root)
        return features
