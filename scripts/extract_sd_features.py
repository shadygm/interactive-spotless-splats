#!/usr/bin/env python3
"""CLI for extracting Stable Diffusion features from a directory of images.

Usage:
    # Extract normal SD features (for SLS-mlp)
    uv run python scripts/extract_sd_features.py /path/to/images --img-size 800

    # Extract clustered SD features (for SLS-agg)
    uv run python scripts/extract_sd_features.py /path/to/images --img-size 800 --cluster --n-clusters 100

    # Extract both (run twice)
    uv run python scripts/extract_sd_features.py /path/to/images --img-size 800
    uv run python scripts/extract_sd_features.py /path/to/images --img-size 800 --cluster

Output:
    For each image <name>.<ext>, writes:
    - <name>_sdfeats.npy          (raw features, shape [C, H, W])
    - <name>_sdfeats_preview.png  (PCA preview for visualization)

    With --cluster:
    - <name>_sdfeats_clustered.npy          (cluster masks, shape [n_clusters, H, W])
    - <name>_sdfeats_clustered_preview.png  (cluster color preview)
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.sd_features import DirectoryExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Extract Stable Diffusion features from images for SpotLessSplats."
    )
    parser.add_argument(
        "image_dir",
        type=str,
        help="Directory containing images to process.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=800,
        help="Resize images to this size (square). If None, keep original resolution.",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Cluster features using AgglomerativeClustering (for SLS-agg).",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=100,
        help="Number of clusters when using --cluster. Default: 100",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=261,
        help="Diffusion timestep. Default: 261",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=4,
        help="Number of ensemble samples. Default: 1 (use 4 for higher quality but more memory)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip saving preview PNGs.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip images that already have feature files.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Re-process images even if feature files exist.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu). Default: cuda",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Model dtype (float16 or float32). Default: float16 (half memory)",
    )
    parser.add_argument(
        "--up-ft-index",
        type=int,
        nargs="+",
        default=[1],
        help="UNet up-block indices to extract. Default: [1]",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
        help="Log level. Default: INFO",
    )

    args = parser.parse_args()

    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level=args.log_level, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

    import torch
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    logger.info(f"Extracting SD features from: {args.image_dir}")
    logger.info(f"  Mode: {'clustered (SLS-agg)' if args.cluster else 'raw (SLS-mlp)'}")
    logger.info(f"  Image size: {args.img_size or 'original'}")
    logger.info(f"  Timestep: {args.t}")
    logger.info(f"  Ensemble size: {args.ensemble_size}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Dtype: {args.dtype}")
    logger.info(f"  Up-block indices: {args.up_ft_index}")
    if args.cluster:
        logger.info(f"  Number of clusters: {args.n_clusters}")
    logger.info(f"  Skip existing: {args.skip_existing}")
    logger.info(f"  Save previews: {not args.no_preview}")
    logger.info("")

    try:
        extractor = DirectoryExtractor(device=args.device, dtype=dtype)
        output_paths = extractor.extract_directory(
            image_dir=args.image_dir,
            img_size=args.img_size,
            cluster=args.cluster,
            n_clusters=args.n_clusters,
            t=args.t,
            up_ft_index=args.up_ft_index,
            ensemble_size=args.ensemble_size,
            save_preview=not args.no_preview,
            skip_existing=args.skip_existing,
        )

        logger.info(f"\nDone! Extracted {len(output_paths)} features.")
        if output_paths:
            logger.info(f"  Example output: {output_paths[0]}")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        logger.error("If the model requires authentication, run: huggingface-cli login")
        raise


if __name__ == "__main__":
    main()
