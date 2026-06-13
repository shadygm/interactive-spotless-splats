## Requirements

- Python >= 3.10
- CUDA-capable GPU
- Linux (Wayland/X11), macOS, or Windows

## Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## Usage

```bash
# Load a 3DGS PLY file
uv run python -m viewer --ply path/to/model.ply

# Load a dataset (auto-detects COLMAP or transforms.json format)
uv run python -m viewer --dataset path/to/dataset/

# Run headless training and save outputs to ./output
uv run python -m viewer --dataset path/to/dataset/ --headless --output ./output

# Customize the viewer window size
uv run python -m viewer --ply model.ply --dataset dataset/ --width 1920 --height 1080

# Enable debug logging
uv run python -m viewer --ply model.ply --log-level DEBUG

# Extract Stable Diffusion features from a directory of images
uv run python scripts/extract_sd_features.py /path/to/images --img-size 800

# Extract clustered Stable Diffusion features
uv run python scripts/extract_sd_features.py /path/to/images --img-size 800 --cluster --n-clusters 100

# See all viewer options
uv run python -m viewer --help

# See all feature-extraction options
uv run python scripts/extract_sd_features.py --help
```

```
usage: __main__.py [-h] [--dataset DATASET] [--ply PLY] [--width WIDTH] [--height HEIGHT] [--log-level {TRACE,DEBUG,INFO,SUCCESS,WARNING,ERROR,CRITICAL}] [--log-file LOG_FILE] [--headless] [--output OUTPUT]

options:
  -h, --help            show this help message and exit
  --dataset DATASET     Path to a dataset directory. Auto-detects COLMAP or transforms.json format.
  --ply PLY
  --width WIDTH
  --height HEIGHT
  --log-level {TRACE,DEBUG,INFO,SUCCESS,WARNING,ERROR,CRITICAL}
                        Log level for stderr (and file if --log-file is set). Default: INFO
  --log-file LOG_FILE   Optional file path to also write logs to
  --headless            Run training in headless mode (requires --dataset). No GUI is shown.
  --output OUTPUT       Output directory for saved PLY files in headless mode. Default: ./output
```

## Controls

| Input | Action |
|-------|--------|
| Left drag | Rotate camera (orbit) / Look around (FPS) |
| Right drag | Pan camera (orbit) |
| Scroll | Zoom (orbit) / Move speed (FPS) |
| WASD | Move camera |
| E / Q | Move up / down |
| R + scroll | Roll camera |
| ESC | Release cursor (FPS) / Quit |
