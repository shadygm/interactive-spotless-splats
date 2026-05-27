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

# Load a COLMAP dataset
uv run python -m viewer --colmap path/to/colmap/

# Example: load a 360_v2 bonsai scene
uv run python -m viewer --colmap ../data/360_v2/bonsai/

# Both + custom resolution
uv run python -m viewer --ply model.ply --colmap dataset/ --width 1920 --height 1080

# Debug logging
uv run python -m viewer --ply model.ply --log-level DEBUG

# See all options
uv run python -m viewer --help
```

```
usage: __main__.py [-h] [--colmap COLMAP] [--ply PLY] [--width WIDTH] [--height HEIGHT] [--log-level {TRACE,DEBUG,INFO,SUCCESS,WARNING,ERROR,CRITICAL}] [--log-file LOG_FILE]

options:
  -h, --help            show this help message and exit
  --colmap COLMAP
  --ply PLY
  --width WIDTH
  --height HEIGHT
  --log-level {TRACE,DEBUG,INFO,SUCCESS,WARNING,ERROR,CRITICAL}
                        Log level for stderr (and file if --log-file is set). Default: INFO
  --log-file LOG_FILE   Optional file path to also write logs to
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
