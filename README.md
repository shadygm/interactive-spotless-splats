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
python -m viewer --ply path/to/model.ply

# Load a COLMAP dataset
python -m viewer --colmap path/to/colmap/

# Both + custom resolution
python -m viewer --ply model.ply --colmap dataset/ --width 1920 --height 1080

# Debug logging
python -m viewer --ply model.ply --log-level DEBUG
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
