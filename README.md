# Interactive Spotless Splats

Interactive viewer and trainer for NeRF On-the-go / SpotLessSplats scenes.

## Requirements

- Python 3.10 or newer
- CUDA-capable GPU recommended for training and splat rendering
- Linux, macOS, or Windows

## Setup

```bash
uv sync
```

## Data

Download the NeRF On-the-go dataset from:

https://rwn17.github.io/nerf-on-the-go/

Open the project page, click the `Data` link, and download the dataset archive (`on-the-go.zip`). Extract it somewhere on disk and point the viewer/trainer at the extracted scene directory.

The repo supports the following dataset layouts:

- COLMAP reconstructions with `sparse/0` or `sparse/`
- `transforms.json` datasets from NeRF On-the-go / nerfstudio-style exports
- Optional `PointCloud.ply` in the dataset root
- Optional semantic sidecars next to images:
  - `*_sdfeats.npy`
  - `*_sdfeats_clustered.npy`

The viewer will auto-detect the dataset format from the path you pass in.

## Usage

Start the viewer:

```bash
uv run python -m viewer --dataset /path/to/dataset
```

Load a 3DGS PLY file:

```bash
uv run python -m viewer --ply /path/to/model.ply
```

Run headless training and save outputs to `./output`:

```bash
uv run python -m viewer --dataset /path/to/dataset --headless --output ./output
```

Adjust the window size:

```bash
uv run python -m viewer --ply /path/to/model.ply --width 1920 --height 1080
```

Enable debug logging:

```bash
uv run python -m viewer --ply /path/to/model.ply --log-level DEBUG
```

Extract Stable Diffusion features from a directory of images:

```bash
uv run python scripts/extract_sd_features.py /path/to/images --img-size 800
```

Extract clustered Stable Diffusion features:

```bash
uv run python scripts/extract_sd_features.py /path/to/images --img-size 800 --cluster --n-clusters 100
```

Show all viewer options:

```bash
uv run python -m viewer --help
```

Show all feature-extraction options:

```bash
uv run python scripts/extract_sd_features.py --help
```

## Controls

| Input | Action |
| --- | --- |
| Left drag | Rotate camera in orbit mode / look around in FPS mode |
| Right drag | Pan camera in orbit mode |
| Scroll | Zoom orbit camera / change FPS move speed |
| `WASD` | Move camera |
| `Q` / `E` | Move down / up |
| `R` + scroll | Roll orbit camera |
| `ESC` | Release FPS cursor or quit |

## Implementation Notes

- The viewer keeps orbit, FPS, and dataset-frustum cameras in sync with the same scene state.
- COLMAP and `transforms.json` inputs are normalized into a common internal representation.
- On-the-go training uses rectified pinhole images and keeps the viewer and trainer aligned on the same camera contract.
- The trainer runs in a background thread so the UI stays responsive.
- Learned splats can be exported back to PLY from the UI or headless mode.

## External Resources

- NeRF On-the-go dataset and project page: https://rwn17.github.io/nerf-on-the-go/
- `gsplat` rasterizer: `external/gsplat`
- `imgui-bundle`, `glfw`, `PyOpenGL`, `pycolmap`, `torch`, `opencv-python`, and related Python packages from `pyproject.toml`
- Stable Diffusion feature extraction script: `scripts/extract_sd_features.py`

