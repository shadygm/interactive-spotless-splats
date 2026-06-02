import sys
import os
import threading
import math
import time
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
from loguru import logger

from gsplat.rendering import rasterization
from gsplat.strategy import MCMCStrategy, DefaultStrategy
from gsplat import export_splats


# ---------------------------------------------------------------------------
# Simple COLMAP parser using pycolmap 4.x
# ---------------------------------------------------------------------------

class _SimpleColmapParser:
    """Minimal COLMAP parser using pycolmap 4.x API."""

    def __init__(self, data_dir: str, factor: int = 1, test_every: int = 8):
        import pycolmap

        self.data_dir = data_dir
        self.factor = factor
        self.test_every = test_every

        # Find sparse reconstruction directory
        sparse_dir = os.path.join(data_dir, "sparse", "0")
        if not os.path.exists(sparse_dir):
            sparse_dir = os.path.join(data_dir, "sparse")
        if not os.path.exists(sparse_dir):
            raise ValueError(f"COLMAP sparse directory not found in {data_dir}")

        # Read reconstruction
        rec = pycolmap.Reconstruction()
        rec.read(sparse_dir)

        if rec.num_images == 0:
            raise ValueError("No images found in COLMAP reconstruction.")

        # Collect camera data
        self.Ks_dict: Dict[int, np.ndarray] = {}
        self.imsize_dict: Dict[int, Tuple[int, int]] = {}
        for cam_id, cam in rec.cameras.items():
            params = np.array(cam.params, dtype=np.float64)
            model = cam.model_name
            if model == "PINHOLE":
                fx, fy, cx, cy = params
            elif model == "SIMPLE_PINHOLE":
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:
                # Default to first 4 params for other models
                fx, fy, cx, cy = params[:4]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            if factor > 1:
                K[:2, :] /= factor
            self.Ks_dict[cam_id] = K
            w = cam.width // factor
            h = cam.height // factor
            self.imsize_dict[cam_id] = (w, h)

        # Collect image data
        w2c_mats = []
        camera_ids = []
        image_names = []

        for img_id in sorted(rec.images.keys()):
            img = rec.images[img_id]
            if not img.has_pose:
                continue
            image_names.append(img.name)
            camera_ids.append(img.camera_id)

            # World-to-camera
            rigid = img.cam_from_world()
            R = np.array(rigid.rotation.matrix(), dtype=np.float64)
            t = np.array(rigid.translation, dtype=np.float64).reshape(3, 1)
            w2c = np.concatenate([np.concatenate([R, t], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
            w2c_mats.append(w2c)

        w2c_mats = np.stack(w2c_mats, axis=0)
        camtoworlds = np.linalg.inv(w2c_mats)

        # Sort by image name for consistent train/val splits
        inds = np.argsort(image_names)
        self.image_names = [image_names[i] for i in inds]
        self.camtoworlds = camtoworlds[inds].astype(np.float32)
        self.camera_ids = [camera_ids[i] for i in inds]

        # Create 0-based contiguous camera indices
        unique_camera_ids = sorted(set(self.camera_ids))
        self.camera_id_to_idx = {cid: idx for idx, cid in enumerate(unique_camera_ids)}
        self.camera_indices = [self.camera_id_to_idx[cid] for cid in self.camera_ids]
        self.num_cameras = len(unique_camera_ids)

        # Image paths: look for pre-resized folder first (e.g. images_4)
        # This matches the simple_trainer.py behavior.
        colmap_image_dir = os.path.join(data_dir, "images")
        if factor > 1:
            image_dir = os.path.join(data_dir, f"images_{factor}")
            if not os.path.exists(image_dir):
                image_dir = colmap_image_dir
        else:
            image_dir = colmap_image_dir
        if not os.path.exists(image_dir):
            raise ValueError(f"Image folder {image_dir} does not exist.")
        self.image_paths = [os.path.join(image_dir, name) for name in self.image_names]

        # Point cloud
        points_list = []
        colors_list = []
        for pt_id in rec.points3D:
            pt = rec.points3D[pt_id]
            points_list.append(pt.xyz)
            colors_list.append(pt.color)
        self.points = np.array(points_list, dtype=np.float32)
        self.points_rgb = np.array(colors_list, dtype=np.uint8)

        # Scene scale
        camera_locations = self.camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = float(np.max(dists))


class _SimpleColmapDataset(torch.utils.data.Dataset):
    """Minimal COLMAP dataset for training.

    Loads images fresh from disk every iteration (no caching).
    The DataLoader workers handle prefetching. This matches simple_trainer.py.
    """

    def __init__(self, parser: _SimpleColmapParser, split: str = "train"):
        self.parser = parser
        self.split = split
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image_path = self.parser.image_paths[index]

        # Load image with imageio (fast, no PIL overhead).
        # If a pre-resized folder was found, this is already downsampled.
        image = imageio.imread(image_path)[..., :3]

        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy().astype(np.float32)
        camtoworld = self.parser.camtoworlds[index].copy()

        data = {
            "K": torch.from_numpy(K),
            "camtoworld": torch.from_numpy(camtoworld),
            "image": torch.from_numpy(image),
            "image_id": item,
            "camera_idx": self.parser.camera_indices[index],
        }
        return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _knn(x: torch.Tensor, K: int = 4) -> torch.Tensor:
    """kNN using scikit-learn KD-tree (fast, low memory)."""
    from sklearn.neighbors import NearestNeighbors

    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Simple PyTorch SSIM implementation."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, window_size, 1, window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, 1, window_size // 2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, 1, window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, 1, window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


# Compile SSIM for speed (low-overhead mode for small graphs)
_compiled_ssim = torch.compile(_ssim, mode="reduce-overhead", dynamic=False, fullgraph=False)


def _rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB to SH DC coefficients."""
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def _create_splats_and_optimizers(
    parser: _SimpleColmapParser,
    device: str,
    sh_degree: int = 3,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
):
    """Create splats and optimizers from COLMAP parser point cloud."""
    points = torch.from_numpy(parser.points).float()
    rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()

    # Initialize scales from kNN distances
    dist2_avg = (_knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)

    N = points.shape[0]
    quats = torch.rand((N, 4))
    opacities = torch.logit(torch.full((N,), init_opacity))

    # SH colors
    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))
    colors[:, 0, :] = _rgb_to_sh(rgbs)

    params = [
        ("means", torch.nn.Parameter(points), 1.6e-4),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
        ("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3),
        ("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20),
    ]

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    optimizers = {
        name: torch.optim.Adam(
            [{"params": splats[name], "lr": lr, "name": name}],
            eps=1e-15,
            betas=(0.9, 0.999),
            fused=True,
        )
        for name, _, lr in params
    }

    return splats, optimizers


# ---------------------------------------------------------------------------
# Config and Trainer
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    max_splats: int = 1_000_000
    num_iterations: int = 30_000
    ssim_lambda: float = 0.2
    sh_degree: int = 3
    batch_size: int = 1
    data_factor: int = 4
    test_every: int = 8
    device: str = "cuda"
    strategy: str = "mcmc"  # "mcmc" or "default"
    headless: bool = False
    output_dir: str = "./output"


class Trainer:
    """Background-thread 3DGS trainer using MCMC strategy."""

    def __init__(self, scene_state, config: Optional[TrainerConfig] = None):
        self.scene_state = scene_state
        self.config = config or TrainerConfig()

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Loss history: list of (iteration, loss_value)
        self.loss_history: List[Tuple[int, float]] = []
        self.current_iteration = 0
        self.current_splats = 0
        self.is_running = False
        self.status_message = "Idle"
        self._train_start_time = 0.0

        # Training state (owned by training thread)
        self._splats = None
        self._optimizers = None
        self._strategy = None
        self._strategy_state = None
        self._trainloader = None
        self._trainloader_iter = None
        self._scheduler = None

    def start(self, colmap_path: str):
        """Start training in a background thread."""
        if self.is_running:
            logger.warning("Trainer already running")
            return

        self._stop_event.clear()
        self.loss_history.clear()
        self.current_iteration = 0
        self.status_message = "Initializing..."

        self._thread = threading.Thread(
            target=self._train_loop, args=(colmap_path,), daemon=True
        )
        self._thread.start()

    def stop(self):
        """Signal the training thread to stop and wait for it."""
        if not self.is_running:
            return
        self._stop_event.set()
        self.status_message = "Stopping..."
        if self._thread:
            self._thread.join(timeout=5.0)
        self.is_running = False
        self.status_message = "Stopped"

    def _train_loop(self, colmap_path: str):
        try:
            self.is_running = True
            cfg = self.config
            device = cfg.device

            logger.info(f"[TRAINER] CUDA available: {torch.cuda.is_available()}")
            logger.info(f"[TRAINER] Using device: {device}")
            if torch.cuda.is_available():
                logger.info(f"[TRAINER] CUDA device: {torch.cuda.get_device_name(0)}")
                # Enable TF32 for faster matmul on Ampere+ GPUs
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision('high')
                # Allow cuDNN to benchmark optimal algorithms for consistent input sizes
                torch.backends.cudnn.benchmark = True

            # Parse COLMAP data
            parser = _SimpleColmapParser(
                data_dir=colmap_path,
                factor=cfg.data_factor,
                test_every=cfg.test_every,
            )
            trainset = _SimpleColmapDataset(parser, split="train")

            scene_scale = parser.scene_scale * 1.1

            # Create splats from point cloud
            self._splats, self._optimizers = _create_splats_and_optimizers(
                parser, device, sh_degree=cfg.sh_degree
            )
            logger.info(f"Initialized {len(self._splats['means'])} splats from point cloud")
            logger.info(f"[TRAINER] Splats device: {self._splats['means'].device}")

            # Setup strategy
            if cfg.strategy == "mcmc":
                self._strategy = MCMCStrategy(
                    cap_max=cfg.max_splats,
                    refine_start_iter=500,
                    refine_stop_iter=max(cfg.num_iterations - 5000, 1000),
                    refine_every=100,
                )
            else:
                self._strategy = DefaultStrategy(
                    cap_max=cfg.max_splats,
                    refine_start_iter=500,
                    refine_stop_iter=max(cfg.num_iterations - 5000, 1000),
                    refine_every=100,
                )
            self._strategy.check_sanity(self._splats, self._optimizers)
            if isinstance(self._strategy, DefaultStrategy):
                self._strategy_state = self._strategy.initialize_state(scene_scale=scene_scale)
            else:
                self._strategy_state = self._strategy.initialize_state()

            # Learning rate scheduler for means
            self._scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self._optimizers["means"], gamma=0.01 ** (1.0 / cfg.num_iterations)
            )

            # DataLoader: use num_workers>0 for prefetching and pin_memory
            # for faster async CPU->GPU transfers. persistent_workers avoids
            # process spawn overhead across epochs.
            loader_kwargs = {
                "batch_size": cfg.batch_size,
                "shuffle": True,
                "num_workers": 4,
                "persistent_workers": True,
                "pin_memory": True,
            }
            self._trainloader = torch.utils.data.DataLoader(trainset, **loader_kwargs)
            self._trainloader_iter = iter(self._trainloader)

            # Training loop
            self._train_start_time = time.time()
            for step in range(cfg.num_iterations):
                if self._stop_event.is_set():
                    break

                t_data_start = time.time()
                try:
                    data = next(self._trainloader_iter)
                except StopIteration:
                    self._trainloader_iter = iter(self._trainloader)
                    data = next(self._trainloader_iter)
                t_data = time.time() - t_data_start

                t_gpu_start = time.time()
                camtoworlds = data["camtoworld"].to(device, non_blocking=True)
                Ks = data["K"].to(device, non_blocking=True)
                pixels = data["image"].to(device, non_blocking=True) / 255.0

                height, width = pixels.shape[1:3]

                # Use the SH degree selected in the UI
                sh_degree_to_use = cfg.sh_degree

                # Forward
                means = self._splats["means"]
                quats = self._splats["quats"]
                scales = torch.exp(self._splats["scales"])
                opacities = torch.sigmoid(self._splats["opacities"])
                colors = torch.cat([self._splats["sh0"], self._splats["shN"]], 1)

                if step == 0:
                    logger.info(f"[TRAINER] means device: {means.device}")
                    logger.info(f"[TRAINER] pixels device: {pixels.device}")
                    logger.info(f"[TRAINER] image size: {height}x{width}")

                renders, alphas, info = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=torch.linalg.inv(camtoworlds),
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_to_use,
                    near_plane=0.01,
                    far_plane=1e10,
                    packed=False,
                )

                colors_render = renders[..., :3]

                # Loss: SSIM + L1
                l1loss = F.l1_loss(colors_render, pixels)
                ssimloss = 1.0 - _compiled_ssim(
                    colors_render.permute(0, 3, 1, 2),
                    pixels.permute(0, 3, 1, 2),
                )
                loss = torch.lerp(l1loss, ssimloss, cfg.ssim_lambda)

                # Strategy pre-backward (DefaultStrategy only)
                if isinstance(self._strategy, DefaultStrategy):
                    self._strategy.step_pre_backward(
                        params=self._splats,
                        optimizers=self._optimizers,
                        state=self._strategy_state,
                        step=step,
                        info=info,
                    )

                # Backward
                loss.backward()

                # Optimize
                for optimizer in self._optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                self._scheduler.step()

                # Strategy post-backward
                if isinstance(self._strategy, MCMCStrategy):
                    self._strategy.step_post_backward(
                        params=self._splats,
                        optimizers=self._optimizers,
                        state=self._strategy_state,
                        step=step,
                        info=info,
                        lr=self._scheduler.get_last_lr()[0],
                    )
                else:
                    self._strategy.step_post_backward(
                        params=self._splats,
                        optimizers=self._optimizers,
                        state=self._strategy_state,
                        step=step,
                        info=info,
                        packed=False,
                    )

                self.current_iteration = step + 1
                self.current_splats = len(self._splats['means'])

                # Every 100 iterations: update viewer and log loss
                if (step + 1) % 100 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_gpu = time.time() - t_gpu_start
                    loss_val = loss.item()
                    with self._lock:
                        self.loss_history.append((step + 1, loss_val))

                    # Export gaussians to scene_state (skip in headless mode)
                    if not cfg.headless:
                        self._export_gaussians()

                    elapsed = time.time() - self._train_start_time
                    it_per_sec = (step + 1) / elapsed if elapsed > 0 else 0.0
                    self.status_message = (
                        f"{step + 1}/{cfg.num_iterations} ({it_per_sec:.1f} it/s)"
                    )
                    logger.info(
                        f"Training... {self.status_message}, "
                        f"loss={loss_val:.4f}, splats={self.current_splats}"
                    )
                    logger.info(f"[TRAINER] Timing: data={t_data:.3f}s, gpu={t_gpu:.3f}s")

            self.status_message = "Training complete"
            logger.info("Training complete")

            # Save final PLY in headless mode
            if cfg.headless and self._splats is not None:
                self._save_ply(colmap_path)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.status_message = f"Error: {e}"
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False

    def _export_gaussians(self):
        """Export current splats to scene_state.gaussians in viewer format."""
        if self._splats is None:
            return

        with torch.no_grad():
            means = self._splats["means"].detach().clone()
            quats = F.normalize(self._splats["quats"], dim=-1).detach().clone()
            scales = torch.exp(self._splats["scales"]).detach().clone()
            opacities = torch.sigmoid(self._splats["opacities"]).detach().clone()
            colors = torch.cat(
                [self._splats["sh0"], self._splats["shN"]], 1
            ).detach().clone()

            # Flip Y axis to match viewer convention (same as PlyLoader)
            means[:, 1] *= -1
            # Quaternions: [w, x, y, z] -> [w, -x, y, -z] for Y flip
            quats[:, 1] *= -1
            quats[:, 3] *= -1

            gaussians = {
                "means": means,
                "quats": quats,
                "scales": scales,
                "opacities": opacities,
                "colors": colors,
                "sh_degree": self.config.sh_degree,
            }

            # Update scene_state under its lock
            with self.scene_state._lock:
                self.scene_state.gaussians = gaussians
                self.scene_state.has_gaussians = True
                self.scene_state._default_gaussians = False
                self.scene_state._bump_version()

    def _save_ply(self, colmap_path: str):
        """Export current splats to a PLY file."""
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)
        scene_name = Path(colmap_path).name or "scene"
        ply_path = os.path.join(cfg.output_dir, f"{scene_name}.ply")

        with torch.no_grad():
            means = self._splats["means"].detach().clone()
            scales = self._splats["scales"].detach().clone()
            quats = self._splats["quats"].detach().clone()
            opacities = self._splats["opacities"].detach().clone()
            sh0 = self._splats["sh0"].detach().clone()
            shN = self._splats["shN"].detach().clone()

        export_splats(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            format="ply",
            save_to=ply_path,
        )
        logger.info(f"Saved PLY to {ply_path}")
