"""Stable Diffusion feature extraction (DIFT-style).

Based on the official DIFT code and the SpotLessSplats feature extraction notebook.
Uses diffusers to extract UNet up-block features from Stable Diffusion 2.1.
"""

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from loguru import logger
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph


def cluster_features(
    features: np.ndarray,
    n_clusters: int = 100,
    n_neighbors: int = 8,
) -> np.ndarray:
    """Cluster SD features into spatial-semantic clusters.

    This is the SLS-agg feature representation. Each cluster is a binary mask
    indicating which pixels belong to that cluster.

    Args:
        features: [C, H, W] raw SD features.
        n_clusters: Number of clusters.
        n_neighbors: Number of neighbors for spatial connectivity graph.

    Returns:
        [n_clusters, H, W] binary cluster masks (float32).
    """
    C, H, W = features.shape
    logger.debug(f"Clustering features: shape={features.shape}, n_clusters={n_clusters}")

    # Flatten features: [H*W, C]
    ft_flat = np.transpose(features.reshape(C, H * W), (1, 0))

    # Build spatial coordinates [0, 1]
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    xv, yv = np.meshgrid(x, y)
    indxy = np.reshape(np.stack([xv, yv], axis=-1), (H * W, 2))

    # KNN graph for spatial connectivity
    logger.debug("Building spatial KNN graph...")
    knn_graph = kneighbors_graph(indxy, n_neighbors, include_self=False)

    # Agglomerative clustering with Ward linkage
    logger.debug("Running AgglomerativeClustering...")
    model = AgglomerativeClustering(
        linkage="ward",
        connectivity=knn_graph,
        n_clusters=n_clusters,
    )
    model.fit(ft_flat)

    # Convert labels to binary masks [n_clusters, H, W]
    cluster_masks = np.array(
        [model.labels_ == i for i in range(n_clusters)],
        dtype=np.float32,
    ).reshape((n_clusters, H, W))

    logger.debug(f"Clustering complete: {n_clusters} clusters")
    return cluster_masks


class MyUNet2DConditionModel(UNet2DConditionModel):
    """UNet wrapper that exposes intermediate up-block features."""

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices: List[int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], torch.FloatTensor]:
        """Forward pass that captures up-block features."""
        default_overall_up_factor = 2**self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # Pre-process
        sample = self.conv_in(sample)

        # Down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        # Mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # Up
        up_ft: Dict[int, torch.Tensor] = {}
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

            if i in up_ft_indices:
                up_ft[i] = sample.detach()

        # Post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return {"up_ft": up_ft}, sample


class OneStepSDPipeline(StableDiffusionPipeline):
    """Pipeline that runs a single denoising step and extracts features."""

    @torch.no_grad()
    def __call__(
        self,
        img_tensor: torch.Tensor,
        t: int,
        up_ft_indices: List[int],
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        """Run one diffusion step and extract features."""
        device = self._execution_device
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        t_tensor = torch.tensor(t, dtype=torch.long, device=device)
        if noise is None:
            noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t_tensor)

        unet_output, noise_pred = self.unet(
            latents_noisy,
            t_tensor,
            up_ft_indices,
            encoder_hidden_states=prompt_embeds,
        )

        # Predict original sample
        latents_clean = self.scheduler.step(noise_pred, t_tensor, latents_noisy).pred_original_sample
        latents_clean = 1 / self.vae.config.scaling_factor * latents_clean
        image = self.vae.decode(latents_clean).sample

        return unet_output, image


class SDFeaturizer:
    """Extract Stable Diffusion features from images.

    Usage:
        featurizer = SDFeaturizer()
        img_tensor = torch.tensor(np.array(Image.open("img.jpg"))) / 255.0
        img_tensor = (img_tensor - 0.5) * 2
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        features, image = featurizer.forward(img_tensor, prompt="", ensemble_size=4)
        # features: [1, C, H, W] where C is the feature dimension
    """

    def __init__(
        self,
        sd_id: str = "sd2-community/stable-diffusion-2-1",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.sd_id = sd_id
        self.dtype = dtype

        logger.info(f"Initializing SD Featurizer with model: {sd_id}")
        logger.info(f"Device: {device}, dtype: {dtype}")

        try:
            # Load UNet with our custom forward
            logger.debug("Loading UNet model...")
            unet = MyUNet2DConditionModel.from_pretrained(
                sd_id,
                subfolder="unet",
                use_safetensors=False,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            logger.debug("UNet loaded successfully")

            logger.debug("Loading Stable Diffusion pipeline...")
            self.pipe = OneStepSDPipeline.from_pretrained(
                sd_id,
                unet=unet,
                safety_checker=None,
                use_safetensors=False,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            logger.debug("Pipeline loaded successfully")

            self.pipe.scheduler = DDIMScheduler.from_pretrained(
                sd_id,
                subfolder="scheduler",
                use_safetensors=False,
            )
            self.pipe.scheduler.set_timesteps(50)
            self.pipe = self.pipe.to(device)
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_slicing()

            logger.info("SD Featurizer initialized successfully")
            logger.info(f"GPU memory used after init: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        except Exception as e:
            logger.error(f"Failed to initialize SD Featurizer: {e}")
            logger.error("If the model requires authentication, run: huggingface-cli login")
            raise

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            logger.info(f"GPU memory after cache clear: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    @torch.no_grad()
    def forward(
        self,
        img_tensor: torch.Tensor,
        prompt: str = "",
        t: int = 261,
        up_ft_index: List[int] = [1],
        ensemble_size: int = 1,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from a single image.

        Args:
            img_tensor: [1, 3, H, W] image in [-1, 1] range.
            prompt: Text prompt (empty string for unconditioned).
            t: Diffusion timestep.
            up_ft_index: List of UNet up-block indices to extract.
            ensemble_size: Number of ensemble samples (average features). Default 1.
            noise: Optional fixed noise for reproducibility.

        Returns:
            features: [1, C, H, W] extracted features.
            image: [1, 3, H, W] reconstructed image.
        """
        img_tensor = img_tensor.to(self.device)
        if ensemble_size > 1:
            img_tensor = img_tensor.repeat(ensemble_size, 1, 1, 1)

        prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )[0]
        if ensemble_size > 1:
            prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1)

        unet_output, image = self.pipe(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=up_ft_index,
            prompt_embeds=prompt_embeds,
            noise=noise,
        )

        # Average features across ensemble
        fts = []
        max_h, max_w = 0, 0
        for i in up_ft_index:
            unet_ft = unet_output["up_ft"][i]  # [ensemble, C, H, W]
            if ensemble_size > 1:
                unet_ft = unet_ft.mean(0, keepdim=True)  # [1, C, H, W]
            max_h = max(max_h, unet_ft.shape[-2])
            max_w = max(max_w, unet_ft.shape[-1])
            fts.append(unet_ft)

        # Resize all to same size
        fts_resized = []
        for ft in fts:
            if ft.shape[-2] != max_h or ft.shape[-1] != max_w:
                ft = F.interpolate(ft, size=(max_h, max_w), mode="bilinear", align_corners=False)
            fts_resized.append(ft)

        features = torch.cat(fts_resized, dim=0)  # [n, C, H, W]
        # If multiple up_ft_index, concatenate channels
        if len(up_ft_index) > 1:
            features = features.permute(1, 0, 2, 3)  # [1, C_total, H, W]
        else:
            features = features  # Already [1, C, H, W]

        # Move back to CPU and clear cache
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return features, image

    def extract_from_path(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        img_size: Optional[int] = None,
        save_preview: bool = True,
        cluster: bool = False,
        n_clusters: int = 100,
        t: int = 261,
        up_ft_index: List[int] = [1],
        ensemble_size: int = 1,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extract features from an image file and save to disk.

        Args:
            image_path: Path to the image.
            output_path: Path to save the .npy file. If None, auto-generated.
            img_size: Resize image to this size. If None, keep original.
            save_preview: Save a PCA-projected RGB preview PNG.
            cluster: If True, cluster features and save cluster masks.
            n_clusters: Number of clusters for SLS-agg.
            t: Diffusion timestep.
            up_ft_index: UNet up-block indices.
            ensemble_size: Number of ensemble samples.

        Returns:
            features: numpy array [C, H, W] or [n_clusters, H, W] if clustered.
            preview: numpy array [H, W, 3] if save_preview=True, else None.
        """
        image_path = Path(image_path)
        if output_path is None:
            suffix = "_sdfeats_clustered" if cluster else "_sdfeats"
            output_path = image_path.parent / f"{image_path.stem}{suffix}.npy"
        else:
            output_path = Path(output_path)

        logger.debug(f"Processing image: {image_path.name}")
        logger.debug(f"  Output: {output_path}")
        logger.debug(f"  Cluster: {cluster}, n_clusters: {n_clusters}")

        # Load image
        img = Image.open(image_path).convert("RGB")
        if img_size is not None:
            img = img.resize((img_size, img_size))
            logger.debug(f"  Resized to {img_size}x{img_size}")

        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = (torch.from_numpy(img_np) - 0.5) * 2
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        # Cast to model dtype (e.g., float16) to avoid type mismatch
        img_tensor = img_tensor.to(self.dtype)

        # Extract features
        logger.debug("  Extracting SD features...")
        features, _ = self.forward(
            img_tensor,
            prompt="",
            t=t,
            up_ft_index=up_ft_index,
            ensemble_size=ensemble_size,
        )

        features_np = features.squeeze(0).cpu().numpy()  # [C, H, W]
        logger.debug(f"  Features shape: {features_np.shape}")

        # Cluster if requested
        if cluster:
            logger.debug(f"  Clustering features into {n_clusters} clusters...")
            features_np = cluster_features(features_np, n_clusters=n_clusters)
            logger.debug(f"  Clustered features shape: {features_np.shape}")

            # Save cluster preview
            if save_preview:
                preview = self._compute_cluster_preview(features_np)
                preview_path = output_path.parent / f"{image_path.stem}_sdfeats_clustered_preview.png"
                preview_img = Image.fromarray((preview * 255).astype(np.uint8))
                preview_img.save(preview_path)
                logger.debug(f"  Saved cluster preview: {preview_path}")
        else:
            # Generate PCA preview for raw features
            if save_preview:
                preview = self._compute_pca_preview(features_np.astype(np.float32))
                preview_path = output_path.parent / f"{image_path.stem}_sdfeats_preview.png"
                preview_img = Image.fromarray((preview * 255).astype(np.uint8))
                preview_img.save(preview_path)
                logger.debug(f"  Saved PCA preview: {preview_path}")

        # Save features
        np.save(output_path, features_np)
        logger.debug(f"  Saved features: {output_path}")

        preview = None
        return features_np, preview

    @staticmethod
    def _compute_pca_preview(features: np.ndarray, n_components: int = 3) -> np.ndarray:
        """Compute PCA projection of features to RGB for visualization.

        Args:
            features: [C, H, W] feature array.
            n_components: Number of PCA components (3 for RGB).

        Returns:
            [H, W, 3] array in [0, 1] range.
        """
        C, H, W = features.shape
        features_flat = features.reshape(C, -1).T  # [H*W, C]

        # Normalize
        mean = features_flat.mean(axis=0, keepdims=True)
        std = features_flat.std(axis=0, keepdims=True) + 1e-8
        features_norm = (features_flat - mean) / std

        # PCA via SVD
        u, s, vh = np.linalg.svd(features_norm, full_matrices=False)
        components = vh[:n_components].T  # [C, 3]

        pca = features_flat @ components  # [H*W, 3]
        pca = pca.reshape(H, W, 3)

        # Normalize to [0, 1]
        pca = (pca - pca.min()) / (pca.max() - pca.min() + 1e-8)

        return pca

    @staticmethod
    def _compute_cluster_preview(cluster_masks: np.ndarray) -> np.ndarray:
        """Compute a color preview of cluster masks.

        Args:
            cluster_masks: [n_clusters, H, W] binary masks.

        Returns:
            [H, W, 3] RGB image in [0, 1] range.
        """
        n_clusters, H, W = cluster_masks.shape
        # Assign a random color to each cluster
        rng = np.random.RandomState(42)
        colors = rng.rand(n_clusters, 3)
        colors = colors / (colors.sum(axis=1, keepdims=True) + 1e-8)

        # Weighted sum of cluster masks
        preview = np.zeros((H, W, 3), dtype=np.float32)
        for i in range(n_clusters):
            preview += cluster_masks[i][:, :, None] * colors[i]

        preview = preview / (preview.max() + 1e-8)
        return preview


class DirectoryExtractor:
    """Batch extract SD features from all images in a directory."""

    def __init__(self, featurizer: Optional[SDFeaturizer] = None, device: str = "cuda", dtype=torch.float16):
        if featurizer is None:
            logger.info("Creating new SDFeaturizer (this may take a moment to download models)...")
            featurizer = SDFeaturizer(device=device, dtype=dtype)
        self.featurizer = featurizer

    def extract_directory(
        self,
        image_dir: Union[str, Path],
        img_size: Optional[int] = None,
        cluster: bool = False,
        n_clusters: int = 100,
        t: int = 261,
        up_ft_index: List[int] = [1],
        ensemble_size: int = 1,
        save_preview: bool = True,
        skip_existing: bool = True,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"),
    ) -> List[Path]:
        """Extract features from all images in a directory.

        Args:
            image_dir: Directory containing images.
            img_size: Resize images to this size. If None, keep original.
            cluster: If True, cluster features (SLS-agg format).
            n_clusters: Number of clusters for SLS-agg.
            t: Diffusion timestep.
            up_ft_index: UNet up-block indices.
            ensemble_size: Number of ensemble samples.
            save_preview: Save PCA preview PNGs.
            skip_existing: Skip images that already have feature files.
            extensions: Image file extensions to process.

        Returns:
            List of paths to saved feature files.
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise ValueError(f"Directory not found: {image_dir}")

        logger.info(f"Scanning directory: {image_dir}")

        # Find all images, excluding any that contain `_sdfeats` in their name
        # (these are previously generated features/previews)
        image_paths = []
        for ext in extensions:
            image_paths.extend(image_dir.glob(f"*{ext}"))
            image_paths.extend(image_dir.glob(f"*{ext.upper()}"))

        # Filter out feature files and previews
        image_paths = [p for p in image_paths if "_sdfeats" not in p.stem]

        # Filter out existing features if skip_existing
        suffix = "_sdfeats_clustered" if cluster else "_sdfeats"
        if skip_existing:
            image_paths = [
                p for p in image_paths
                if not (p.parent / f"{p.stem}{suffix}.npy").exists()
            ]

        image_paths = sorted(image_paths)
        if not image_paths:
            logger.warning("No images to process (all have features or directory is empty).")
            return []

        logger.info(f"Found {len(image_paths)} images to process")
        logger.info(f"Mode: {'clustered (SLS-agg)' if cluster else 'raw (SLS-mlp)'}")
        logger.info(f"Output suffix: {suffix}")

        output_paths = []

        for img_path in tqdm(image_paths, desc="Extracting SD features", unit="img"):
            try:
                output_path = img_path.parent / f"{img_path.stem}{suffix}.npy"
                self.featurizer.extract_from_path(
                    img_path,
                    output_path=output_path,
                    img_size=img_size,
                    cluster=cluster,
                    n_clusters=n_clusters,
                    save_preview=save_preview,
                    t=t,
                    up_ft_index=up_ft_index,
                    ensemble_size=ensemble_size,
                )
                output_paths.append(output_path)
                # Clear CUDA cache after each image to prevent OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    mem_used = torch.cuda.memory_allocated() / 1e9
                    if mem_used > 10:
                        logger.warning(f"High GPU memory usage: {mem_used:.2f} GB")
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        logger.info(f"Extraction complete! Processed {len(output_paths)}/{len(image_paths)} images")
        return output_paths


if __name__ == "__main__":
    # Simple test
    import sys

    if len(sys.argv) > 1:
        test_dir = Path(sys.argv[1])
        extractor = DirectoryExtractor()
        extractor.extract_directory(test_dir, img_size=800, save_preview=True)
    else:
        print("Usage: python sd_features.py <image_dir>")
