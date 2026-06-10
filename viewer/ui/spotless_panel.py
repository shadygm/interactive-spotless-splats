from __future__ import annotations

from typing import Optional

import numpy as np
import imageio.v2 as imageio
from imgui_bundle import imgui, immvision

from viewer.ui.panels import Panel


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    return np.ascontiguousarray(image)


def _overlay_mask(render: np.ndarray, mask: Optional[np.ndarray], opacity: float) -> np.ndarray:
    if mask is None:
        return _to_uint8_rgb(render)
    render = np.clip(np.asarray(render, dtype=np.float32), 0.0, 1.0)
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    mask = np.clip(mask, 0.0, 1.0)
    inlier = mask > 0.5
    overlay = render.copy()
    overlay[inlier] = (1.0 - opacity) * overlay[inlier] + opacity * np.array([0.2, 0.9, 0.2], dtype=np.float32)
    overlay[~inlier] = (1.0 - opacity) * overlay[~inlier] + opacity * np.array([0.95, 0.2, 0.2], dtype=np.float32)
    return _to_uint8_rgb(overlay)


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        return _to_uint8_rgb(values)

    values = values - values.min()
    values = values / (values.max() + 1e-8)
    r = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return _to_uint8_rgb(np.stack([r, g, b], axis=-1))


def _feature_preview(features: np.ndarray, mode: str, channel: int) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 3:
        return _to_uint8_rgb(features)

    c, h, w = features.shape
    if mode == "Channel":
        channel = int(channel) % max(c, 1)
        preview = features[channel]
        preview = (preview - preview.min()) / (preview.max() - preview.min() + 1e-8)
        return _to_uint8_rgb(preview)
    if mode == "Norm":
        preview = np.linalg.norm(features, axis=0)
        preview = (preview - preview.min()) / (preview.max() - preview.min() + 1e-8)
        return _to_uint8_rgb(preview)

    flat = features.reshape(c, -1).T
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True) + 1e-8
    flat_norm = (flat - mean) / std
    _, _, vh = np.linalg.svd(flat_norm, full_matrices=False)
    components = vh[:3].T
    preview = (flat @ components).reshape(h, w, 3)
    preview = (preview - preview.min()) / (preview.max() - preview.min() + 1e-8)
    return _to_uint8_rgb(preview)


class SpotlessPanel(Panel):
    def __init__(self, trainer, scene_state):
        self.trainer = trainer
        self.scene_state = scene_state
        self._feature_modes = ["PCA", "Channel", "Norm"]
        self._feature_mode_idx = 0
        self._channel_idx = 0
        self._mask_opacity = 0.5
        self._show_overlay = True
        self._step_mode = False
        self._preview_cache_key = None
        self._preview_images = {
            "gt": None,
            "render": None,
            "features": None,
            "error": None,
        }
        self._feature_preview_cache = {}

    def draw(self):
        imgui.text("Spotless Training")
        imgui.separator()

        has_features = bool(getattr(self.trainer, "_semantics_available", False))
        if has_features:
            imgui.text_colored(imgui.ImVec4(0.7, 0.9, 0.7, 1.0), "Semantic features detected")
        else:
            imgui.text_disabled("No semantic feature files found for this dataset")

        changed_enabled, enabled = imgui.checkbox("Enable Spotless Masking", self.trainer.config.semantics)
        if changed_enabled:
            self.trainer.config.semantics = enabled

        changed_loss, loss_idx = imgui.combo(
            "Loss Type",
            0 if self.trainer.config.loss_type == "l1" else 1,
            ["l1", "robust"],
        )
        if changed_loss:
            self.trainer.config.loss_type = ["l1", "robust"][loss_idx]

        imgui.text("Semantic Mode")
        is_mlp = not self.trainer.config.cluster
        if imgui.radio_button("SLS-mlp", is_mlp):
            self.trainer.config.cluster = False
        imgui.same_line()
        if imgui.radio_button("SLS-agg", self.trainer.config.cluster):
            self.trainer.config.cluster = True

        imgui.separator()

        changed_step, step_mode = imgui.checkbox("Step-by-step Mode", self._step_mode)
        if changed_step:
            self._step_mode = step_mode
            self.trainer.set_step_mode(step_mode)

        imgui.begin_disabled(not self._step_mode)
        if imgui.button("Next Iteration", imgui.ImVec2(imgui.get_content_region_avail().x, 0)):
            self.trainer.request_next_step()
        imgui.end_disabled()

        imgui.separator()

        snapshot = self.trainer.get_spotless_snapshot()
        if snapshot is None:
            imgui.text_disabled("No spotless snapshot yet. Start training to inspect batches.")
            return

        imgui.text(f"Step: {snapshot.get('step', 0)}")
        imgui.text(f"Loss: {snapshot.get('loss', 0.0):.4f}")
        if snapshot.get("spot_loss") is not None:
            imgui.text(f"Spot loss: {snapshot['spot_loss']:.4f}")
        if snapshot.get("semantic_kind"):
            imgui.text(f"Semantic kind: {snapshot['semantic_kind']}")

        imgui.separator()
        changed_mode, self._feature_mode_idx = imgui.combo(
            "Feature Viz Mode",
            self._feature_mode_idx,
            self._feature_modes,
        )
        if changed_mode:
            self._feature_mode_idx = int(self._feature_mode_idx)

        if self._feature_modes[self._feature_mode_idx] == "Channel":
            feature = snapshot.get("semantics")
            max_channel = 0 if feature is None else max(int(feature.shape[0]) - 1, 0)
            changed_channel, channel = imgui.slider_int("Channel", self._channel_idx, 0, max_channel)
            if changed_channel:
                self._channel_idx = channel

        changed_opacity, opacity = imgui.slider_float("Mask Opacity", self._mask_opacity, 0.0, 1.0)
        if changed_opacity:
            self._mask_opacity = opacity

        changed_overlay, overlay = imgui.checkbox("Show Mask Overlay", self._show_overlay)
        if changed_overlay:
            self._show_overlay = overlay

        gt = snapshot.get("gt")
        render = snapshot.get("render")
        features = snapshot.get("semantics")
        feature_preview_path = snapshot.get("semantic_preview_path")
        mask = snapshot.get("mask")
        error_map = snapshot.get("error_map")
        if gt is None or render is None:
            imgui.text_disabled("Missing preview images in the current snapshot.")
            return

        cache_key = (
            snapshot.get("step", 0),
            self._feature_mode_idx,
            self._channel_idx,
            self._show_overlay,
            float(self._mask_opacity),
        )
        refresh = cache_key != self._preview_cache_key
        if refresh:
            self._preview_cache_key = cache_key
            if self._show_overlay:
                render_vis = _overlay_mask(render, mask, self._mask_opacity)
            else:
                render_vis = _to_uint8_rgb(render)

            feature_vis = self._load_feature_preview(
                feature_preview_path,
                features,
                self._feature_modes[self._feature_mode_idx],
                snapshot.get("semantic_kind"),
                render_vis.shape,
            )

            error_vis = _jet_colormap(
                error_map
                if error_map is not None
                else np.mean(np.abs(np.asarray(render, dtype=np.float32) - np.asarray(gt, dtype=np.float32)), axis=-1)
            )

            self._preview_images["gt"] = _to_uint8_rgb(gt)
            self._preview_images["render"] = render_vis
            self._preview_images["features"] = feature_vis
            self._preview_images["error"] = error_vis

        gt_vis = self._preview_images["gt"]
        render_vis = self._preview_images["render"]
        feature_vis = self._preview_images["features"]
        error_vis = self._preview_images["error"]

        available_width = max(imgui.get_content_region_avail().x, 320.0)
        column_width = max(260.0, (available_width - 8.0) * 0.5)

        self._draw_image("GT", gt_vis, column_width, "gt", refresh)
        imgui.same_line()
        self._draw_image("Render", render_vis, column_width, "render", refresh)
        self._draw_image("Features", feature_vis, column_width, "features", refresh)
        imgui.same_line()
        self._draw_image("Error Map", error_vis, column_width, "error", refresh)

    def _load_feature_preview(
        self,
        preview_path: Optional[str],
        features: Optional[np.ndarray],
        mode: str,
        semantic_kind: Optional[str],
        render_shape: tuple[int, ...],
    ) -> np.ndarray:
        use_saved_preview = bool(preview_path) and (mode == "PCA" or semantic_kind == "clustered")
        if use_saved_preview and preview_path:
            cached = self._feature_preview_cache.get(preview_path)
            if cached is None:
                try:
                    cached = imageio.imread(preview_path)
                    if cached.ndim == 2:
                        cached = np.repeat(cached[..., None], 3, axis=-1)
                    if cached.shape[-1] == 4:
                        cached = cached[..., :3]
                    cached = _to_uint8_rgb(cached)
                    self._feature_preview_cache[preview_path] = cached
                except Exception:
                    cached = None
            if cached is not None:
                return cached

        if features is not None:
            return _feature_preview(features, mode, self._channel_idx)
        return np.zeros(render_shape, dtype=np.uint8)

    def _draw_image(self, label: str, image: np.ndarray, width: float, slot: str, refresh: bool):
        imgui.begin_group()
        imgui.text(label)
        immvision.image_display_resizable(
            f"{label}##spotless_{slot}",
            image,
            size=imgui.ImVec2(width, 0.0),
            refresh_image=refresh,
            resizable=True,
            show_options_button=True,
        )
        imgui.end_group()
