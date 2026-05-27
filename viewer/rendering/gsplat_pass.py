import time
import numpy as np
import torch
from OpenGL.GL import *
from loguru import logger

from gsplat import rasterization

from viewer.rendering.colormaps import _COLORMAPS, _COLORMAP_VIRIDIS


class GsplatPass:
    """Owns the gsplat rasterization bridge."""

    def __init__(self, device="cuda"):
        self.device = device
        # Cache colormap LUTs as GPU tensors for fast depth rendering
        self._colormap_cache = {}

    def render(self, camera, gaussians, width, height, render_settings, texture):
        """Render Gaussians via gsplat and upload to OpenGL texture."""
        if gaussians is None:
            logger.debug("[GSPLAT] gaussians is None, skipping render")
            return False

        t0 = time.perf_counter()
        means = gaussians["means"]
        quats = gaussians["quats"]
        scales = gaussians["scales"]
        opacities = gaussians["opacities"]
        colors = gaussians["colors"]
        ply_sh_degree = gaussians.get("sh_degree", 0)

        # Apply render settings overrides (clamp to available SH in PLY)
        rs = render_settings
        if rs:
            sh_degree = min(rs.sh_degree, ply_sh_degree)
        else:
            sh_degree = ply_sh_degree

        user_mode = rs.render_mode if rs else "RGB"
        # Map user-facing mode to gsplat internal render_mode
        gsplat_render_mode = "D" if user_mode == "Depth" else "RGB"
        near_plane = rs.near_plane if rs else 0.01
        far_plane = rs.far_plane if rs else 1000.0
        bg_color = rs.background_color if rs else [0.0, 0.0, 0.0]
        rasterize_mode = rs.rasterize_mode if rs else "classic"
        eps2d = rs.eps2d if rs else 0.3
        radius_clip = rs.radius_clip if rs else 0.0
        depth_cmap = rs.depth_colormap if rs else "viridis"

        # Cheap debug logs (no GPU sync)
        logger.debug(f"[GSPLAT] Device: {self.device}, torch.cuda.is_available(): {torch.cuda.is_available()}")
        logger.debug(f"[GSPLAT] means device={means.device}, shape={means.shape}, dtype={means.dtype}")
        logger.debug(f"[GSPLAT] quats device={quats.device}, shape={quats.shape}")
        logger.debug(f"[GSPLAT] scales device={scales.device}, shape={scales.shape}")
        logger.debug(f"[GSPLAT] opacities device={opacities.device}, shape={opacities.shape}")
        logger.debug(f"[GSPLAT] colors device={colors.device}, shape={colors.shape}")
        logger.debug(f"[GSPLAT] ply_sh_degree={ply_sh_degree}, effective_sh_degree={sh_degree}")
        logger.debug(f"[GSPLAT] user_mode={user_mode}, gsplat_mode={gsplat_render_mode}, near={near_plane}, far={far_plane}, bg={bg_color}")
        logger.debug(f"[GSPLAT] rasterize_mode={rasterize_mode}, eps2d={eps2d}, depth_cmap={depth_cmap}")

        c2w = camera.get_view_matrix()

        # gsplat expects OpenCV/COLMAP convention:
        #   camera looks down +Z and Y points down in camera space.
        # Our camera uses OpenGL convention (looks down -Z, Y up).
        # Flip both Y and Z to match.
        c2w_gsplat = c2w.copy()
        c2w_gsplat[:3, 1] *= -1
        c2w_gsplat[:3, 2] *= -1

        viewmat = np.linalg.inv(c2w_gsplat).astype(np.float32)
        viewmat_t = torch.from_numpy(viewmat).to(self.device)[None, :, :]

        K = camera.get_K()
        K_t = torch.from_numpy(K).to(self.device)[None, :, :]

        cam_pos = c2w[:3, 3]
        logger.debug(f"[GSPLAT] Camera position: {cam_pos}")
        logger.debug(f"[GSPLAT] Render resolution: {width}x{height}")

        bg = torch.tensor(bg_color, device=self.device, dtype=torch.float32)[None, :]
        t1 = time.perf_counter()

        with torch.no_grad():
            render_colors, render_alphas, meta = rasterization(
                means, quats, scales, opacities, colors,
                viewmat_t, K_t, width, height,
                sh_degree=sh_degree,
                render_mode=gsplat_render_mode,
                backgrounds=bg,
                near_plane=near_plane,
                far_plane=far_plane,
                rasterize_mode=rasterize_mode,
                eps2d=eps2d,
                radius_clip=radius_clip,
                packed=False,
            )
        t2 = time.perf_counter()

        rc = render_colors[0]
        ra = render_alphas[0]

        # Convert gsplat output to RGBA on GPU, then copy once to CPU.
        if user_mode == "Depth":
            # Depth: keep everything on GPU to avoid multiple CPU syncs.
            depth = rc[..., 0] if rc.ndim > 2 else rc
            dmin = depth.min()
            dmax = depth.max()
            drange = dmax - dmin
            logger.debug(f"[GSPLAT] Depth raw min/max/mean: {dmin.item():.4f}/{dmax.item():.4f}/{depth.mean().item():.4f}")
            if drange > 1e-6:
                depth_norm = (depth - dmin) / drange
            else:
                depth_norm = torch.zeros_like(depth)
            depth_norm = torch.clamp(depth_norm, 0.0, 1.0)

            # Apply colormap on GPU via cached LUT tensor
            if depth_cmap not in self._colormap_cache:
                self._colormap_cache[depth_cmap] = torch.from_numpy(
                    _COLORMAPS.get(depth_cmap, _COLORMAP_VIRIDIS)
                ).to(self.device)
            lut_t = self._colormap_cache[depth_cmap]
            indices = (depth_norm * 255).to(torch.long)
            rgb = lut_t[indices]  # HxWx3, on GPU
            rgb_uint8 = rgb.to(torch.uint8)

            alpha = torch.clamp(ra, 0.0, 1.0)
            if alpha.ndim == 2:
                alpha = alpha.unsqueeze(-1)
            alpha_uint8 = (alpha * 255.0).to(torch.uint8)

            # Stack on GPU, then single CPU copy.
            rgba_t = torch.cat([rgb_uint8, alpha_uint8], dim=-1)
            rgba = rgba_t.cpu().numpy()
        else:
            # Pure RGB: do everything on GPU to avoid multiple CPU syncs.
            rgb = torch.clamp(rc, 0.0, 1.0)
            if rgb.ndim == 2:
                rgb = rgb.unsqueeze(-1).expand(-1, -1, 3)
            rgb_uint8 = (rgb * 255.0).to(torch.uint8)

            alpha = torch.clamp(ra, 0.0, 1.0)
            if alpha.ndim == 2:
                alpha = alpha.unsqueeze(-1)
            alpha_uint8 = (alpha * 255.0).to(torch.uint8)

            # Stack on GPU, then single CPU copy.
            rgba_t = torch.cat([rgb_uint8, alpha_uint8], dim=-1)
            rgba = rgba_t.cpu().numpy()

        t3 = time.perf_counter()

        texture.bind()
        # Use glTexSubImage2D when the texture size hasn't changed to avoid
        # reallocation overhead.
        if width == texture.width and height == texture.height:
            texture.sub_image(width, height, rgba)
        else:
            texture.image(width, height, rgba)
        texture.unbind()

        t4 = time.perf_counter()

        logger.debug(
            f"gsplat breakdown (ms): prep={(t1-t0)*1000:.2f} raster={(t2-t1)*1000:.2f} "
            f"cpu_copy={(t3-t2)*1000:.2f} gl_upload={(t4-t3)*1000:.2f} total={(t4-t0)*1000:.2f}"
        )
        return True
