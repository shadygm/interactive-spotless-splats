from dataclasses import dataclass, field


@dataclass
class RenderSettings:
    """Mutable render settings that trigger VBO rebuilds on change."""

    frustum_color: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    frustum_size: float = 0.001  # fraction of scene diagonal
    show_frustums: bool = True
    point_size: float = 1.0
    _frustum_version: int = 0

    # Splat render settings
    sh_degree: int = 0  # 0 = SH0 only, 1..3 = higher SH bands
    render_mode: str = "RGB"  # "RGB" or "Depth"
    depth_colormap: str = "viridis"  # viridis, plasma, jet, hot, cool, turbo
    near_plane: float = 0.01
    far_plane: float = 1000.0
    background_color: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rasterize_mode: str = "classic"  # classic, antialiased
    eps2d: float = 0.0
    radius_clip: float = 0.0  # skip Gaussians with 2D radius <= this (pixels)

    # Camera settings
    camera_mode: str = "orbit"  # "orbit" or "fps"

    def bump_frustum(self):
        self._frustum_version += 1

    def get_frustum_version(self):
        return self._frustum_version
