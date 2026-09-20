"""V3.3 Stage-2 configuration groups."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class GridConfig:
    azimuth_step_deg: float = 20.0
    elevation_step_deg: float = 20.0
    azimuth_extension_ratio: float = 0.10
    azimuth_min_extension_deg: float = 20.0
    azimuth_close_loop_gap_deg: float = 90.0
    elevation_strategy: str = "eye_pitch_fov"
    elevation_percentiles: tuple[float, float] = (2.0, 98.0)
    elevation_extension_ratio: float = 0.10
    elevation_min_extension_deg: float = 10.0
    elevation_hard_min_deg: float = -80.0
    elevation_hard_max_deg: float = 80.0
    up_axis: np.ndarray = field(default_factory=lambda: np.array([0., 1., 0.]))

    def validate(self):
        if self.elevation_strategy != "eye_pitch_fov":
            raise ValueError("V3.3 requires elevation_strategy='eye_pitch_fov'.")
        if self.azimuth_step_deg <= 0 or self.elevation_step_deg <= 0:
            raise ValueError("Angular steps must be positive.")
        if self.elevation_hard_min_deg >= self.elevation_hard_max_deg:
            raise ValueError("Invalid elevation hard bounds.")


@dataclass
class RadiusConfig:
    center_guard_clearance_ratio: float = 1.0
    nominal_captured_max_ratio: float = 2.0
    over_nominal_strategy: str = "one_third_depth_extension"
    emergency_strategy: str = "skybox_or_geometry_q99"

    def validate(self):
        if self.center_guard_clearance_ratio <= 0 or self.nominal_captured_max_ratio <= 0:
            raise ValueError("Radius ratios must be positive.")
        if self.over_nominal_strategy != "one_third_depth_extension":
            raise ValueError("Unsupported V3.3 radius extension strategy.")


@dataclass
class RendererConfig:
    backend: str = "auto"
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    clamp_color_min: bool = False
    render_pano_depths: bool = False

    def validate(self):
        if self.backend not in ("auto", "gs_render", "gsplat"):
            raise ValueError("renderer.backend must be auto, gs_render, or gsplat")
        if self.clamp_color_min:
            raise ValueError("V3.3 requires clamp_color_min=false")
