"""V3.3 camera eye-pitch/FOV and circular proposal-domain analysis."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from viewpoint_framework.scene_types import CameraMode, CircularInterval
from viewpoint_framework.view_space import expand_circular_interval


@dataclass
class ViewDomainStats:
    eye_pitch_range_deg: tuple[float, float]
    effective_pitch_fov_range_deg: tuple[float, float]
    position_elevation_range_deg: tuple[float, float]
    generation_elevation_range_deg: tuple[float, float]
    generation_azimuth: CircularInterval
    azimuth_extension_deg: float
    azimuth_close_loop_applied: bool


def analyze_view_domain(cameras, observed_azimuth, mode, config):
    rows = []
    for camera in cameras:
        if not np.isfinite(camera.c2w).all():
            continue
        pitch = float(np.degrees(np.arcsin(np.clip(
            np.dot(camera.forward, config.up_axis), -1.0, 1.0,
        ))))
        fov_x = 2.0 * np.degrees(np.arctan(float(camera.width) / (2.0 * camera.fx)))
        fov_y = 2.0 * np.degrees(np.arctan(float(camera.height) / (2.0 * camera.fy)))
        half = 0.5 * max(fov_x, fov_y)
        rows.append((pitch, pitch - half, pitch + half))
    if not rows:
        raise ValueError("V3.3 view-domain analysis requires finite captured cameras.")
    values = np.asarray(rows, dtype=np.float64)
    low_q, high_q = config.elevation_percentiles
    eye = tuple(float(x) for x in np.percentile(values[:, 0], [low_q, high_q]))
    envelope = (
        float(np.percentile(values[:, 1], low_q)),
        float(np.percentile(values[:, 2], high_q)),
    )
    positional = ((-envelope[1], -envelope[0])
                  if mode == CameraMode.OUTSIDE_IN else envelope)
    span = max(0.0, positional[1] - positional[0])
    extension = max(config.elevation_extension_ratio * span,
                    config.elevation_min_extension_deg)
    elevation = (
        max(config.elevation_hard_min_deg, positional[0] - extension),
        min(config.elevation_hard_max_deg, positional[1] + extension),
    )
    az_extension = max(config.azimuth_extension_ratio * observed_azimuth.span_deg,
                       config.azimuth_min_extension_deg)
    azimuth = expand_circular_interval(observed_azimuth, az_extension)
    close_loop = 360.0 - azimuth.span_deg <= config.azimuth_close_loop_gap_deg + 1e-9
    if close_loop:
        azimuth = CircularInterval(-180.0, 180.0 - 1e-9, 360.0, False)
    return ViewDomainStats(
        eye, envelope, positional, elevation, azimuth, az_extension, close_loop,
    )
