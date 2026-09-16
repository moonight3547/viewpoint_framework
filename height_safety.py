"""V3.2 geometry-aware local/global camera-height safety."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import warnings

import numpy as np


EPS = 1e-10
LOCAL_RELIABLE = "LOCAL_RELIABLE"
LOCAL_HOLE_UNCERTAIN = "LOCAL_HOLE_UNCERTAIN"
LOCAL_UNAVAILABLE = "LOCAL_UNAVAILABLE"


@dataclass
class LocalHeightConfig:
    enabled: bool = False
    strategy: str = "geometry_depth_up_down"
    probe_resolution: int = 64
    central_crop_ratio: float = 0.5
    alpha_threshold: float = 0.05
    hole_ratio_threshold: float = 0.5
    center_patch_size: int = 3
    center_min_valid_ratio: float = 0.5
    use_nearest_depth: bool = True

    def validate(self) -> None:
        if self.strategy != "geometry_depth_up_down":
            raise ValueError(f"Unsupported local-height strategy: {self.strategy}")
        if self.probe_resolution < 8:
            raise ValueError("local-height probe_resolution must be >= 8")
        if not 0.0 < self.central_crop_ratio <= 1.0:
            raise ValueError("central_crop_ratio must be in (0,1]")
        if not 0.0 <= self.alpha_threshold <= 1.0:
            raise ValueError("alpha_threshold must be in [0,1]")
        if not 0.0 <= self.hole_ratio_threshold <= 1.0:
            raise ValueError("hole_ratio_threshold must be in [0,1]")
        if self.center_patch_size <= 0 or self.center_patch_size % 2 == 0:
            raise ValueError("center_patch_size must be a positive odd integer")
        if not 0.0 < self.center_min_valid_ratio <= 1.0:
            raise ValueError("center_min_valid_ratio must be in (0,1]")
        if not self.use_nearest_depth:
            raise ValueError("V3.2 currently requires use_nearest_depth=true")


@dataclass
class GlobalHeightConfig:
    enabled: bool = False
    strategy: str = "strict_local_intersection"
    fallback: str = "captured_height_range"

    def validate(self) -> None:
        if self.strategy != "strict_local_intersection":
            raise ValueError(f"Unsupported global-height strategy: {self.strategy}")
        if self.fallback != "captured_height_range":
            raise ValueError(f"Unsupported global-height fallback: {self.fallback}")


@dataclass
class HeightSideProbe:
    valid_count: int
    valid_ratio: float
    hole_ratio: float
    center_valid_count: int
    center_valid_ratio: float
    center_valid: bool
    raw_limit: Optional[float]
    safe_limit: Optional[float]
    status: str


@dataclass
class LocalHeightProbeResult:
    azimuth_deg: float
    rho: float
    trajectory_height_min: float
    trajectory_height_max: float
    lower_raw: Optional[float]
    upper_raw: Optional[float]
    lower_safe: Optional[float]
    upper_safe: Optional[float]
    down_probe: HeightSideProbe
    up_probe: HeightSideProbe
    lower_status: str
    upper_status: str
    is_extension_column: bool = False


@dataclass
class GlobalHeightLimits:
    height_min: float
    height_max: float
    lower_source: str
    upper_source: str
    lower_contributor_azimuth: Optional[float]
    upper_contributor_azimuth: Optional[float]
    reliable_lower_count: int
    reliable_upper_count: int
    fallback_to_captured_lower: bool
    fallback_to_captured_upper: bool
    conflict: bool


@dataclass
class EffectiveHeightLimits:
    height_min: float
    height_max: float
    lower_source: str
    upper_source: str


@dataclass
class HeightClipResult:
    radius: float
    original_height: float
    clipped_height: float
    clipped: bool
    reachable: bool
    boundary: Optional[str]


def _crop_bounds(width: int, height: int, ratio: float):
    crop_w = max(1, int(round(width * ratio)))
    crop_h = max(1, int(round(height * ratio)))
    x0 = max(0, (width - crop_w) // 2)
    y0 = max(0, (height - crop_h) // 2)
    return x0, y0, min(width, x0 + crop_w), min(height, y0 + crop_h)


def _world_hit_heights(renderer, camera, render, center, up, config):
    depth = np.asarray(render.depth, dtype=np.float64)
    alpha = np.asarray(render.alpha, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0.0) & (alpha >= config.alpha_threshold)
    x0, y0, x1, y1 = _crop_bounds(render.width, render.height, config.central_crop_ratio)
    crop_valid = valid[y0:y1, x0:x1]
    total = int(crop_valid.size)
    count = int(np.count_nonzero(crop_valid))

    half = config.center_patch_size // 2
    cx = int(round((render.width - 1) * 0.5))
    cy = int(round((render.height - 1) * 0.5))
    px0, px1 = max(0, cx - half), min(render.width, cx + half + 1)
    py0, py1 = max(0, cy - half), min(render.height, cy + half + 1)
    center_patch = valid[py0:py1, px0:px1]
    center_count = int(np.count_nonzero(center_patch))
    center_total = max(int(center_patch.size), 1)
    center_ratio = center_count / center_total
    center_valid = center_ratio >= config.center_min_valid_ratio

    heights = np.empty((0,), dtype=np.float64)
    if count:
        ys, xs = np.nonzero(crop_valid)
        xs = xs + x0
        ys = ys + y0
        z = depth[ys, xs]
        K = renderer.scaled_intrinsics(camera, render.width, render.height).astype(np.float64)
        camera_points = np.stack(
            ((xs - K[0, 2]) * z / K[0, 0],
             (ys - K[1, 2]) * z / K[1, 1], z), axis=1,
        )
        world = camera_points @ camera.rotation_c2w.T + camera.position[None]
        heights = (world - np.asarray(center)[None]) @ np.asarray(up)
        heights = heights[np.isfinite(heights)]
    return heights, count, count / max(total, 1), 1.0 - count / max(total, 1), center_count, center_ratio, center_valid


def _side_result(values, seed_height, clearance, upward, diagnostics, config):
    count, valid_ratio, hole_ratio, center_count, center_ratio, center_valid = diagnostics
    if upward:
        eligible = values[values > seed_height + EPS]
        raw = float(np.min(eligible)) if len(eligible) else None
        safe = None if raw is None else raw - clearance
    else:
        eligible = values[values < seed_height - EPS]
        raw = float(np.max(eligible)) if len(eligible) else None
        safe = None if raw is None else raw + clearance
    if raw is None:
        status = LOCAL_UNAVAILABLE
    elif hole_ratio > config.hole_ratio_threshold and not center_valid:
        status = LOCAL_HOLE_UNCERTAIN
    else:
        status = LOCAL_RELIABLE
    return HeightSideProbe(
        valid_count=count, valid_ratio=valid_ratio, hole_ratio=hole_ratio,
        center_valid_count=center_count, center_valid_ratio=center_ratio,
        center_valid=center_valid, raw_limit=raw, safe_limit=safe, status=status,
    )


def unavailable_local_height(azimuth_deg, rho, height_min, height_max, *, extension=False):
    empty = HeightSideProbe(0, 0.0, 1.0, 0, 0.0, False, None, None, LOCAL_UNAVAILABLE)
    return LocalHeightProbeResult(
        azimuth_deg=float(azimuth_deg), rho=float(rho),
        trajectory_height_min=float(height_min), trajectory_height_max=float(height_max),
        lower_raw=None, upper_raw=None, lower_safe=None, upper_safe=None,
        down_probe=empty, up_probe=empty,
        lower_status=LOCAL_UNAVAILABLE, upper_status=LOCAL_UNAVAILABLE,
        is_extension_column=bool(extension),
    )


def probe_local_height(renderer, up_camera, down_camera, center, up,
                       azimuth_deg, rho, trajectory_height_min,
                       trajectory_height_max, clearance, config):
    config.validate()
    up_render = renderer.render_geometry_depth(up_camera, max_image_dim=config.probe_resolution)
    down_render = renderer.render_geometry_depth(down_camera, max_image_dim=config.probe_resolution)
    up_values, *up_diag = _world_hit_heights(renderer, up_camera, up_render, center, up, config)
    down_values, *down_diag = _world_hit_heights(renderer, down_camera, down_render, center, up, config)
    upper = _side_result(up_values, trajectory_height_max, clearance, True, up_diag, config)
    lower = _side_result(down_values, trajectory_height_min, clearance, False, down_diag, config)
    return LocalHeightProbeResult(
        azimuth_deg=float(azimuth_deg), rho=float(rho),
        trajectory_height_min=float(trajectory_height_min),
        trajectory_height_max=float(trajectory_height_max),
        lower_raw=lower.raw_limit, upper_raw=upper.raw_limit,
        lower_safe=lower.safe_limit, upper_safe=upper.safe_limit,
        down_probe=lower, up_probe=upper,
        lower_status=lower.status, upper_status=upper.status,
    )


def build_global_height_limits(local_results: Sequence[LocalHeightProbeResult],
                               captured_heights, config=None):
    cfg = config or GlobalHeightConfig(enabled=True)
    cfg.validate()
    captured = np.asarray(captured_heights, dtype=np.float64)
    captured = captured[np.isfinite(captured)]
    if not len(captured):
        raise ValueError("Global height requires at least one valid captured height.")
    lowers = [(r.lower_safe, r.azimuth_deg) for r in local_results
              if not r.is_extension_column and r.lower_status == LOCAL_RELIABLE
              and r.lower_safe is not None]
    uppers = [(r.upper_safe, r.azimuth_deg) for r in local_results
              if not r.is_extension_column and r.upper_status == LOCAL_RELIABLE
              and r.upper_safe is not None]
    if lowers:
        lower, lower_az = max(lowers, key=lambda x: (x[0], -x[1]))
        lower_source, fallback_lower = "strict_local_intersection", False
    else:
        lower, lower_az = float(np.min(captured)), None
        lower_source, fallback_lower = "captured_height_fallback", True
    if uppers:
        upper, upper_az = min(uppers, key=lambda x: (x[0], x[1]))
        upper_source, fallback_upper = "strict_local_intersection", False
    else:
        upper, upper_az = float(np.max(captured)), None
        upper_source, fallback_upper = "captured_height_fallback", True
    conflict = bool(lower >= upper)
    if conflict:
        warnings.warn(
            "GLOBAL_HEIGHT_INTERVAL_CONFLICT: strict local intersection is empty; "
            "falling back to captured height range.", RuntimeWarning,
        )
        lower, upper = float(np.min(captured)), float(np.max(captured))
        lower_source = upper_source = "captured_height_fallback_conflict"
        fallback_lower = fallback_upper = True
    return GlobalHeightLimits(
        height_min=float(lower), height_max=float(upper),
        lower_source=lower_source, upper_source=upper_source,
        lower_contributor_azimuth=lower_az,
        upper_contributor_azimuth=upper_az,
        reliable_lower_count=len(lowers), reliable_upper_count=len(uppers),
        fallback_to_captured_lower=fallback_lower,
        fallback_to_captured_upper=fallback_upper,
        conflict=conflict,
    )


def resolve_effective_height(local, global_limits, *, force_global=False):
    if force_global or local.is_extension_column:
        return EffectiveHeightLimits(
            global_limits.height_min, global_limits.height_max,
            "global_extension", "global_extension",
        )
    if local.lower_status == LOCAL_RELIABLE:
        lower, lower_source = local.lower_safe, "local"
    elif local.lower_status == LOCAL_HOLE_UNCERTAIN and local.lower_safe is not None:
        lower, lower_source = max(local.lower_safe, global_limits.height_min), "local_global_strict"
    else:
        lower, lower_source = global_limits.height_min, "global"
    if local.upper_status == LOCAL_RELIABLE:
        upper, upper_source = local.upper_safe, "local"
    elif local.upper_status == LOCAL_HOLE_UNCERTAIN and local.upper_safe is not None:
        upper, upper_source = min(local.upper_safe, global_limits.height_max), "local_global_strict"
    else:
        upper, upper_source = global_limits.height_max, "global"
    if lower is None or upper is None or lower > upper:
        return EffectiveHeightLimits(
            global_limits.height_min, global_limits.height_max,
            "global_effective_conflict", "global_effective_conflict",
        )
    return EffectiveHeightLimits(float(lower), float(upper), lower_source, upper_source)


def clip_radius_to_height_limit(signed_radius, direction, up, height_min, height_max):
    radius = float(signed_radius)
    vertical = float(np.dot(np.asarray(direction), np.asarray(up)))
    height = radius * vertical
    if height_min <= height <= height_max:
        return HeightClipResult(radius, height, height, False, True, None)
    boundary = "lower" if height < height_min else "upper"
    target = float(height_min if boundary == "lower" else height_max)
    if abs(vertical) <= EPS:
        return HeightClipResult(radius, height, height, False, False, boundary)
    clipped = target / vertical
    same_sign = (radius > 0 and clipped > 0) or (radius < 0 and clipped < 0)
    reachable = bool(same_sign and abs(clipped) <= abs(radius) + EPS and abs(clipped) > EPS)
    if not reachable:
        return HeightClipResult(radius, height, height, False, False, boundary)
    return HeightClipResult(float(clipped), height, float(clipped * vertical), True, True, boundary)
