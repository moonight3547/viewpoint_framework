"""V3.3 robust local height and coverage-consensus global height."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import numpy as np

from viewpoint_framework.height_safety import (
    LOCAL_HOLE_UNCERTAIN, LOCAL_RELIABLE, LOCAL_UNAVAILABLE,
    EffectiveHeightLimits, HeightClipResult, clip_radius_to_height_limit,
)


@dataclass
class RobustSide:
    status: str
    raw_limit: float | None
    safe_limit: float | None
    q10_distance: float | None
    center_min_distance: float | None
    selected_distance: float | None
    valid_ratio: float
    hole_ratio: float
    center_valid: bool


@dataclass
class LocalHeightV33:
    azimuth_deg: float
    rho: float
    h_cross: float | None
    probe_origin: list[float] | None
    lower: RobustSide
    upper: RobustSide
    source: str

    @property
    def reliable_interval(self):
        if (self.lower.status == LOCAL_RELIABLE and self.upper.status == LOCAL_RELIABLE
                and self.lower.safe_limit is not None and self.upper.safe_limit is not None
                and self.lower.safe_limit < self.upper.safe_limit):
            return float(self.lower.safe_limit), float(self.upper.safe_limit)
        return None


@dataclass
class GlobalHeightV33:
    height_min: float
    height_max: float
    strategy: str
    support_ratio: float
    reliable_count: int
    required_count: int
    consensus_band_count: int
    consensus_ambiguous: bool
    bands: list[tuple[float, float]]


def unavailable_local(azimuth, rho, source, h_cross=None, origin=None):
    side = RobustSide(LOCAL_UNAVAILABLE, None, None, None, None, None, 0., 1., False)
    return LocalHeightV33(float(azimuth), float(rho), h_cross,
                          None if origin is None else np.asarray(origin).tolist(),
                          side, side, source)


def _render_distances(renderer, camera, render, center, up, h_cross, config):
    depth = np.asarray(render.depth, dtype=np.float64)
    alpha = np.asarray(render.alpha, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0) & (alpha >= config.alpha_threshold)
    ratio = float(config.central_crop_ratio)
    cw, ch = max(1, round(render.width * ratio)), max(1, round(render.height * ratio))
    x0, y0 = (render.width - cw) // 2, (render.height - ch) // 2
    crop = valid[y0:y0 + ch, x0:x0 + cw]
    ys, xs = np.nonzero(crop)
    xs, ys = xs + x0, ys + y0
    heights = np.empty(0, dtype=np.float64)
    if len(xs):
        z = depth[ys, xs]
        k = renderer.scaled_intrinsics(camera, render.width, render.height).astype(np.float64)
        cam = np.stack(((xs-k[0, 2])*z/k[0, 0], (ys-k[1, 2])*z/k[1, 1], z), axis=1)
        world = cam @ camera.rotation_c2w.T + camera.position[None]
        heights = (world - np.asarray(center)[None]) @ np.asarray(up)
    half = config.center_patch_size // 2
    cx, cy = round((render.width-1)*.5), round((render.height-1)*.5)
    px0, px1 = max(0, cx-half), min(render.width, cx+half+1)
    py0, py1 = max(0, cy-half), min(render.height, cy+half+1)
    patch_valid = valid[py0:py1, px0:px1]
    pys, pxs = np.nonzero(patch_valid)
    center_heights = np.empty(0, dtype=np.float64)
    if len(pxs):
        pxs, pys = pxs + px0, pys + py0
        z = depth[pys, pxs]
        k = renderer.scaled_intrinsics(camera, render.width, render.height).astype(np.float64)
        cam = np.stack(((pxs-k[0, 2])*z/k[0, 0], (pys-k[1, 2])*z/k[1, 1], z), axis=1)
        world = cam @ camera.rotation_c2w.T + camera.position[None]
        center_heights = (world - np.asarray(center)[None]) @ np.asarray(up)
    return (np.abs(heights[np.isfinite(heights)] - h_cross),
            np.abs(center_heights[np.isfinite(center_heights)] - h_cross),
            float(np.count_nonzero(crop)) / max(crop.size, 1))


def _side(distances, center_distances, valid_ratio, h_cross, clearance, upward, config):
    hole_ratio = 1.0 - valid_ratio
    center_valid = bool(len(center_distances))
    if not len(distances):
        return RobustSide(LOCAL_UNAVAILABLE, None, None, None, None, None,
                          valid_ratio, hole_ratio, center_valid)
    q10 = float(np.quantile(distances, config.depth_quantile))
    center_min = float(np.min(center_distances)) if center_valid else None
    selected = min(q10, center_min) if center_min is not None else q10
    raw = h_cross + selected if upward else h_cross - selected
    safe = raw - clearance if upward else raw + clearance
    status = (LOCAL_HOLE_UNCERTAIN
              if hole_ratio > config.hole_ratio_threshold and not center_valid
              else LOCAL_RELIABLE)
    return RobustSide(status, float(raw), float(safe), q10, center_min, selected,
                      valid_ratio, hole_ratio, center_valid)


def probe_local_height(renderer, up_camera, down_camera, center, up, azimuth,
                       rho, h_cross, clearance, config):
    up_render = renderer.render_geometry_depth(up_camera, max_image_dim=config.probe_resolution)
    down_render = renderer.render_geometry_depth(down_camera, max_image_dim=config.probe_resolution)
    ud, uc, ur = _render_distances(renderer, up_camera, up_render, center, up, h_cross, config)
    dd, dc, dr = _render_distances(renderer, down_camera, down_render, center, up, h_cross, config)
    return LocalHeightV33(
        float(azimuth), float(rho), float(h_cross), up_camera.position.tolist(),
        _side(dd, dc, dr, h_cross, clearance, False, config),
        _side(ud, uc, ur, h_cross, clearance, True, config), "local_p_traj",
    )


def coverage_consensus(local_results, captured_heights, support_ratio=.85, min_reliable=5):
    captured = np.asarray(captured_heights, dtype=np.float64)
    captured = captured[np.isfinite(captured)]
    if not len(captured):
        raise ValueError("Global height requires captured heights.")
    intervals = [r.reliable_interval for r in local_results if r.reliable_interval is not None]
    n = len(intervals)
    median = float(np.median(captured))
    if n < min_reliable:
        if n:
            lo, hi = max(x[0] for x in intervals), min(x[1] for x in intervals)
            if lo < hi:
                return GlobalHeightV33(lo, hi, "strict_small_n", support_ratio, n, n, 1, False, [(lo, hi)])
        return GlobalHeightV33(float(captured.min()), float(captured.max()),
                               "captured_fallback", support_ratio, n, n, 0, False, [])
    required = ceil(support_ratio * n)
    endpoints = sorted({v for interval in intervals for v in interval})
    segments = []
    for lo, hi in zip(endpoints[:-1], endpoints[1:]):
        mid = .5 * (lo + hi)
        count = sum(a <= mid <= b for a, b in intervals)
        if count >= required:
            if segments and abs(segments[-1][1] - lo) <= 1e-10:
                segments[-1] = (segments[-1][0], hi)
            else:
                segments.append((lo, hi))
    strategy = "coverage_consensus"
    bands = segments
    if not bands:
        median_segments = []
        for lo, hi in zip(endpoints[:-1], endpoints[1:]):
            midpoint = .5*(lo+hi)
            support = sum(a <= midpoint <= b for a, b in intervals)
            if lo <= median <= hi and support > 0:
                median_segments.append((support, lo, hi))
        if median_segments:
            _, lo, hi = max(median_segments, key=lambda x: (x[0], x[2]-x[1]))
            bands, strategy = [(lo, hi)], "median_segment_fallback"
        else:
            return GlobalHeightV33(float(captured.min()), float(captured.max()),
                                   "captured_fallback", support_ratio, n, required, 0, False, [])
    containing = [b for b in bands if b[0] <= median <= b[1]]
    selected = containing[0] if containing else min(
        bands, key=lambda b: min(abs(median-b[0]), abs(median-b[1])))
    return GlobalHeightV33(selected[0], selected[1], strategy, support_ratio,
                           n, required, len(bands), len(bands) > 1, bands)


def effective_height(local, global_height):
    interval = local.reliable_interval
    if interval is None:
        return EffectiveHeightLimits(global_height.height_min, global_height.height_max,
                                     "global", "global")
    return EffectiveHeightLimits(interval[0], interval[1], "local", "local")


__all__ = ["probe_local_height", "coverage_consensus", "effective_height",
           "unavailable_local", "clip_radius_to_height_limit", "HeightClipResult"]
