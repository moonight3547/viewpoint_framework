"""Shared radius-placement primitives used by the V3.2 and V3.3 pipelines."""
from __future__ import annotations

import numpy as np

from viewpoint_framework.scene_types import CameraMode


EPS = 1e-10


def initial_position_from_rho(center, horizontal_direction, up, rho, elevation_deg):
    """Keep horizontal distance rho: r=rho/cos(el), h=rho*tan(el)."""
    if not np.isfinite(elevation_deg) or abs(elevation_deg) >= 89.9:
        raise ValueError("Positional elevation must be strictly inside (-89.9, 89.9).")
    if not np.isfinite(rho) or rho <= EPS:
        raise ValueError("Horizontal radius must be finite and positive.")
    elevation = np.radians(elevation_deg)
    direction = np.cos(elevation) * horizontal_direction + np.sin(elevation) * up
    radius = float(rho / np.cos(elevation))
    return np.asarray(center) + radius * direction, radius, direction


def propose_signed_radius(initial_radius, depth, mode, strategy, minimum_radius,
                          clearance, depth_margin_ratio, center_cross_extra_ratio):
    """V3.2 radial policy before upper-bound clipping."""
    if strategy == "prior_only":
        return initial_radius
    move = max(0.0, depth - clearance * depth_margin_ratio)
    if mode == CameraMode.OUTSIDE_IN:
        return initial_radius + move
    lower = min(initial_radius, max(minimum_radius, EPS))
    if strategy == "center_crossing_depth":
        opposite = move - initial_radius
        if opposite >= lower + clearance * center_cross_extra_ratio:
            return -opposite
    return max(lower, initial_radius - move)


__all__ = ["initial_position_from_rho", "propose_signed_radius"]
