"""Small, opt-in V3.0 placement path: horizontal prior + positional elevation.

Initial collisions are rejected. Valid initial positions may undergo bounded
V2-style radial adjustment, including crossing the actual 3D scene center.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np

from viewpoint_framework.geometry_safety import PointCloudSafety, local_clearance_threshold
from viewpoint_framework.pose_generation import (
    CandidateStatus, GeneratedCandidate, PoseGenerationResult, build_generated_camera,
)
from viewpoint_framework.scene_types import CameraMode, to_jsonable
from viewpoint_framework.trajectory_safe_field import TrajectorySafeField
from viewpoint_framework.view_space import direction_to_azimuth_elevation

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
    """V2 radial policy before upper-bound clipping (V3 skips overshoots)."""
    if strategy == "prior_only":
        return initial_radius
    move = max(0.0, depth - clearance * depth_margin_ratio)
    if mode == CameraMode.OUTSIDE_IN:
        return initial_radius + move
    # A global dominant bbox must not move a smaller, trajectory-supported prior
    # outward when an inward adjustment cannot reach the bbox lower shell.
    lower = min(initial_radius, max(minimum_radius, EPS))
    if strategy == "center_crossing_depth":
        opposite = move - initial_radius
        if opposite >= lower + clearance * center_cross_extra_ratio:
            return -opposite
    return max(lower, initial_radius - move)


def _validate_config(config, points):
    if points is None or config.geometry.strategy != "pointcloud_knn":
        raise ValueError("V3.0 requires point-cloud geometry and pointcloud_knn safety.")
    if config.geometry.clearance_strategy != "local_horizontal_radius":
        raise ValueError("V3.0 requires clearance_strategy='local_horizontal_radius'.")
    if not config.reject_unsafe_initial_prior:
        raise ValueError("V3.0 requires reject_unsafe_initial_prior=true; recovery is deferred.")
    if config.invalid_depth_behavior != "keep_prior":
        raise ValueError("V3.0 invalid_depth_behavior must be keep_prior.")
    if config.outside_placement_strategy not in ("prior_only", "depth_backoff"):
        raise ValueError("Unsupported V3 outside placement.")
    if config.inside_placement_strategy not in ("prior_only", "no_crossing", "center_crossing_depth"):
        raise ValueError("Unsupported V3 inside placement.")
    for value in (config.depth_margin_ratio, config.center_cross_extra_ratio, config.height_guard.margin_ratio):
        if not np.isfinite(value) or value < 0:
            raise ValueError("V3 margins must be finite and nonnegative.")
    if (not np.isfinite(config.geometry.path_step_ratio) or config.geometry.path_step_ratio <= 0
            or config.geometry.max_path_samples < 2):
        raise ValueError("Invalid path sampling parameters.")


def generate_v3_candidates(captured_cameras, profile, bbox, view_limits, grid,
                           mode_result, point_cloud_points, depth_probe, config):
    _validate_config(config, point_cloud_points)
    frame = profile.coordinate_frame
    center = np.asarray(profile.center_fit.center, dtype=np.float64)
    trajectory = TrajectorySafeField(captured_cameras, center, frame, config.trajectory_safe_field)
    safety = PointCloudSafety(point_cloud_points, config.geometry)
    radius_max = (float(config.adjustment_radius_max) if config.adjustment_radius_max is not None
                  else float(max(bbox.generation_radius)))
    if not np.isfinite(radius_max) or radius_max <= 0:
        raise ValueError("adjustment_radius_max must be finite and positive.")
    radius_min = float(min(bbox.generation_radius))
    estimates = {az: trajectory.query(az) for az in sorted({p.azimuth_deg for p in grid})}
    cameras, candidates = [], []
    for point in grid:
        estimate = estimates[point.azimuth_deg]
        interval = estimate.selected_interval
        strategy = (config.outside_placement_strategy if mode_result.mode == CameraMode.OUTSIDE_IN
                    else config.inside_placement_strategy)
        meta = {
            "version": "3.0", "elevation_semantics": "position",
            "grid_azimuth_deg": point.azimuth_deg, "grid_elevation_deg": point.elevation_deg,
            "trajectory_branch_id": interval.branch_id if interval else None,
            "horizontal_safe_intervals": to_jsonable(estimate.intervals),
            "nearest_support_angle_deg": estimate.nearest_support_angle_deg,
            "selected_horizontal_radius": interval.rho_preferred if interval else None,
            "local_clearance_threshold": None, "proposal_safe": None,
            "height_guard_enabled": config.height_guard.enabled, "height_guard_pass": None,
            "adjustment_radius_max": radius_max, "adjustment_radius_exceeded": False,
            "initial_radius_exceeds_adjustment_max": False,
            "adjustment_attempted": False, "adjustment_applied": False,
            "adjustment_distance": 0.0, "adjustment_skip_reason": None,
            "proposed_signed_radius": None, "final_geometry_safe": False,
        }
        candidate = GeneratedCandidate(
            grid_id=point.grid_id, row=point.row, col=point.col,
            azimuth_deg=point.azimuth_deg, elevation_deg=point.elevation_deg,
            direction=point.direction.copy(), mode=mode_result.mode,
            initial_radius=0.0, initial_radius_source=estimate.source,
            initial_radius_confidence=estimate.confidence, final_signed_radius=0.0,
            crossed_center=False, placement_strategy=strategy, depth_probe=None,
            initial_clearance=None, final_clearance=None, path_safe_fraction=None,
            camera=None, status=CandidateStatus.REJECTED, geometry_metadata=meta,
        )
        candidates.append(candidate)
        if interval is None:
            candidate.reject_reason = estimate.source
            continue
        az = np.radians(point.azimuth_deg)
        horizontal = np.sin(az) * frame.x_axis + np.cos(az) * frame.z_axis
        try:
            initial, r0, direction = initial_position_from_rho(
                center, horizontal, frame.y_axis, interval.rho_preferred, point.elevation_deg,
            )
        except ValueError:
            candidate.reject_reason = "INVALID_POSITION_ELEVATION"
            continue
        candidate.initial_radius = r0
        candidate.final_signed_radius = r0
        meta["initial_position"] = initial.tolist()
        meta["initial_radius_exceeds_adjustment_max"] = r0 > radius_max
        threshold = local_clearance_threshold(interval.rho_preferred, trajectory.median_captured_rho, config.geometry)
        meta["local_clearance_threshold"] = threshold
        safe, candidate.initial_clearance = safety.is_position_safe(initial, threshold)
        meta["proposal_safe"] = bool(safe and np.isfinite(candidate.initial_clearance))
        if not meta["proposal_safe"]:
            candidate.reject_reason = "UNSAFE_INITIAL_PRIOR"
            continue
        initial_height = float(np.dot(initial - center, frame.y_axis))
        if config.height_guard.enabled and not config.height_guard.allows(initial_height, interval):
            meta["height_guard_pass"] = False
            candidate.reject_reason = "HEIGHT_OUT_OF_TRAJECTORY_RANGE"
            continue

        final, signed = initial.copy(), r0
        if strategy != "prior_only" and r0 <= radius_max:
            meta["adjustment_attempted"] = True
            probe_forward = direction if mode_result.mode == CameraMode.OUTSIDE_IN else -direction
            probe_camera = build_generated_camera(-1, initial, probe_forward, captured_cameras, profile, config)
            depth = depth_probe.probe(probe_camera)
            usable = bool(depth.valid and np.isfinite(depth.depth) and depth.depth > 0)
            if not usable:
                # Invalid probes do not prevent a geometry-checked prior surviving.
                depth = replace(depth, valid=False, **{
                    key: (float(getattr(depth, key)) if np.isfinite(getattr(depth, key)) else 0.0)
                    for key in ("depth", "confidence", "valid_ratio", "depth_q10", "depth_median", "depth_mean")
                })
                meta["adjustment_skip_reason"] = "INVALID_DEPTH_PROBE"
            candidate.depth_probe = depth
            if usable:
                proposed = propose_signed_radius(
                    r0, depth.depth, mode_result.mode, strategy, radius_min, threshold,
                    config.depth_margin_ratio, config.center_cross_extra_ratio,
                )
                meta["proposed_signed_radius"] = proposed
                if not np.isfinite(proposed) or abs(proposed) > radius_max:
                    meta["adjustment_radius_exceeded"] = True
                    meta["adjustment_skip_reason"] = "ADJUSTMENT_EXCEEDS_RADIUS_MAX"
                else:
                    final = center + proposed * direction
                    if config.use_path_safety:
                        path = safety.safe_path_fraction(initial, final, threshold)
                        candidate.path_safe_fraction = path.safe_fraction
                        final = initial + path.safe_fraction * (final - initial)
                    signed = float(np.dot(final - center, direction))
                    # Keep V2's lower shell intent, but never clip a checked point
                    # into an unchecked location (especially near the center).
                    if abs(signed) < min(r0, max(radius_min, EPS)):
                        final, signed = initial.copy(), r0
                        meta["adjustment_skip_reason"] = "PATH_CLIPPED_BELOW_MIN_RADIUS"
        elif strategy != "prior_only":
            meta["adjustment_skip_reason"] = "INITIAL_EXCEEDS_ADJUSTMENT_RADIUS_MAX"

        candidate.final_signed_radius = signed
        candidate.crossed_center = signed < 0
        meta["final_position"] = final.tolist()
        meta["adjustment_distance"] = float(np.linalg.norm(final - initial))
        meta["adjustment_applied"] = meta["adjustment_distance"] > EPS
        meta["position_height"] = float(np.dot(final - center, frame.y_axis))
        height_ok = config.height_guard.allows(meta["position_height"], interval) if config.height_guard.enabled else None
        meta["height_guard_pass"] = height_ok
        final_safe, candidate.final_clearance = safety.is_position_safe(final, threshold)
        meta["final_geometry_safe"] = bool(final_safe and np.isfinite(candidate.final_clearance))
        if height_ok is False:
            candidate.reject_reason = "HEIGHT_OUT_OF_TRAJECTORY_RANGE"
            continue
        if not meta["final_geometry_safe"]:
            candidate.reject_reason = "FINAL_POSITION_TOO_CLOSE_TO_GEOMETRY"
            continue
        radial = final - center
        length = float(np.linalg.norm(radial))
        if not np.isfinite(radial).all() or length <= EPS:
            candidate.reject_reason = "INVALID_VIEW_ORIENTATION"
            continue
        radial /= length
        forward = -radial if mode_result.mode == CameraMode.OUTSIDE_IN else radial
        camera = build_generated_camera(len(cameras), final, forward, captured_cameras, profile, config)
        if not np.isfinite(camera.c2w).all():
            candidate.reject_reason = "INVALID_VIEW_ORIENTATION"
            continue
        # Stage3 consumes these angles/direction. After crossing, advertise the
        # actual radial direction, retaining the original grid identity in meta.
        candidate.direction = radial
        candidate.azimuth_deg, candidate.elevation_deg = direction_to_azimuth_elevation(radial, frame)
        meta["position_elevation_deg"] = candidate.elevation_deg
        candidate.camera = camera
        candidate.status = CandidateStatus.VALID
        cameras.append(camera)

    reasons = Counter(c.reject_reason for c in candidates if c.reject_reason)
    diagnostics = {
        "version": "3.0", "elevation_semantics": "position", "angular_grid_count": len(grid),
        "trajectory_branch_count": trajectory.branch_count,
        "trajectory_jump_rejected_count": trajectory.jump_rejected_count,
        "trajectory_invalid_pose_count": trajectory.invalid_pose_count,
        "trajectory_sample_count": len(trajectory.samples),
        "trajectory_typical_step": trajectory.typical_step,
        "median_captured_horizontal_radius": trajectory.median_captured_rho,
        "trajectory_supported_grid_count": sum(estimates[p.azimuth_deg].selected_interval is not None for p in grid),
        "trajectory_unsupported_grid_count": sum(estimates[p.azimuth_deg].selected_interval is None for p in grid),
        "multi_interval_grid_count": sum(len(estimates[p.azimuth_deg].intervals) > 1 for p in grid),
        "initial_geometry_unsafe_count": reasons["UNSAFE_INITIAL_PRIOR"],
        "height_guard_rejected_count": reasons["HEIGHT_OUT_OF_TRAJECTORY_RANGE"],
        "depth_probe_failed_count": sum(c.depth_probe is not None and not c.depth_probe.valid for c in candidates),
        "adjustment_radius_exceeded_count": sum(c.geometry_metadata["adjustment_radius_exceeded"] for c in candidates),
        "adjustment_applied_count": sum(c.geometry_metadata["adjustment_applied"] for c in candidates),
        "final_geometry_rejected_count": reasons["FINAL_POSITION_TOO_CLOSE_TO_GEOMETRY"],
        "valid_candidate_count": len(cameras), "rejection_reasons": dict(reasons),
    }
    if config.console_log_candidates:
        for c in candidates:
            m = c.geometry_metadata
            print(f"[S2:V3_CANDIDATE] grid={c.grid_id} az={m['grid_azimuth_deg']:.3f} "
                  f"el={m['grid_elevation_deg']:.3f} branch={m['trajectory_branch_id']} "
                  f"rho={m['selected_horizontal_radius']} r0={c.initial_radius:.4f} "
                  f"r_final={c.final_signed_radius:.4f} clr_req={m['local_clearance_threshold']} "
                  f"clr_init={c.initial_clearance} clr_final={c.final_clearance} "
                  f"adjust_skip={m['adjustment_skip_reason']} status={c.status.value} reject={c.reject_reason}")
        print(f"[S2:V3_FUNNEL] {diagnostics}")
    return PoseGenerationResult(mode_result, bbox, view_limits, candidates, cameras, config, diagnostics)
