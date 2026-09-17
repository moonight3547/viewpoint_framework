"""V3.2 placement: segment-ray rho plus local/global height safety."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np

from viewpoint_framework.geometry_safety import PointCloudSafety, local_clearance_threshold
from viewpoint_framework.height_safety import (
    LOCAL_HOLE_UNCERTAIN,
    LOCAL_RELIABLE,
    LOCAL_UNAVAILABLE,
    build_global_height_limits,
    clip_radius_to_height_limit,
    probe_local_height,
    resolve_effective_height,
    unavailable_local_height,
)
from viewpoint_framework.pose_generation import (
    CandidateStatus, GeneratedCandidate, PoseGenerationResult, build_generated_camera,
)
from viewpoint_framework.pose_generation_v3 import initial_position_from_rho, propose_signed_radius
from viewpoint_framework.scene_types import CameraMode, to_jsonable
from viewpoint_framework.trajectory_safe_field import TrajectorySafeField
from viewpoint_framework.view_space import direction_to_azimuth_elevation


EPS = 1e-10


def _interval_contains(interval, azimuth_deg):
    delta = (float(azimuth_deg) - float(interval.start_deg)) % 360.0
    return bool(delta <= float(interval.span_deg) + 1e-8)


def _validate(config, points, depth_probe):
    if points is None or config.geometry.strategy != "pointcloud_knn":
        raise ValueError("V3.2 requires point-cloud geometry and pointcloud_knn safety.")
    if config.geometry.clearance_strategy != "local_horizontal_radius":
        raise ValueError("V3.2 requires clearance_strategy='local_horizontal_radius'.")
    if not config.reject_unsafe_initial_prior:
        raise ValueError("V3.2 requires reject_unsafe_initial_prior=true.")
    if config.trajectory_safe_field.rho_strategy != "segment_ray_min":
        raise ValueError("V3.2 requires rho_strategy='segment_ray_min'.")
    if not config.local_height.enabled or not config.global_height.enabled:
        raise ValueError("V3.2 requires local_height and global_height enabled.")
    config.local_height.validate()
    config.global_height.validate()
    renderer = getattr(depth_probe, "renderer", None)
    if renderer is None or not hasattr(renderer, "render_geometry_depth"):
        raise ValueError("V3.2 height safety requires a geometry-only Gaussian renderer.")
    return renderer


def _height_source(limits):
    return f"lower:{limits.lower_source}|upper:{limits.upper_source}"


def resolve_final_camera_forward(mode, radial_direction, grid_direction):
    """Apply the collection semantics after radial placement.

    Outside-in always faces scene center.  Inside-out preserves the sampled
    grid direction: it faces away on the original side, and therefore faces
    scene center after crossing to the opposite side.
    """
    radial = np.asarray(radial_direction, dtype=np.float64)
    grid = np.asarray(grid_direction, dtype=np.float64)
    return -radial if mode == CameraMode.OUTSIDE_IN else grid


def generate_v32_candidates(captured_cameras, profile, bbox, view_limits, grid,
                            mode_result, point_cloud_points, depth_probe, config):
    renderer = _validate(config, point_cloud_points, depth_probe)
    frame = profile.coordinate_frame
    center = np.asarray(profile.center_fit.center, dtype=np.float64)
    up = np.asarray(frame.y_axis, dtype=np.float64)
    trajectory = TrajectorySafeField(captured_cameras, center, frame, config.trajectory_safe_field)
    safety = PointCloudSafety(point_cloud_points, config.geometry)
    radius_max = (float(config.adjustment_radius_max) if config.adjustment_radius_max is not None
                  else float(max(bbox.generation_radius)))
    if not np.isfinite(radius_max) or radius_max <= 0:
        raise ValueError("adjustment_radius_max must be finite and positive.")
    radius_min = float(min(bbox.generation_radius))
    azimuths = sorted({float(point.azimuth_deg) for point in grid})

    # Pass 1: trajectory support and one geometry-only up/down probe per
    # non-extension phi column. Extension columns intentionally use global height.
    supports = {az: trajectory.query_segment_ray_min(az) for az in azimuths}
    extension = {az: not _interval_contains(bbox.observed_azimuth, az) for az in azimuths}
    clearances, local_results = {}, {}
    for az in azimuths:
        support = supports[az]
        if support.rho is None:
            continue
        clearance = local_clearance_threshold(
            support.rho, trajectory.median_captured_rho, config.geometry,
        )
        clearances[az] = clearance
        h_min = float(support.trajectory_height_min)
        h_max = float(support.trajectory_height_max)
        if extension[az]:
            local_results[az] = unavailable_local_height(
                az, support.rho, h_min, h_max, extension=True,
            )
            continue
        radians = np.radians(az)
        horizontal = np.sin(radians) * frame.x_axis + np.cos(radians) * frame.z_axis
        anchor = center + support.rho * horizontal
        up_camera = build_generated_camera(
            -2, anchor + h_max * up, up, captured_cameras, profile, config,
        )
        down_camera = build_generated_camera(
            -3, anchor + h_min * up, -up, captured_cameras, profile, config,
        )
        local_results[az] = probe_local_height(
            renderer, up_camera, down_camera, center, up, az, support.rho,
            h_min, h_max, clearance, config.local_height,
        )

    # Pass 2/3: strict common global interval, then per-column effective limits.
    global_limits = build_global_height_limits(
        list(local_results.values()), trajectory.captured_heights, config.global_height,
    )
    effective = {
        az: resolve_effective_height(local_results[az], global_limits, force_global=extension[az])
        for az in local_results
    }

    cameras, candidates = [], []
    for point in grid:
        azimuth = float(point.azimuth_deg)
        support = supports[azimuth]
        strategy = (config.outside_placement_strategy if mode_result.mode == CameraMode.OUTSIDE_IN
                    else config.inside_placement_strategy)
        limits = effective.get(azimuth)
        local = local_results.get(azimuth)
        meta = {
            "version": "3.2", "elevation_semantics": "position",
            "grid_azimuth_deg": azimuth, "grid_elevation_deg": point.elevation_deg,
            "grid_direction": None, "is_extension_column": extension[azimuth],
            "rho": support.rho, "rho_source": support.rho_source,
            "direct_intersection_count": support.direct_intersection_count,
            "selected_segment_start_index": support.selected_segment_start_index,
            "selected_segment_end_index": support.selected_segment_end_index,
            "selected_segment_t": support.selected_segment_t,
            "trajectory_cross_height": support.trajectory_cross_height,
            "trajectory_height_min": support.trajectory_height_min,
            "trajectory_height_max": support.trajectory_height_max,
            "fallback_confidence": support.fallback_confidence,
            "low_confidence_trajectory_fallback": support.fallback_low_confidence,
            "local_height": to_jsonable(local) if local is not None else None,
            "effective_height": to_jsonable(limits) if limits is not None else None,
            "local_clearance_threshold": clearances.get(azimuth),
            "raw_initial_position": None, "raw_initial_height": None,
            "initial_height_limit_min": None, "initial_height_limit_max": None,
            "initial_height_limit_source": None,
            "initial_height_clip_applied": False,
            "initial_radius_before_height_clip": None,
            "initial_radius_after_height_clip": None,
            "corrected_initial_position": None, "corrected_initial_height": None,
            "proposal_safe": None,
            "adjustment_radius_max": radius_max, "adjustment_radius_exceeded": False,
            "radius_max_applied": False, "crossing_radius_capped": False,
            "crossing_failed_fallback_initial": False, "adjustment_type": "none",
            "initial_radius_exceeds_adjustment_max": False,
            "adjustment_attempted": False, "adjustment_applied": False,
            "adjustment_distance": 0.0, "adjustment_skip_reason": None,
            "proposed_signed_radius": None,
            "proposed_final_position": None, "proposed_final_height": None,
            "final_height_limit_min": None, "final_height_limit_max": None,
            "final_height_limit_source": None,
            "final_height_clip_applied": False,
            "final_radius_before_height_clip": None,
            "final_radius_after_height_clip": None,
            "final_height_clip_geometry_fallback_initial": False,
            "final_height_limit_unreachable_fallback_initial": False,
            "final_geometry_safe": False,
            "position_direction": None, "position_azimuth_deg": None,
            "position_elevation_deg": None, "camera_forward": None,
            "fps_direction": None, "fps_direction_semantics": "final_position_radial",
            "inside_out_forward_semantics": "preserve_grid_direction_crossing_faces_center",
            "depth_probe_excludes_skybox": True,
        }
        candidate = GeneratedCandidate(
            grid_id=point.grid_id, row=point.row, col=point.col,
            azimuth_deg=point.azimuth_deg, elevation_deg=point.elevation_deg,
            direction=point.direction.copy(), mode=mode_result.mode,
            initial_radius=0.0, initial_radius_source=support.rho_source,
            initial_radius_confidence=(
                1.0 if support.fallback_confidence is None
                else float(support.fallback_confidence)
            ),
            final_signed_radius=0.0, crossed_center=False,
            placement_strategy=strategy, depth_probe=None,
            initial_clearance=None, final_clearance=None, path_safe_fraction=None,
            camera=None, status=CandidateStatus.REJECTED, geometry_metadata=meta,
        )
        candidates.append(candidate)
        if support.fallback_low_confidence:
            candidate.notes.append("LOW_CONFIDENCE_TRAJECTORY_FALLBACK")
        if support.rho is None or limits is None:
            candidate.reject_reason = "NO_TRAJECTORY_RAY_INTERSECTION"
            continue

        radians = np.radians(azimuth)
        horizontal = np.sin(radians) * frame.x_axis + np.cos(radians) * frame.z_axis
        try:
            raw_initial, raw_radius, direction = initial_position_from_rho(
                center, horizontal, up, support.rho, point.elevation_deg,
            )
        except ValueError:
            candidate.reject_reason = "INVALID_POSITION_ELEVATION"
            continue
        meta["grid_direction"] = direction.tolist()
        raw_height = float(np.dot(raw_initial - center, up))
        meta["raw_initial_position"] = raw_initial.tolist()
        meta["raw_initial_height"] = raw_height
        meta["initial_radius_before_height_clip"] = raw_radius
        meta["initial_height_limit_min"] = limits.height_min
        meta["initial_height_limit_max"] = limits.height_max
        meta["initial_height_limit_source"] = _height_source(limits)
        initial_clip = clip_radius_to_height_limit(
            raw_radius, direction, up, limits.height_min, limits.height_max,
        )
        if not initial_clip.reachable:
            candidate.reject_reason = "INITIAL_HEIGHT_LIMIT_UNREACHABLE"
            continue
        initial = center + initial_clip.radius * direction
        r0 = float(initial_clip.radius)
        meta["initial_height_clip_applied"] = initial_clip.clipped
        meta["initial_radius_after_height_clip"] = r0
        meta["corrected_initial_position"] = initial.tolist()
        meta["corrected_initial_height"] = initial_clip.clipped_height
        candidate.initial_radius = r0
        candidate.final_signed_radius = r0
        meta["initial_radius_exceeds_adjustment_max"] = r0 > radius_max
        threshold = clearances[azimuth]
        safe, candidate.initial_clearance = safety.is_position_safe(initial, threshold)
        meta["proposal_safe"] = bool(safe and np.isfinite(candidate.initial_clearance))
        if not meta["proposal_safe"]:
            candidate.reject_reason = (
                "INITIAL_HEIGHT_CLIP_GEOMETRY_COLLISION" if initial_clip.clipped
                else "UNSAFE_INITIAL_PRIOR"
            )
            continue

        final, signed = initial.copy(), r0
        crossing_attempted = False
        final_height_clipped = False
        adjustment_allowed = (
            strategy != "prior_only"
            and (mode_result.mode == CameraMode.INSIDE_OUT or r0 <= radius_max)
        )
        if adjustment_allowed:
            meta["adjustment_attempted"] = True
            probe_forward = direction if mode_result.mode == CameraMode.OUTSIDE_IN else -direction
            probe_camera = build_generated_camera(-1, initial, probe_forward, captured_cameras, profile, config)
            depth = depth_probe.probe(probe_camera)
            usable = bool(depth.valid and np.isfinite(depth.depth) and depth.depth > 0)
            if not usable:
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
                if not np.isfinite(proposed):
                    meta["adjustment_skip_reason"] = "INVALID_ADJUSTMENT_RADIUS"
                else:
                    effective_radius = float(proposed)
                    if mode_result.mode == CameraMode.OUTSIDE_IN and effective_radius > radius_max:
                        meta["adjustment_radius_exceeded"] = True
                        meta["radius_max_applied"] = True
                        meta["adjustment_skip_reason"] = "ADJUSTMENT_EXCEEDS_RADIUS_MAX"
                        effective_radius = r0
                    elif mode_result.mode == CameraMode.INSIDE_OUT and effective_radius < 0:
                        crossing_attempted = True
                        meta["adjustment_type"] = "inside_out_crossing"
                        if abs(effective_radius) > radius_max:
                            effective_radius = -radius_max
                            meta["adjustment_radius_exceeded"] = True
                            meta["radius_max_applied"] = True
                            meta["crossing_radius_capped"] = True
                    elif mode_result.mode == CameraMode.INSIDE_OUT:
                        meta["adjustment_type"] = "inside_out_same_side_inward"
                    elif effective_radius != r0:
                        meta["adjustment_type"] = "outside_in_outward"

                    proposed_position = center + effective_radius * direction
                    meta["proposed_final_position"] = proposed_position.tolist()
                    meta["proposed_final_height"] = float(np.dot(proposed_position - center, up))
                    final_limits = global_limits if effective_radius < 0 else limits
                    meta["final_height_limit_min"] = final_limits.height_min
                    meta["final_height_limit_max"] = final_limits.height_max
                    meta["final_height_limit_source"] = (
                        "global_inside_out_crossing" if effective_radius < 0 else _height_source(final_limits)
                    )
                    meta["final_radius_before_height_clip"] = effective_radius
                    final_clip = clip_radius_to_height_limit(
                        effective_radius, direction, up,
                        final_limits.height_min, final_limits.height_max,
                    )
                    if not final_clip.reachable:
                        final, signed = initial.copy(), r0
                        meta["final_height_limit_unreachable_fallback_initial"] = True
                        if crossing_attempted:
                            meta["crossing_failed_fallback_initial"] = True
                        meta["adjustment_skip_reason"] = "FINAL_HEIGHT_LIMIT_UNREACHABLE_FALLBACK_INITIAL"
                    else:
                        effective_radius = final_clip.radius
                        final_height_clipped = final_clip.clipped
                        meta["final_height_clip_applied"] = final_clip.clipped
                        meta["final_radius_after_height_clip"] = effective_radius
                        final = center + effective_radius * direction
                        if config.use_path_safety or crossing_attempted:
                            path = safety.safe_path_fraction(initial, final, threshold)
                            candidate.path_safe_fraction = path.safe_fraction
                            if crossing_attempted and path.safe_fraction < 1.0 - 1e-9:
                                final = initial.copy()
                                meta["crossing_failed_fallback_initial"] = True
                                meta["adjustment_skip_reason"] = "CROSSING_PATH_UNSAFE_FALLBACK_INITIAL"
                            else:
                                final = initial + path.safe_fraction * (final - initial)
                        signed = float(np.dot(final - center, direction))
                        if (not crossing_attempted and not final_height_clipped
                                and abs(signed) < min(r0, max(radius_min, EPS))):
                            final, signed = initial.copy(), r0
                            meta["adjustment_skip_reason"] = "PATH_CLIPPED_BELOW_MIN_RADIUS"
        elif strategy != "prior_only":
            meta["adjustment_skip_reason"] = "INITIAL_EXCEEDS_ADJUSTMENT_RADIUS_MAX"

        candidate.final_signed_radius = signed
        candidate.crossed_center = signed < 0
        meta["final_position"] = final.tolist()
        meta["adjustment_distance"] = float(np.linalg.norm(final - initial))
        meta["adjustment_applied"] = meta["adjustment_distance"] > EPS
        final_safe, candidate.final_clearance = safety.is_position_safe(final, threshold)
        if not final_safe and final_height_clipped:
            final, signed = initial.copy(), r0
            candidate.final_clearance = candidate.initial_clearance
            final_safe = True
            meta["final_height_clip_geometry_fallback_initial"] = True
            if crossing_attempted:
                meta["crossing_failed_fallback_initial"] = True
            meta["adjustment_skip_reason"] = "FINAL_HEIGHT_CLIP_GEOMETRY_FALLBACK_INITIAL"
        elif not final_safe and crossing_attempted:
            final, signed = initial.copy(), r0
            candidate.final_clearance = candidate.initial_clearance
            final_safe = True
            meta["crossing_failed_fallback_initial"] = True
            meta["adjustment_skip_reason"] = "CROSSING_FINAL_UNSAFE_FALLBACK_INITIAL"

        candidate.final_signed_radius = signed
        candidate.crossed_center = signed < 0
        meta["crossed_center"] = candidate.crossed_center
        meta["final_signed_radius"] = signed
        meta["final_position"] = final.tolist()
        meta["final_height"] = float(np.dot(final - center, up))
        meta["adjustment_distance"] = float(np.linalg.norm(final - initial))
        meta["adjustment_applied"] = meta["adjustment_distance"] > EPS
        meta["final_geometry_safe"] = bool(final_safe and np.isfinite(candidate.final_clearance))
        if not meta["final_geometry_safe"]:
            candidate.reject_reason = "FINAL_POSITION_TOO_CLOSE_TO_GEOMETRY"
            continue

        radial = final - center
        length = float(np.linalg.norm(radial))
        if not np.isfinite(radial).all() or length <= EPS:
            candidate.reject_reason = "INVALID_VIEW_ORIENTATION"
            continue
        radial /= length
        forward = resolve_final_camera_forward(mode_result.mode, radial, direction)
        camera = build_generated_camera(len(cameras), final, forward, captured_cameras, profile, config)
        candidate.direction = radial
        candidate.azimuth_deg, candidate.elevation_deg = direction_to_azimuth_elevation(radial, frame)
        meta["position_direction"] = radial.tolist()
        meta["position_azimuth_deg"] = candidate.azimuth_deg
        meta["position_elevation_deg"] = candidate.elevation_deg
        meta["camera_forward"] = forward.tolist()
        meta["fps_direction"] = radial.tolist()
        meta["renderer_near_plane"] = float(renderer.config.near_plane)
        skybox_margin = float(renderer.skybox_radius * renderer.config.skybox.radial_band_ratio)
        if len(getattr(renderer, "skybox_means_np", ())) and not renderer.camera_inside_skybox(camera, skybox_margin):
            candidate.notes.append("CAMERA_OUTSIDE_SKYBOX")
        candidate.camera = camera
        candidate.status = CandidateStatus.VALID
        cameras.append(camera)

    reasons = Counter(c.reject_reason for c in candidates if c.reject_reason)
    local_values = list(local_results.values())
    diagnostics = {
        "version": "3.2", "angular_grid_count": len(grid),
        "trajectory_branch_count": trajectory.branch_count,
        "trajectory_jump_rejected_count": trajectory.jump_rejected_count,
        "trajectory_invalid_pose_count": trajectory.invalid_pose_count,
        "median_captured_horizontal_radius": trajectory.median_captured_rho,
        "direct_rho_columns": sum(s.direct_intersection_count > 0 for s in supports.values()),
        "fallback_rho_columns": sum(s.rho is not None and s.direct_intersection_count == 0 for s in supports.values()),
        "low_confidence_rho_fallback_columns": sum(s.fallback_low_confidence and s.rho is not None for s in supports.values()),
        "extension_global_height_columns": sum(extension.values()),
        "local_up_reliable": sum(r.upper_status == LOCAL_RELIABLE for r in local_values),
        "local_up_hole_uncertain": sum(r.upper_status == LOCAL_HOLE_UNCERTAIN for r in local_values),
        "local_up_unavailable": sum(r.upper_status == LOCAL_UNAVAILABLE for r in local_values),
        "local_down_reliable": sum(r.lower_status == LOCAL_RELIABLE for r in local_values),
        "local_down_hole_uncertain": sum(r.lower_status == LOCAL_HOLE_UNCERTAIN for r in local_values),
        "local_down_unavailable": sum(r.lower_status == LOCAL_UNAVAILABLE for r in local_values),
        "global_height_min": global_limits.height_min,
        "global_height_max": global_limits.height_max,
        "global_lower_captured_fallback": global_limits.fallback_to_captured_lower,
        "global_upper_captured_fallback": global_limits.fallback_to_captured_upper,
        "global_height_conflict": global_limits.conflict,
        "initial_height_clip_count": sum(c.geometry_metadata["initial_height_clip_applied"] for c in candidates),
        "initial_height_unreachable_count": reasons["INITIAL_HEIGHT_LIMIT_UNREACHABLE"],
        "initial_height_clip_geometry_reject_count": reasons["INITIAL_HEIGHT_CLIP_GEOMETRY_COLLISION"],
        "final_height_clip_count": sum(c.geometry_metadata["final_height_clip_applied"] for c in candidates),
        "final_height_unreachable_fallback_count": sum(c.geometry_metadata["final_height_limit_unreachable_fallback_initial"] for c in candidates),
        "final_height_geometry_fallback_count": sum(c.geometry_metadata["final_height_clip_geometry_fallback_initial"] for c in candidates),
        "inside_out_crossing_count": sum(c.crossed_center for c in candidates),
        "inside_out_crossing_fallback_initial_count": sum(
            c.geometry_metadata["crossing_failed_fallback_initial"] for c in candidates
        ),
        "valid_candidate_count": len(cameras),
        "rejection_reasons": dict(reasons),
    }
    placement = {
        "version": "3.2",
        "scene_center": center.tolist(),
        "coordinate_frame": {
            "x_axis": np.asarray(frame.x_axis).tolist(),
            "up_axis": up.tolist(),
            "z_axis": np.asarray(frame.z_axis).tolist(),
        },
        "trajectory_columns": [to_jsonable(supports[az]) | {"is_extension_column": extension[az]} for az in azimuths],
        "local_height_columns": [to_jsonable(local_results[az]) for az in azimuths if az in local_results],
        "effective_height_columns": [
            {"azimuth_deg": az, **to_jsonable(effective[az])}
            for az in azimuths if az in effective
        ],
        "global_height": to_jsonable(global_limits),
    }
    if global_limits.conflict:
        print("[S2:V3.2_WARNING] GLOBAL_HEIGHT_INTERVAL_CONFLICT; using captured height range.")
    if diagnostics["final_height_geometry_fallback_count"]:
        print("[S2:V3.2_WARNING] final height-clipped geometry collision; "
              f"fell back to initial for {diagnostics['final_height_geometry_fallback_count']} candidate(s).")
    if diagnostics["inside_out_crossing_fallback_initial_count"]:
        print("[S2:V3.2_WARNING] inside-out crossing failed safety validation; "
              "fell back to safe initial position for "
              f"{diagnostics['inside_out_crossing_fallback_initial_count']} candidate(s).")
    if config.console_log_candidates:
        for candidate in candidates:
            meta = candidate.geometry_metadata
            print(
                f"[S2:V3.2_CANDIDATE] grid={candidate.grid_id} "
                f"az={meta['grid_azimuth_deg']:.3f} el={meta['grid_elevation_deg']:.3f} "
                f"rho={meta['rho']} rho_source={meta['rho_source']} "
                f"extension={meta['is_extension_column']} r0={candidate.initial_radius:.4f} "
                f"r_final={candidate.final_signed_radius:.4f} "
                f"height_clip=({meta['initial_height_clip_applied']},"
                f"{meta['final_height_clip_applied']}) status={candidate.status.value} "
                f"reject={candidate.reject_reason}"
            )
        print(f"[S2:V3.2_FUNNEL] {diagnostics}")
    return PoseGenerationResult(
        mode_result, bbox, view_limits, candidates, cameras, config,
        diagnostics=diagnostics, placement_metadata=placement,
    )
