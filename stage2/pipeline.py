"""V3.3 Stage-2 pipeline assembled from explicit proposal/safety substages."""
from collections import Counter
from dataclasses import replace
import numpy as np

from viewpoint_framework.geometry_safety import PointCloudSafety, local_clearance_threshold
from viewpoint_framework.pose_generation import (
    CandidateStatus, GeneratedCandidate, PoseGenerationResult, build_generated_camera,
)
from viewpoint_framework.pose_generation_v3 import initial_position_from_rho
from viewpoint_framework.scene_types import CameraMode, to_jsonable
from viewpoint_framework.stage2.height import (
    effective_height, probe_local_height, resolve_global_height,
    unavailable_local, clip_radius_to_height_limit,
)
from viewpoint_framework.stage2.radius import (
    center_guard, classify_inside_signed_radius, resolve_radius_targets,
)
from viewpoint_framework.stage2.trajectory import (
    propagate_cross_heights, resolve_all_radii,
)
from viewpoint_framework.trajectory_safe_field import TrajectorySafeField
from viewpoint_framework.view_space import direction_to_azimuth_elevation


EPS = 1e-10


def _validate(config, points, depth_probe):
    if points is None:
        raise ValueError("V3.3 requires point-cloud geometry.")
    renderer = getattr(depth_probe, "renderer", None)
    if renderer is None or not hasattr(renderer, "render_geometry_depth"):
        raise ValueError("V3.3 requires a geometry-only Gaussian renderer.")
    config.grid.validate(); config.radius.validate(); config.renderer.validate()
    config.local_height.validate(); config.global_height.validate()
    return renderer


def _horizontal(azimuth, frame):
    angle = np.radians(float(azimuth))
    return np.sin(angle)*frame.x_axis + np.cos(angle)*frame.z_axis


def generate_v33_candidates(captured_cameras, profile, bbox, view_limits, grid,
                            mode_result, point_cloud_points, depth_probe, config,
                            view_domain=None):
    renderer = _validate(config, point_cloud_points, depth_probe)
    center = np.asarray(profile.center_fit.center, dtype=np.float64)
    frame, up = profile.coordinate_frame, profile.coordinate_frame.y_axis
    trajectory = TrajectorySafeField(captured_cameras, center, frame,
                                     config.trajectory_safe_field)
    safety = PointCloudSafety(point_cloud_points, config.geometry)
    azimuths = sorted({float(p.azimuth_deg) for p in grid})
    ordered, priors = resolve_all_radii(
        azimuths, bbox.observed_azimuth, bbox.generation_azimuth, trajectory,
        bbox.angular_extension_deg,
    )
    clearances = {
        az: local_clearance_threshold(prior.rho, trajectory.median_captured_rho,
                                      config.geometry)
        for az, prior in priors.items() if prior.rho is not None
    }
    propagate_cross_heights(ordered, priors, bbox.observed_azimuth, center, frame,
                            safety, clearances)

    local_results = {}
    for azimuth in ordered:
        prior = priors[azimuth]
        if prior.rho is None or prior.h_cross is None:
            if prior.rho is not None:
                local_results[azimuth] = unavailable_local(
                    azimuth, prior.rho, "global_no_safe_h_cross")
            continue
        origin = center + prior.rho*_horizontal(azimuth, frame) + prior.h_cross*up
        up_camera = build_generated_camera(-2, origin, up, captured_cameras, profile, config)
        down_camera = build_generated_camera(-3, origin, -up, captured_cameras, profile, config)
        local_results[azimuth] = probe_local_height(
            renderer, up_camera, down_camera, center, up, azimuth, prior.rho,
            prior.h_cross, clearances[azimuth], config.local_height,
        )
        local_results[azimuth].source = prior.height_source

    global_height = resolve_global_height(
        list(local_results.values()), trajectory.captured_heights,
        config.global_height)
    effective = {az: effective_height(local, global_height)
                 for az, local in local_results.items()}
    captured_max = float(np.max(np.hypot(
        ((trajectory.positions-center)@trajectory.basis)[:, 0],
        ((trajectory.positions-center)@trajectory.basis)[:, 2],
    )))
    nominal_cap = config.radius.nominal_captured_max_ratio*captured_max
    geometry_radii = np.linalg.norm(
        np.asarray(renderer.geometry_means_np)-center[None], axis=1)
    emergency_base = (float(renderer.skybox_radius)
                      if len(getattr(renderer, "skybox_means_np", ()))
                      else float(np.quantile(geometry_radii, .99)))
    near_plane = float(getattr(renderer.config, "near_plane", 0.01))
    cameras, candidates = [], []
    for point in grid:
        azimuth = float(point.azimuth_deg)
        prior = priors[azimuth]
        limits = effective.get(azimuth, global_height)
        meta = {
            "grid_azimuth_deg": azimuth,
            "grid_elevation_deg": float(point.elevation_deg),
            "rho": prior.rho,
            "rho_source": prior.rho_source,
            "column_kind": prior.kind,
            "h_cross": prior.h_cross,
            "height_source": getattr(limits, "lower_source", "global"),
            "nominal_radius_cap": nominal_cap,
            "emergency_radius_base": emergency_base,
            "radius_extension_attempted": False,
            "radius_extension_blocked_by_hole": False,
            "crossing_failed_fallback_initial": False,
        }
        candidate = GeneratedCandidate(
            grid_id=point.grid_id, row=point.row, col=point.col,
            azimuth_deg=azimuth, elevation_deg=float(point.elevation_deg),
            direction=point.direction.copy(), mode=mode_result.mode,
            initial_radius=0.0, initial_radius_source=prior.rho_source,
            initial_radius_confidence=1.0 if prior.rho is not None else 0.0,
            final_signed_radius=0.0, crossed_center=False,
            placement_strategy="v3.3_trajectory_local_global",
            depth_probe=None, initial_clearance=None, final_clearance=None,
            path_safe_fraction=None, camera=None,
            status=CandidateStatus.REJECTED, geometry_metadata=meta,
        )
        candidates.append(candidate)
        if prior.rho is None:
            candidate.reject_reason = "NO_TRAJECTORY_RADIUS"
            continue

        horizontal = _horizontal(azimuth, frame)
        try:
            _, raw_radius, direction = initial_position_from_rho(
                center, horizontal, up, prior.rho, point.elevation_deg)
        except ValueError:
            candidate.reject_reason = "INVALID_POSITION_ELEVATION"
            continue
        initial_clip = clip_radius_to_height_limit(
            raw_radius, direction, up, limits.height_min, limits.height_max)
        if not initial_clip.reachable:
            candidate.reject_reason = "INITIAL_HEIGHT_LIMIT_UNREACHABLE"
            continue
        initial = center + initial_clip.radius * direction
        r0 = float(initial_clip.radius)
        threshold = clearances[azimuth]
        safe, candidate.initial_clearance = safety.is_position_safe(initial, threshold)
        candidate.initial_radius = r0
        candidate.final_signed_radius = r0
        meta.update({
            "raw_initial_position": (center + raw_radius * direction).tolist(),
            "corrected_initial_position": initial.tolist(),
            "initial_position": initial.tolist(),
            "initial_radius_before_height_clip": raw_radius,
            "initial_radius_after_height_clip": r0,
            "initial_height_clip_applied": bool(initial_clip.clipped),
            "local_clearance": threshold,
        })
        if not safe:
            candidate.reject_reason = "UNSAFE_INITIAL_POSITION"
            continue

        probe_forward = direction if mode_result.mode == CameraMode.OUTSIDE_IN else -direction
        probe_camera = build_generated_camera(
            -1, initial, probe_forward, captured_cameras, profile, config)
        depth = depth_probe.probe(probe_camera)
        candidate.depth_probe = depth
        usable = bool(depth.valid and np.isfinite(depth.depth) and depth.depth > 0)
        movement = max(0.0, float(depth.depth) - config.depth_margin_ratio * threshold) if usable else 0.0
        raw_signed = (r0 + movement if mode_result.mode == CameraMode.OUTSIDE_IN
                      else r0 - movement)
        guard = center_guard(threshold, near_plane,
                             config.radius.center_guard_clearance_ratio)
        meta["center_guard"] = guard
        branch = "outside_in"
        if mode_result.mode == CameraMode.INSIDE_OUT:
            branch, raw_signed = classify_inside_signed_radius(raw_signed, guard)
        probe_signed = raw_signed

        emergency_margin = max(threshold, near_plane)
        emergency_cap = max(guard, emergency_base - emergency_margin)
        if abs(raw_signed) > emergency_cap:
            raw_signed = float(np.copysign(emergency_cap, raw_signed))
            meta["emergency_radius_clipped"] = True
        targets = resolve_radius_targets(
            abs(raw_signed), nominal_cap,
            hole_detected=bool(getattr(depth, "hole_detected", False)),
        )
        meta.update({
            "radius_probe": abs(probe_signed),
            "radius_extension_target": targets.extension,
            "radius_over_nominal": targets.over_nominal,
            "radius_extension_accepted": False,
            "radius_nominal_retry": False,
        })
        signs = -1.0 if raw_signed < 0 else 1.0
        attempts = []
        if targets.extension is not None:
            attempts.append(signs * targets.extension)
            meta["radius_extension_attempted"] = True
        attempts.append(signs * targets.nominal)
        if targets.over_nominal and not targets.extension_allowed:
            meta["radius_extension_blocked_by_hole"] = True
        # Avoid repeated attempts, while retaining the safe initial fallback.
        unique_attempts = []
        for value in attempts:
            if not any(abs(value-old) <= EPS for old in unique_attempts):
                unique_attempts.append(value)

        final, signed = initial.copy(), r0
        chosen = "safe_initial"
        for target in unique_attempts:
            final_limits = global_height if target < 0 else limits
            clipped = clip_radius_to_height_limit(
                target, direction, up,
                final_limits.height_min, final_limits.height_max)
            if not clipped.reachable:
                continue
            proposed = center + clipped.radius * direction
            path = safety.safe_path_fraction(initial, proposed, threshold)
            if not path.fully_safe:
                continue
            endpoint_safe, endpoint_clearance = safety.is_position_safe(proposed, threshold)
            if not endpoint_safe:
                continue
            final, signed = proposed, float(clipped.radius)
            candidate.final_clearance = endpoint_clearance
            candidate.path_safe_fraction = path.safe_fraction
            chosen = ("over_nominal_extension" if abs(target) > nominal_cap + EPS
                      else "nominal_or_probe")
            meta["radius_extension_accepted"] = chosen == "over_nominal_extension"
            meta["radius_nominal_retry"] = bool(
                meta["radius_extension_attempted"] and not meta["radius_extension_accepted"])
            break
        else:
            candidate.final_clearance = candidate.initial_clearance
            candidate.path_safe_fraction = 1.0
            if raw_signed < 0:
                meta["crossing_failed_fallback_initial"] = True

        candidate.final_signed_radius = signed
        candidate.crossed_center = bool(signed < 0)
        radial = final - center
        radial_norm = float(np.linalg.norm(radial))
        if not np.isfinite(radial).all() or radial_norm <= EPS:
            candidate.reject_reason = "INVALID_VIEW_ORIENTATION"
            continue
        radial /= radial_norm
        # Contract: outside-in and crossed inside-out face scene center;
        # non-crossed inside-out keeps facing away from it.
        forward = (-radial if (mode_result.mode == CameraMode.OUTSIDE_IN or signed < 0)
                   else radial)
        camera = build_generated_camera(
            len(cameras), final, forward, captured_cameras, profile, config)
        candidate.direction = radial
        candidate.azimuth_deg, candidate.elevation_deg = direction_to_azimuth_elevation(radial, frame)
        candidate.camera = camera
        candidate.status = CandidateStatus.VALID
        cameras.append(camera)
        meta.update({
            "probe_usable": usable,
            "probe_hole_detected": bool(getattr(depth, "hole_detected", False)),
            "raw_signed_radius": raw_signed,
            "radius_branch": branch,
            "radius_choice": chosen,
            "emergency_radius_cap": emergency_cap,
            "final_position": final.tolist(),
            "final_signed_radius": signed,
            "camera_forward": forward.tolist(),
            "final_height_source": "global_crossing" if signed < 0 else meta["height_source"],
        })

    reasons = Counter(c.reject_reason for c in candidates if c.reject_reason)
    diagnostics = {
        "version": "3.3",
        "angular_grid_count": len(grid),
        "valid_candidate_count": len(cameras),
        "rejection_reasons": dict(reasons),
        "direct_radius_columns": sum(p.kind == "observed" for p in priors.values()),
        "extension_radius_columns": sum(p.kind == "extension" for p in priors.values()),
        "close_loop_gap_columns": sum(p.kind == "close_loop_gap" for p in priors.values()),
        "propagated_height_columns": sum(p.height_source.startswith("propagated_") for p in priors.values()),
        "global_height_columns": sum(p.h_cross is None for p in priors.values()),
        "inside_out_crossing_count": sum(c.crossed_center for c in candidates),
        "inside_out_crossing_fallback_initial_count": sum(
            c.geometry_metadata.get("crossing_failed_fallback_initial", False)
            for c in candidates),
        "hole_blocked_radius_extension_count": sum(
            c.geometry_metadata.get("radius_extension_blocked_by_hole", False)
            for c in candidates),
        "global_height_strategy": global_height.strategy,
        "global_height_band_count": global_height.consensus_band_count,
        "view_domain": to_jsonable(view_domain) if view_domain is not None else None,
    }
    placement = {
        "version": "3.3",
        "scene_center": center.tolist(),
        "coordinate_frame": {
            "x_axis": np.asarray(frame.x_axis).tolist(),
            "up_axis": up.tolist(),
            "z_axis": np.asarray(frame.z_axis).tolist(),
        },
        "median_captured_horizontal_radius": trajectory.median_captured_rho,
        "trajectory_columns": [to_jsonable(priors[a]) for a in ordered],
        "local_height_columns": [to_jsonable(local_results[a]) for a in ordered if a in local_results],
        "effective_height_columns": [
            {"azimuth_deg": a, **to_jsonable(effective[a])}
            for a in ordered if a in effective],
        "global_height": to_jsonable(global_height),
        "nominal_radius_cap": nominal_cap,
        "emergency_radius_base": emergency_base,
        "view_domain": to_jsonable(view_domain) if view_domain is not None else None,
    }
    if diagnostics["inside_out_crossing_fallback_initial_count"]:
        print("[S2:V3.3_WARNING] inside-out crossing failed safety validation; "
              f"used safe initial position for {diagnostics['inside_out_crossing_fallback_initial_count']} candidate(s).")
    print("[S2:V3.3] "
          f"grid={len(grid)} valid={len(cameras)} rejected={len(grid)-len(cameras)} "
          f"height={global_height.strategy} crossing_fallback="
          f"{diagnostics['inside_out_crossing_fallback_initial_count']}")
    return PoseGenerationResult(
        mode=mode_result, bbox=bbox, view_limits=view_limits,
        candidates=candidates, valid_cameras=cameras, config=config,
        diagnostics=diagnostics,
        renderer_metadata=getattr(renderer, "metadata", {}),
        placement_metadata=placement,
    )
