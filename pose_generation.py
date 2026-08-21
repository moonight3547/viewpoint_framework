#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage-2 candidate viewpoint pose generation.

Scope of this module
--------------------
1. Freeze one default observation mode (outside-in or inside-out).
2. Export endpoint ``view_limits.json`` for that mode.
3. Sample an azimuth/elevation grid inside the generation bbox.
4. Project each angular grid point onto a camera-radius prior.
5. Optionally use reverse 3DGS depth to move the camera backward for a wider view.
6. Use point-cloud geometry as a hard position/path safety guard.
7. Export *all safe candidates*; view selection/coverage optimization is intentionally
   left to the next stage.

The default path is intentionally simple and durable.  Strategy strings are kept at
important decision boundaries so later ablation work does not require rewriting the
pipeline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from viewpoint_framework.cameras_util import Camera
from viewpoint_framework.geometry_safety import (
    GeometrySafetyConfig,
    PointCloudSafety,
)
from viewpoint_framework.gs_depth_probe import (
    DepthProbeResult,
    NullDepthProbe,
)
from viewpoint_framework.scene_types import (
    CameraMode,
    GlobalCollectionMode,
    SceneProfile,
    ViewBBox,
    to_jsonable,
)
from viewpoint_framework.scene_understanding import SceneUnderstandingResult
from viewpoint_framework.view_space import (
    azimuth_elevation_to_direction,
    expand_circular_interval,
    minimal_circular_interval,
    sample_circular_interval,
)


EPS = 1e-10


class CandidateStatus(str, Enum):
    VALID = "valid"
    REJECTED = "rejected"


@dataclass
class PoseGenerationConfig:
    """Compact configuration with strategy hooks for later ablations."""

    # Default observation mode.
    mode_strategy: str = "count_majority"     # count_majority | weighted_majority | legacy_majority | forced
    forced_mode: Optional[str] = None          # outside_in | inside_out

    # Angular grid.
    grid_strategy: str = "uniform"             # uniform | cos_elevation
    azimuth_step_deg: float = 20.0
    elevation_step_deg: float = 20.0
    include_bbox_end: bool = True

    # Initial radial projection.
    radius_strategy: str = "directional"       # directional | azimuth_only | bbox_median
    radius_min_confidence: float = 0.0
    radius_fallback: str = "mode_median"       # mode_median | bbox_median

    # Camera placement.
    outside_placement_strategy: str = "depth_backoff"  # prior_only | depth_backoff
    inside_placement_strategy: str = "center_crossing_depth"  # prior_only | no_crossing | center_crossing_depth
    depth_margin_ratio: float = 1.0             # multiplied by geometry safety clearance
    center_cross_extra_ratio: float = 0.5       # extra clearance beyond center
    invalid_depth_behavior: str = "keep_prior" # keep_prior | use_radius_max

    # Point-cloud hard safety.
    geometry: GeometrySafetyConfig = field(default_factory=GeometrySafetyConfig)
    use_path_safety: bool = True
    reject_unsafe_initial_prior: bool = False

    # Generated intrinsics: matches the old panorama generator by default.
    intrinsics_strategy: str = "first_scaled"  # first_scaled | first | median_scaled
    focal_ratio: float = 0.7
    center_principal_point: bool = True

    # Endpoint view-limits export.
    view_limits_strategy: str = "generation_bbox"  # generation_bbox | observed_bbox
    view_limits_unwrap_azimuth: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PoseGenerationConfig":
        payload = dict(data)
        geometry_data = payload.pop("geometry", {})
        cfg = cls(**payload)
        cfg.geometry = GeometrySafetyConfig(**geometry_data)
        return cfg


@dataclass
class ObservationModeResult:
    mode: CameraMode
    confidence: float
    strategy: str
    note: str = ""


@dataclass
class AngularGridPoint:
    grid_id: int
    row: int
    col: int
    azimuth_deg: float
    elevation_deg: float
    direction: np.ndarray


@dataclass
class GeneratedCandidate:
    grid_id: int
    row: int
    col: int
    azimuth_deg: float
    elevation_deg: float
    direction: np.ndarray

    mode: CameraMode
    initial_radius: float
    initial_radius_source: str
    initial_radius_confidence: float

    final_signed_radius: float
    crossed_center: bool
    placement_strategy: str

    depth_probe: Optional[DepthProbeResult]
    initial_clearance: Optional[float]
    final_clearance: Optional[float]
    path_safe_fraction: Optional[float]

    camera: Optional[Camera]
    status: CandidateStatus
    reject_reason: Optional[str] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class PoseGenerationResult:
    mode: ObservationModeResult
    bbox: ViewBBox
    view_limits: Dict[str, Any]
    candidates: List[GeneratedCandidate]
    valid_cameras: List[Camera]
    config: PoseGenerationConfig


# -----------------------------------------------------------------------------
# Mode selection
# -----------------------------------------------------------------------------

def choose_observation_mode(
    profile: SceneProfile,
    config: PoseGenerationConfig,
) -> ObservationModeResult:
    summary = profile.mode_summary
    strategy = config.mode_strategy

    if strategy == "forced":
        if config.forced_mode not in (CameraMode.OUTSIDE_IN.value, CameraMode.INSIDE_OUT.value):
            raise ValueError(
                "forced mode requires forced_mode='outside_in' or 'inside_out'."
            )
        return ObservationModeResult(
            mode=CameraMode(config.forced_mode),
            confidence=1.0,
            strategy=strategy,
            note="user-forced default observation mode",
        )

    if strategy in ("count_majority", "legacy_majority"):
        outside_score = float(summary.outside_in_count)
        inside_score = float(summary.inside_out_count)
    elif strategy == "weighted_majority":
        outside_score = float(summary.outside_in_weight)
        inside_score = float(summary.inside_out_weight)
    else:
        raise ValueError(f"Unknown mode_strategy: {strategy}")

    total = outside_score + inside_score
    if outside_score >= inside_score:
        mode = CameraMode.OUTSIDE_IN
        winner = outside_score
    else:
        mode = CameraMode.INSIDE_OUT
        winner = inside_score

    confidence = winner / total if total > EPS else 0.0
    note = ""
    if total <= EPS:
        # A deterministic fallback is preferable to failing the whole generation
        # stage when scene-understanding classification is weak.
        mode = CameraMode.OUTSIDE_IN
        note = "no reliable mode support; deterministic outside-in fallback"

    return ObservationModeResult(
        mode=mode,
        confidence=float(confidence),
        strategy=strategy,
        note=note,
    )


# -----------------------------------------------------------------------------
# BBox fallback + view_limits export
# -----------------------------------------------------------------------------

def _fallback_bbox_from_relations(
    profile: SceneProfile,
    mode: CameraMode,
) -> ViewBBox:
    """Emergency bbox when the first-stage per-mode bbox is missing.

    Use mode-compatible relations first; if classification was too weak, use all
    finite camera relations.  This keeps stage-2 usable while stage-1 is still being
    tuned on real scenes.
    """
    relations = [
        r for r in profile.camera_relations
        if r.mode == mode
        and r.azimuth_deg is not None
        and r.elevation_deg is not None
        and np.isfinite(r.radius)
        and r.radius > EPS
    ]
    note = "fallback bbox from selected-mode camera relations"

    if not relations:
        relations = [
            r for r in profile.camera_relations
            if r.azimuth_deg is not None
            and r.elevation_deg is not None
            and np.isfinite(r.radius)
            and r.radius > EPS
        ]
        note = "fallback bbox from all finite camera relations"

    if not relations:
        raise ValueError("Cannot build view bbox: no finite camera spherical relations.")

    azimuths = [float(r.azimuth_deg) for r in relations]
    elevations = np.asarray([float(r.elevation_deg) for r in relations])
    radii = np.asarray([float(r.radius) for r in relations])

    observed_az = minimal_circular_interval(azimuths)
    generation_az = expand_circular_interval(observed_az, 5.0)

    e_min, e_max = float(np.min(elevations)), float(np.max(elevations))
    generation_e = (max(-89.0, e_min - 5.0), min(89.0, e_max + 5.0))

    if len(radii) >= 4:
        r_min, r_max = [float(x) for x in np.percentile(radii, [5.0, 95.0])]
    else:
        r_min, r_max = float(np.min(radii)), float(np.max(radii))
    if r_max < r_min + EPS:
        r_max = r_min + max(1e-3, 0.05 * max(r_min, 1.0))

    return ViewBBox(
        mode=mode,
        strategy="stage2_fallback",
        camera_indices=[int(r.camera_index) for r in relations],
        observed_azimuth=observed_az,
        observed_elevation_deg=(e_min, e_max),
        observed_radius=(r_min, r_max),
        generation_azimuth=generation_az,
        generation_elevation_deg=generation_e,
        generation_radius=(r_min, r_max),
        angular_extension_deg=5.0,
        radius_extension=0.0,
        support_camera_count=len(relations),
        notes=[note],
    )


def resolve_generation_bbox(
    profile: SceneProfile,
    mode: CameraMode,
) -> ViewBBox:
    key = mode.value
    if key in profile.view_bboxes:
        return profile.view_bboxes[key]
    return _fallback_bbox_from_relations(profile, mode)


def _interval_to_unwrapped_bounds(interval) -> Tuple[float, float]:
    start = float(interval.start_deg)
    end = start + float(interval.span_deg)
    return start, end


def build_view_limits(
    profile: SceneProfile,
    bbox: ViewBBox,
    config: PoseGenerationConfig,
) -> Dict[str, Any]:
    if config.view_limits_strategy == "generation_bbox":
        az = bbox.generation_azimuth
        elevation_min, elevation_max = bbox.generation_elevation_deg
        radius_min, radius_max = bbox.generation_radius
    elif config.view_limits_strategy == "observed_bbox":
        az = bbox.observed_azimuth
        elevation_min, elevation_max = bbox.observed_elevation_deg
        radius_min, radius_max = bbox.observed_radius
    else:
        raise ValueError(
            f"Unknown view_limits_strategy: {config.view_limits_strategy}"
        )

    if config.view_limits_unwrap_azimuth:
        min_phi, max_phi = _interval_to_unwrapped_bounds(az)
    else:
        min_phi, max_phi = float(az.start_deg), float(az.end_deg)

    # Legacy endpoint convention: Theta is polar angle = 90 - elevation.
    min_theta = 90.0 - float(elevation_max)
    max_theta = 90.0 - float(elevation_min)

    frame = profile.coordinate_frame
    # Columns map scene-frame coordinates to world coordinates.
    gravity_coordinate = np.stack(
        [frame.x_axis, frame.y_axis, frame.z_axis], axis=1
    )

    return {
        "gravityCoordinate": gravity_coordinate.tolist(),
        "minPhi": float(min_phi),
        "maxPhi": float(max_phi),
        "minTheta": float(min_theta),
        "maxTheta": float(max_theta),
        "minRadius": float(max(radius_min, 0.0)),
        "maxRadius": float(max(radius_max, radius_min)),
        "minX": 0.0,
        "maxX": 0.0,
        "minY": 0.0,
        "maxY": 0.0,
        "minZ": 0.0,
        "maxZ": 0.0,
        "target": profile.center_fit.center.astype(float).tolist(),
    }


# -----------------------------------------------------------------------------
# Angular grid
# -----------------------------------------------------------------------------

def _sample_elevations(elevation_range: Tuple[float, float], step_deg: float) -> np.ndarray:
    if step_deg <= 0:
        raise ValueError("elevation_step_deg must be > 0.")
    lo, hi = [float(x) for x in elevation_range]
    if hi < lo:
        lo, hi = hi, lo
    if hi - lo <= EPS:
        return np.asarray([lo], dtype=np.float64)

    values = np.arange(lo, hi + 0.5 * step_deg, step_deg, dtype=np.float64)
    values = values[values <= hi + EPS]
    if len(values) == 0 or abs(float(values[-1]) - hi) > 1e-8:
        values = np.concatenate([values, [hi]])
    return values


def generate_angular_grid(
    profile: SceneProfile,
    bbox: ViewBBox,
    config: PoseGenerationConfig,
) -> List[AngularGridPoint]:
    if config.azimuth_step_deg <= 0:
        raise ValueError("azimuth_step_deg must be > 0.")

    elevations = _sample_elevations(
        bbox.generation_elevation_deg,
        config.elevation_step_deg,
    )

    points: List[AngularGridPoint] = []
    grid_id = 0

    for row, elevation in enumerate(elevations):
        if config.grid_strategy == "uniform":
            az_step = float(config.azimuth_step_deg)
        elif config.grid_strategy == "cos_elevation":
            cosine = max(abs(float(np.cos(np.radians(elevation)))), 0.20)
            az_step = min(90.0, float(config.azimuth_step_deg) / cosine)
        else:
            raise ValueError(f"Unknown grid_strategy: {config.grid_strategy}")

        azimuths = sample_circular_interval(
            bbox.generation_azimuth,
            step_deg=az_step,
            include_end=config.include_bbox_end,
        )

        for col, azimuth in enumerate(azimuths):
            direction = azimuth_elevation_to_direction(
                azimuth_deg=float(azimuth),
                elevation_deg=float(elevation),
                frame=profile.coordinate_frame,
            )
            points.append(
                AngularGridPoint(
                    grid_id=grid_id,
                    row=row,
                    col=col,
                    azimuth_deg=float(azimuth),
                    elevation_deg=float(elevation),
                    direction=direction,
                )
            )
            grid_id += 1

    return points


# -----------------------------------------------------------------------------
# Camera construction / IO
# -----------------------------------------------------------------------------

def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        raise ValueError("Cannot normalize near-zero vector.")
    return vector / norm


def build_c2w_from_forward(
    position: np.ndarray,
    forward: np.ndarray,
    world_up: np.ndarray,
    fallback_axis: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build +X right / +Y down / +Z forward c2w used by this repository."""
    position = np.asarray(position, dtype=np.float64).reshape(3)
    forward = _normalize(forward)
    up = _normalize(world_up)

    right = np.cross(forward, up)
    if float(np.linalg.norm(right)) <= 1e-6:
        fallback = (
            np.asarray(fallback_axis, dtype=np.float64)
            if fallback_axis is not None
            else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        )
        fallback = fallback - float(np.dot(fallback, forward)) * forward
        if float(np.linalg.norm(fallback)) <= 1e-6:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        up = _normalize(fallback)
        right = np.cross(forward, up)

    right = _normalize(right)
    true_up = _normalize(np.cross(right, forward))
    down = -true_up

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = position
    return c2w


def _generated_intrinsics(
    cameras: Sequence[Camera],
    config: PoseGenerationConfig,
) -> Tuple[float, float, float, float, int, int]:
    if len(cameras) == 0:
        raise ValueError("At least one captured camera is required for intrinsics.")

    if config.intrinsics_strategy in ("first", "first_scaled"):
        ref = cameras[0]
        fx, fy = float(ref.fx), float(ref.fy)
        width, height = int(ref.width), int(ref.height)
        cx, cy = float(ref.cx), float(ref.cy)
    elif config.intrinsics_strategy == "median_scaled":
        width = int(cameras[0].width)
        height = int(cameras[0].height)
        fx = float(np.median([c.fx for c in cameras]))
        fy = float(np.median([c.fy for c in cameras]))
        cx = float(np.median([c.cx for c in cameras]))
        cy = float(np.median([c.cy for c in cameras]))
    else:
        raise ValueError(
            f"Unknown intrinsics_strategy: {config.intrinsics_strategy}"
        )

    if config.intrinsics_strategy.endswith("scaled"):
        fx *= float(config.focal_ratio)
        fy *= float(config.focal_ratio)

    if config.center_principal_point:
        cx = width / 2.0
        cy = height / 2.0

    return fx, fy, cx, cy, width, height


def build_generated_camera(
    index: int,
    position: np.ndarray,
    forward: np.ndarray,
    cameras: Sequence[Camera],
    profile: SceneProfile,
    config: PoseGenerationConfig,
) -> Camera:
    fx, fy, cx, cy, width, height = _generated_intrinsics(cameras, config)
    c2w = build_c2w_from_forward(
        position=position,
        forward=forward,
        world_up=profile.coordinate_frame.y_axis,
        fallback_axis=profile.coordinate_frame.z_axis,
    )
    w2c = np.linalg.inv(c2w)
    return Camera(
        index=index,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        w2c=w2c,
        c2w=c2w,
    )


def camera_to_18d(camera: Camera) -> List[float]:
    return [
        float(camera.fx),
        float(camera.fy),
        float(camera.cx),
        float(camera.cy),
        int(camera.width),
        int(camera.height),
        *camera.w2c[:3, :4].reshape(-1).astype(float).tolist(),
    ]


def save_cameras_json(cameras: Sequence[Camera], output_path: str) -> None:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([camera_to_18d(c) for c in cameras], f, indent=2)


# -----------------------------------------------------------------------------
# Initial radius projection
# -----------------------------------------------------------------------------

def _mode_radii(profile: SceneProfile, mode: CameraMode) -> np.ndarray:
    radii = [
        float(r.radius)
        for r in profile.camera_relations
        if r.mode == mode and np.isfinite(r.radius) and r.radius > EPS
    ]
    if not radii:
        radii = [
            float(r.radius)
            for r in profile.camera_relations
            if np.isfinite(r.radius) and r.radius > EPS
        ]
    return np.asarray(radii, dtype=np.float64)


def resolve_initial_radius(
    point: AngularGridPoint,
    mode: CameraMode,
    bbox: ViewBBox,
    scene_result: SceneUnderstandingResult,
    config: PoseGenerationConfig,
) -> Tuple[float, float, str]:
    field = scene_result.radius_fields.get(mode.value)
    estimate = None

    if config.radius_strategy == "directional" and field is not None:
        estimate = field.query(point.direction)
    elif config.radius_strategy == "azimuth_only" and field is not None:
        horizontal_direction = azimuth_elevation_to_direction(
            azimuth_deg=point.azimuth_deg,
            elevation_deg=0.0,
            frame=scene_result.profile.coordinate_frame,
        )
        estimate = field.query(horizontal_direction)
    elif config.radius_strategy == "bbox_median":
        estimate = None
    else:
        if config.radius_strategy not in ("directional", "azimuth_only", "bbox_median"):
            raise ValueError(f"Unknown radius_strategy: {config.radius_strategy}")

    r_min, r_max = [float(x) for x in bbox.generation_radius]
    if r_max < r_min:
        r_min, r_max = r_max, r_min

    if (
        estimate is not None
        and estimate.valid
        and float(estimate.confidence) >= float(config.radius_min_confidence)
        and np.isfinite(estimate.nominal)
        and estimate.nominal > EPS
    ):
        radius = float(estimate.nominal)
        confidence = float(estimate.confidence)
        source = f"radius_field:{estimate.strategy}"
    else:
        if config.radius_fallback == "mode_median":
            radii = _mode_radii(scene_result.profile, mode)
            if len(radii):
                radius = float(np.median(radii))
                source = "fallback:mode_median"
            else:
                radius = 0.5 * (r_min + r_max)
                source = "fallback:bbox_median"
        elif config.radius_fallback == "bbox_median":
            radius = 0.5 * (r_min + r_max)
            source = "fallback:bbox_median"
        else:
            raise ValueError(f"Unknown radius_fallback: {config.radius_fallback}")
        confidence = 0.0

    radius = float(np.clip(radius, r_min, r_max))
    return radius, confidence, source


# -----------------------------------------------------------------------------
# Placement
# -----------------------------------------------------------------------------

def _build_probe_camera(
    position: np.ndarray,
    forward: np.ndarray,
    captured_cameras: Sequence[Camera],
    profile: SceneProfile,
    config: PoseGenerationConfig,
) -> Camera:
    # index is irrelevant for a temporary renderer probe.
    return build_generated_camera(
        index=-1,
        position=position,
        forward=forward,
        cameras=captured_cameras,
        profile=profile,
        config=config,
    )


def _apply_path_safety(
    start: np.ndarray,
    target: np.ndarray,
    safety: Optional[PointCloudSafety],
    config: PoseGenerationConfig,
) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
    if safety is None or not config.use_path_safety:
        return np.asarray(target, dtype=np.float64), 1.0, None

    result = safety.safe_path_fraction(start, target)
    adjusted = start + result.safe_fraction * (np.asarray(target) - np.asarray(start))
    return adjusted, float(result.safe_fraction), float(result.min_clearance)


def _signed_radius(center: np.ndarray, position: np.ndarray, direction: np.ndarray) -> float:
    return float(np.dot(np.asarray(position) - np.asarray(center), direction))


def _place_outside_in(
    point: AngularGridPoint,
    initial_radius: float,
    bbox: ViewBBox,
    center: np.ndarray,
    captured_cameras: Sequence[Camera],
    profile: SceneProfile,
    config: PoseGenerationConfig,
    depth_probe,
    safety: Optional[PointCloudSafety],
) -> Tuple[np.ndarray, float, DepthProbeResult, Optional[float], List[str]]:
    u = point.direction
    r_min, r_max = [float(x) for x in bbox.generation_radius]
    r_min, r_max = min(r_min, r_max), max(r_min, r_max)
    initial_radius = float(np.clip(initial_radius, r_min, r_max))
    start = center + initial_radius * u
    notes: List[str] = []

    if config.outside_placement_strategy == "prior_only":
        depth_result = NullDepthProbe().probe()
        target = start.copy()
    elif config.outside_placement_strategy == "depth_backoff":
        # Normal view looks toward center (-u); reverse probe looks away (+u).
        probe_camera = _build_probe_camera(
            position=start,
            forward=u,
            captured_cameras=captured_cameras,
            profile=profile,
            config=config,
        )
        depth_result = depth_probe.probe(probe_camera)
        if depth_result.valid:
            margin = (
                safety.clearance * float(config.depth_margin_ratio)
                if safety is not None
                else max(1e-3, 0.02 * max(initial_radius, 1.0))
            )
            move = max(0.0, float(depth_result.depth) - margin)
            target_radius = min(r_max, initial_radius + move)
        elif config.invalid_depth_behavior == "use_radius_max":
            target_radius = r_max
            notes.append("invalid reverse depth -> radius_max fallback")
        else:
            target_radius = initial_radius
            notes.append("invalid reverse depth -> keep radius prior")
        target = center + target_radius * u
    else:
        raise ValueError(
            f"Unknown outside_placement_strategy: {config.outside_placement_strategy}"
        )

    adjusted, path_fraction, _ = _apply_path_safety(start, target, safety, config)
    signed = float(np.clip(_signed_radius(center, adjusted, u), r_min, r_max))
    adjusted = center + signed * u
    return adjusted, signed, depth_result, path_fraction, notes


def _place_inside_out(
    point: AngularGridPoint,
    initial_radius: float,
    bbox: ViewBBox,
    center: np.ndarray,
    captured_cameras: Sequence[Camera],
    profile: SceneProfile,
    config: PoseGenerationConfig,
    depth_probe,
    safety: Optional[PointCloudSafety],
) -> Tuple[np.ndarray, float, DepthProbeResult, Optional[float], List[str]]:
    u = point.direction
    r_min, r_max = [float(x) for x in bbox.generation_radius]
    r_min, r_max = min(r_min, r_max), max(r_min, r_max)
    initial_radius = float(np.clip(initial_radius, r_min, r_max))
    start = center + initial_radius * u
    notes: List[str] = []

    if config.inside_placement_strategy == "prior_only":
        depth_result = NullDepthProbe().probe()
        target_signed = initial_radius
    else:
        # Inside-out normal view points +u, so the reverse probe points toward center (-u).
        probe_camera = _build_probe_camera(
            position=start,
            forward=-u,
            captured_cameras=captured_cameras,
            profile=profile,
            config=config,
        )
        depth_result = depth_probe.probe(probe_camera)

        if not depth_result.valid:
            target_signed = initial_radius
            notes.append("invalid reverse depth -> keep radius prior")
        else:
            margin = (
                safety.clearance * float(config.depth_margin_ratio)
                if safety is not None
                else max(1e-3, 0.02 * max(initial_radius, 1.0))
            )
            usable_move = max(0.0, float(depth_result.depth) - margin)

            if config.inside_placement_strategy == "no_crossing":
                target_signed = max(r_min, initial_radius - usable_move)
            elif config.inside_placement_strategy == "center_crossing_depth":
                opposite_free = usable_move - initial_radius
                extra = (
                    safety.clearance * float(config.center_cross_extra_ratio)
                    if safety is not None
                    else 0.0
                )

                # Cross center only if the rendered free-space continues far enough
                # to place a camera on the opposite side at at least r_min.
                if opposite_free >= r_min + extra:
                    target_signed = -min(r_max, opposite_free)
                    notes.append("reverse depth supports center crossing")
                else:
                    target_signed = max(r_min, initial_radius - usable_move)
                    notes.append("center crossing unsupported -> approach center")
            else:
                raise ValueError(
                    f"Unknown inside_placement_strategy: {config.inside_placement_strategy}"
                )

    target = center + target_signed * u
    adjusted, path_fraction, _ = _apply_path_safety(start, target, safety, config)
    signed = _signed_radius(center, adjusted, u)

    # Preserve endpoint radius semantics.  If a proposed crossing could not reach a
    # valid opposite-shell radius after path clipping, keep the camera on the
    # original side at the closest allowed radius instead of leaving it near zero.
    if signed < 0.0 and abs(signed) < r_min:
        fallback = center + r_min * u
        fallback_adjusted, fallback_fraction, _ = _apply_path_safety(
            start, fallback, safety, config
        )
        adjusted = fallback_adjusted
        path_fraction = fallback_fraction
        signed = max(r_min, _signed_radius(center, adjusted, u))
        notes.append("crossing path too short -> positive-side fallback")

    if signed >= 0.0:
        signed = float(np.clip(signed, r_min, r_max))
    else:
        signed = -float(np.clip(abs(signed), r_min, r_max))
    adjusted = center + signed * u
    return adjusted, signed, depth_result, path_fraction, notes


# -----------------------------------------------------------------------------
# Main generation
# -----------------------------------------------------------------------------

def generate_candidate_poses(
    captured_cameras: Sequence[Camera],
    scene_result: SceneUnderstandingResult,
    point_cloud_points: Optional[np.ndarray] = None,
    depth_probe=None,
    config: Optional[PoseGenerationConfig] = None,
) -> PoseGenerationResult:
    config = config or PoseGenerationConfig()
    profile = scene_result.profile
    depth_probe = depth_probe or NullDepthProbe()

    mode_result = choose_observation_mode(profile, config)
    bbox = resolve_generation_bbox(profile, mode_result.mode)
    view_limits = build_view_limits(profile, bbox, config)
    grid = generate_angular_grid(profile, bbox, config)

    safety = None
    if point_cloud_points is not None and config.geometry.strategy != "none":
        safety = PointCloudSafety(point_cloud_points, config.geometry)

    center = profile.center_fit.center.astype(np.float64)
    candidates: List[GeneratedCandidate] = []
    valid_cameras: List[Camera] = []

    for point in grid:
        initial_radius, radius_confidence, radius_source = resolve_initial_radius(
            point=point,
            mode=mode_result.mode,
            bbox=bbox,
            scene_result=scene_result,
            config=config,
        )
        initial_position = center + initial_radius * point.direction

        initial_clearance = None
        if safety is not None:
            initial_safe, initial_clearance = safety.is_position_safe(initial_position)
            if not initial_safe and config.reject_unsafe_initial_prior:
                candidates.append(
                    GeneratedCandidate(
                        grid_id=point.grid_id,
                        row=point.row,
                        col=point.col,
                        azimuth_deg=point.azimuth_deg,
                        elevation_deg=point.elevation_deg,
                        direction=point.direction,
                        mode=mode_result.mode,
                        initial_radius=initial_radius,
                        initial_radius_source=radius_source,
                        initial_radius_confidence=radius_confidence,
                        final_signed_radius=initial_radius,
                        crossed_center=False,
                        placement_strategy="initial_reject",
                        depth_probe=None,
                        initial_clearance=initial_clearance,
                        final_clearance=initial_clearance,
                        path_safe_fraction=0.0,
                        camera=None,
                        status=CandidateStatus.REJECTED,
                        reject_reason="UNSAFE_INITIAL_RADIUS",
                    )
                )
                continue

        if mode_result.mode == CameraMode.OUTSIDE_IN:
            final_position, signed_radius, depth_result, path_fraction, notes = _place_outside_in(
                point=point,
                initial_radius=initial_radius,
                bbox=bbox,
                center=center,
                captured_cameras=captured_cameras,
                profile=profile,
                config=config,
                depth_probe=depth_probe,
                safety=safety,
            )
            forward = -point.direction
            placement_strategy = config.outside_placement_strategy
        else:
            final_position, signed_radius, depth_result, path_fraction, notes = _place_inside_out(
                point=point,
                initial_radius=initial_radius,
                bbox=bbox,
                center=center,
                captured_cameras=captured_cameras,
                profile=profile,
                config=config,
                depth_probe=depth_probe,
                safety=safety,
            )
            # IMPORTANT: generation policy remains inside-out even after crossing center.
            forward = point.direction
            placement_strategy = config.inside_placement_strategy

        final_clearance = None
        if safety is not None:
            final_safe, final_clearance = safety.is_position_safe(final_position)
            if not final_safe:
                candidates.append(
                    GeneratedCandidate(
                        grid_id=point.grid_id,
                        row=point.row,
                        col=point.col,
                        azimuth_deg=point.azimuth_deg,
                        elevation_deg=point.elevation_deg,
                        direction=point.direction,
                        mode=mode_result.mode,
                        initial_radius=initial_radius,
                        initial_radius_source=radius_source,
                        initial_radius_confidence=radius_confidence,
                        final_signed_radius=signed_radius,
                        crossed_center=bool(signed_radius < 0.0),
                        placement_strategy=placement_strategy,
                        depth_probe=depth_result,
                        initial_clearance=initial_clearance,
                        final_clearance=final_clearance,
                        path_safe_fraction=path_fraction,
                        camera=None,
                        status=CandidateStatus.REJECTED,
                        reject_reason="FINAL_POSITION_TOO_CLOSE_TO_GEOMETRY",
                        notes=notes,
                    )
                )
                continue

        camera = build_generated_camera(
            index=len(valid_cameras),
            position=final_position,
            forward=forward,
            cameras=captured_cameras,
            profile=profile,
            config=config,
        )
        valid_cameras.append(camera)

        candidates.append(
            GeneratedCandidate(
                grid_id=point.grid_id,
                row=point.row,
                col=point.col,
                azimuth_deg=point.azimuth_deg,
                elevation_deg=point.elevation_deg,
                direction=point.direction,
                mode=mode_result.mode,
                initial_radius=initial_radius,
                initial_radius_source=radius_source,
                initial_radius_confidence=radius_confidence,
                final_signed_radius=signed_radius,
                crossed_center=bool(signed_radius < 0.0),
                placement_strategy=placement_strategy,
                depth_probe=depth_result,
                initial_clearance=initial_clearance,
                final_clearance=final_clearance,
                path_safe_fraction=path_fraction,
                camera=camera,
                status=CandidateStatus.VALID,
                notes=notes,
            )
        )

    return PoseGenerationResult(
        mode=mode_result,
        bbox=bbox,
        view_limits=view_limits,
        candidates=candidates,
        valid_cameras=valid_cameras,
        config=config,
    )


def save_pose_generation_result(
    result: PoseGenerationResult,
    output_dir: str,
) -> Dict[str, str]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    view_limits_path = output_dir / "view_limits.json"
    gen_cameras_path = output_dir / "gen_cameras.json"
    meta_path = output_dir / "gen_cameras_meta.json"

    # view_limits is intentionally written first: it is an endpoint contract
    # independent of candidate selection in the next stage.
    with open(view_limits_path, "w", encoding="utf-8") as f:
        json.dump(result.view_limits, f, indent=2)

    save_cameras_json(result.valid_cameras, str(gen_cameras_path))

    meta = {
        "mode": to_jsonable(result.mode),
        "bbox": to_jsonable(result.bbox),
        "config": to_jsonable(asdict(result.config)),
        "num_candidates": len(result.candidates),
        "num_valid": len(result.valid_cameras),
        "num_rejected": len(result.candidates) - len(result.valid_cameras),
        "candidates": to_jsonable(result.candidates),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "view_limits": str(view_limits_path),
        "gen_cameras": str(gen_cameras_path),
        "gen_cameras_meta": str(meta_path),
    }
