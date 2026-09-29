#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared Stage-2 contracts and V3.2/V3.3 candidate-generation dispatch."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from viewpoint_framework.utils.cameras import Camera
from viewpoint_framework.geometry_safety import GeometrySafetyConfig
from viewpoint_framework.gs_depth_probe import DepthProbeResult
from viewpoint_framework.gs_renderer import RendererNearPlaneConfig
from viewpoint_framework.height_safety import GlobalHeightConfig, LocalHeightConfig
from viewpoint_framework.stage2.config import GridConfig, RadiusConfig, RendererConfig
from viewpoint_framework.skybox_detection import SkyboxDetectionConfig
from viewpoint_framework.scene_types import (
    CameraMode,
    CameraSceneRelation,
    GlobalCollectionMode,
    SceneProfile,
    ViewBBox,
    to_jsonable,
)
from viewpoint_framework.scene_understanding import SceneUnderstandingResult
from viewpoint_framework.trajectory_safe_field import TrajectorySafeFieldConfig
from viewpoint_framework.view_space import (
    azimuth_elevation_to_direction,
    direction_to_azimuth_elevation,
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

    version: str = "3.2"

    # Default observation mode.
    mode_strategy: str = "binary_count_majority"
    forced_mode: Optional[str] = None          # outside_in | inside_out

    # Angular grid.
    grid_strategy: str = "uniform"             # uniform | cos_elevation
    azimuth_step_deg: float = 20.0
    elevation_step_deg: float = 20.0
    include_bbox_end: bool = True

    # Reconstruct Stage-2 spatial support from a fresh binary classification
    # around the refined Stage-1 center.
    bbox_strategy: str = "scene_profile"  # scene_profile | binary_dominant
    azimuth_extension_ratio: float = 0.10
    elevation_extension_ratio: float = 0.10
    bbox_elevation_percentiles: Tuple[float, float] = (2.0, 98.0)
    bbox_radius_percentiles: Tuple[float, float] = (5.0, 95.0)

    # Camera placement.
    outside_placement_strategy: str = "depth_backoff"  # prior_only | depth_backoff
    inside_placement_strategy: str = "center_crossing_depth"  # prior_only | no_crossing | center_crossing_depth
    depth_margin_ratio: float = 1.0             # multiplied by geometry safety clearance
    center_cross_extra_ratio: float = 0.5       # extra clearance beyond center

    # Point-cloud hard safety.
    geometry: GeometrySafetyConfig = field(default_factory=GeometrySafetyConfig)
    use_path_safety: bool = True
    reject_unsafe_initial_prior: bool = True

    intrinsics_strategy: str = "first"  # first | first_scaled | median_scaled
    focal_ratio: float = 1.0
    center_principal_point: bool = False

    # Endpoint view-limits export.
    view_limits_strategy: str = "generation_bbox"  # generation_bbox | observed_bbox | observed_dominant_radius
    view_limits_unwrap_azimuth: bool = True
    view_limits_radius_percentiles: Tuple[float, float] = (40.0, 60.0)

    console_log_candidates: bool = False

    position_strategy: str = "trajectory_safe_field"
    trajectory_safe_field: TrajectorySafeFieldConfig = field(default_factory=TrajectorySafeFieldConfig)
    adjustment_radius_max: Optional[float] = None
    skybox: SkyboxDetectionConfig = field(default_factory=SkyboxDetectionConfig)
    renderer_near_plane: RendererNearPlaneConfig = field(default_factory=RendererNearPlaneConfig)
    local_height: LocalHeightConfig = field(default_factory=LocalHeightConfig)
    global_height: GlobalHeightConfig = field(default_factory=GlobalHeightConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    radius: RadiusConfig = field(default_factory=RadiusConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    # V3.3 keeps its expanded angular proposal, while placement can be
    # independently pinned to the previously validated V3.2 radius/height path.
    v33_placement_strategy: str = "v3_2"  # v3_2 | v3_3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PoseGenerationConfig":
        payload = dict(data)
        geometry_data = payload.pop("geometry", {})
        trajectory_data = payload.pop("trajectory_safe_field", {})
        skybox_data = payload.pop("skybox", {})
        near_plane_data = payload.pop("renderer_near_plane", {})
        local_height_data = payload.pop("local_height", {})
        global_height_data = payload.pop("global_height", {})
        grid_data = payload.pop("grid", {})
        radius_data = payload.pop("radius", {})
        renderer_data = payload.pop("renderer", {})
        cfg = cls(**payload)
        cfg.geometry = GeometrySafetyConfig(**geometry_data)
        cfg.trajectory_safe_field = TrajectorySafeFieldConfig(**trajectory_data)
        cfg.skybox = SkyboxDetectionConfig(**skybox_data)
        cfg.renderer_near_plane = RendererNearPlaneConfig(**near_plane_data)
        cfg.local_height = LocalHeightConfig(**local_height_data)
        cfg.global_height = GlobalHeightConfig(**global_height_data)
        cfg.grid = GridConfig(**grid_data)
        cfg.radius = RadiusConfig(**radius_data)
        cfg.renderer = RendererConfig(**renderer_data)
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
    geometry_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PoseGenerationResult:
    mode: ObservationModeResult
    bbox: ViewBBox
    view_limits: Dict[str, Any]
    candidates: List[GeneratedCandidate]
    valid_cameras: List[Camera]
    config: PoseGenerationConfig
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    renderer_metadata: Dict[str, Any] = field(default_factory=dict)
    placement_metadata: Dict[str, Any] = field(default_factory=dict)


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

    if strategy == "count_majority":
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


def classify_stage2_binary(
    cameras: Sequence[Camera],
    center: np.ndarray,
) -> List[CameraSceneRelation]:
    """Assign every finite captured pose to Outside-In or Inside-Out.

    Unlike Stage 1, this intentionally has no ambiguous/outlier acquisition
    labels.  Stage-1 robust weights remain useful for understanding and center
    refinement, while Stage 2 needs complete spatial support.
    """

    center = np.asarray(center, dtype=np.float64)
    relations: List[CameraSceneRelation] = []
    for camera in cameras:
        position = np.asarray(camera.position, dtype=np.float64)
        forward = np.asarray(camera.forward, dtype=np.float64)
        forward /= max(float(np.linalg.norm(forward)), EPS)
        center_vec = center - position
        radius = float(np.linalg.norm(center_vec))
        if not np.isfinite(radius) or radius <= EPS:
            raise ValueError(
                f"Camera {camera.index} is at/invalid relative to the refined scene center."
            )
        radial = (position - center) / radius
        lam = float(np.dot(forward, center_vec))
        residual = float(np.linalg.norm(center_vec - lam * forward))
        alignment = float(
            np.degrees(np.arccos(np.clip(abs(lam) / radius, 0.0, 1.0)))
        )
        mode = CameraMode.OUTSIDE_IN if lam > 0.0 else CameraMode.INSIDE_OUT
        relations.append(
            CameraSceneRelation(
                camera_index=int(camera.index),
                position=position,
                forward=forward,
                radius=radius,
                radial_direction=radial,
                lambda_center=lam,
                sight_residual=residual,
                residual_ratio=residual / radius,
                alignment_deg=alignment,
                robust_weight=1.0,
                mode=mode,
                confidence=1.0,
            )
        )
    return relations


def choose_binary_count_majority(
    relations: Sequence[CameraSceneRelation],
) -> ObservationModeResult:
    outside = sum(r.mode == CameraMode.OUTSIDE_IN for r in relations)
    inside = len(relations) - outside
    mode = CameraMode.INSIDE_OUT if inside > outside else CameraMode.OUTSIDE_IN
    winner = max(outside, inside)
    return ObservationModeResult(
        mode=mode,
        confidence=winner / float(max(len(relations), 1)),
        strategy="binary_count_majority",
        note=f"all captured poses binary-classified: outside={outside}, inside={inside}",
    )


def _percentile_bounds(
    values: np.ndarray,
    percentiles: Tuple[float, float],
    name: str,
) -> Tuple[float, float]:
    low, high = [float(x) for x in percentiles]
    if not (0.0 <= low <= high <= 100.0):
        raise ValueError(f"Invalid {name} percentiles: {percentiles}")
    result = np.percentile(np.asarray(values, dtype=np.float64), [low, high])
    return float(result[0]), float(result[1])


def build_dominant_bbox(
    relations: Sequence[CameraSceneRelation],
    mode: CameraMode,
    profile: SceneProfile,
    config: PoseGenerationConfig,
) -> Tuple[ViewBBox, List[CameraSceneRelation]]:
    support = [r for r in relations if r.mode == mode]
    if not support:
        raise ValueError(f"Binary classification has no {mode.value} support.")

    frame = profile.coordinate_frame
    azimuths: List[float] = []
    elevations: List[float] = []
    radii: List[float] = []
    for relation in support:
        az, el = direction_to_azimuth_elevation(
            relation.radial_direction, frame
        )
        relation.azimuth_deg = float(az)
        relation.elevation_deg = float(el)
        azimuths.append(float(az))
        elevations.append(float(el))
        radii.append(float(relation.radius))

    observed_azimuth = minimal_circular_interval(azimuths)
    az_extension = max(
        0.0,
        float(config.azimuth_extension_ratio) * observed_azimuth.span_deg,
    )
    # expand_circular_interval correctly keeps arcs >180 degrees expanding and
    # saturates only when the full 360-degree circle has been reached.
    generation_azimuth = expand_circular_interval(observed_azimuth, az_extension)

    observed_elevation = _percentile_bounds(
        np.asarray(elevations),
        config.bbox_elevation_percentiles,
        "bbox elevation",
    )
    elevation_span = max(0.0, observed_elevation[1] - observed_elevation[0])
    elevation_extension = max(
        0.0,
        float(config.elevation_extension_ratio) * elevation_span,
    )
    generation_elevation = (
        max(-89.9, observed_elevation[0] - elevation_extension),
        min(89.9, observed_elevation[1] + elevation_extension),
    )

    observed_radius = _percentile_bounds(
        np.asarray(radii), config.bbox_radius_percentiles, "bbox radius"
    )
    if observed_radius[1] < observed_radius[0] + EPS:
        observed_radius = (
            observed_radius[0],
            observed_radius[0] + max(1e-3, 0.05 * max(observed_radius[0], 1.0)),
        )

    bbox = ViewBBox(
        mode=mode,
        strategy="v2_binary_dominant_robust",
        camera_indices=[int(r.camera_index) for r in support],
        observed_azimuth=observed_azimuth,
        observed_elevation_deg=observed_elevation,
        observed_radius=observed_radius,
        generation_azimuth=generation_azimuth,
        generation_elevation_deg=generation_elevation,
        generation_radius=observed_radius,
        angular_extension_deg=float(az_extension),
        radius_extension=0.0,
        support_camera_count=len(support),
        notes=[
            f"Azimuth extension per side={config.azimuth_extension_ratio:.4f}*span.",
            f"Elevation extension per side={config.elevation_extension_ratio:.4f}*span "
            f"({elevation_extension:.4f} deg).",
        ],
    )
    return bbox, support


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
    dominant_relations: Optional[Sequence[CameraSceneRelation]] = None,
) -> Dict[str, Any]:
    if config.view_limits_strategy == "generation_bbox":
        az = bbox.generation_azimuth
        elevation_min, elevation_max = bbox.generation_elevation_deg
        radius_min, radius_max = bbox.generation_radius
    elif config.view_limits_strategy == "observed_bbox":
        az = bbox.observed_azimuth
        elevation_min, elevation_max = bbox.observed_elevation_deg
        radius_min, radius_max = bbox.observed_radius
    elif config.view_limits_strategy == "observed_dominant_radius":
        if not dominant_relations:
            raise ValueError(
                "observed_dominant_radius requires dominant binary relations."
            )
        az = bbox.observed_azimuth
        elevation_min, elevation_max = bbox.observed_elevation_deg
        radius_min, radius_max = _percentile_bounds(
            np.asarray([r.radius for r in dominant_relations], dtype=np.float64),
            config.view_limits_radius_percentiles,
            "view-limits radius",
        )
    else:
        raise ValueError(
            f"Unknown view_limits_strategy: {config.view_limits_strategy}"
        )

    if config.view_limits_unwrap_azimuth:
        min_phi, max_phi = _interval_to_unwrapped_bounds(az)
    else:
        min_phi, max_phi = float(az.start_deg), float(az.end_deg)

    # Endpoint convention: Theta is polar angle = 90 - elevation.
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
            half_open_full_circle=str(config.version).startswith("3.3"),
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
# V3.2 / V3.3 candidate generation
# -----------------------------------------------------------------------------

def generate_candidate_poses(
    captured_cameras: Sequence[Camera],
    scene_result: SceneUnderstandingResult,
    point_cloud_points: Optional[np.ndarray] = None,
    depth_probe=None,
    config: Optional[PoseGenerationConfig] = None,
) -> PoseGenerationResult:
    """Generate candidates through the supported V3.2 or V3.3 pipeline."""

    config = config or PoseGenerationConfig()
    version = str(config.version)
    if config.position_strategy != "trajectory_safe_field":
        raise ValueError(
            "Only position_strategy='trajectory_safe_field' is supported."
        )
    if not (version.startswith("3.2") or version.startswith("3.3")):
        raise ValueError(
            f"Unsupported pose-generation version {config.version!r}; "
            "supported versions are 3.2 and 3.3."
        )

    profile = scene_result.profile
    generation_relations: Optional[List[CameraSceneRelation]] = None
    dominant_relations: Optional[List[CameraSceneRelation]] = None
    if config.mode_strategy == "binary_count_majority":
        generation_relations = classify_stage2_binary(
            captured_cameras, profile.center_fit.center
        )
        mode_result = choose_binary_count_majority(generation_relations)
    else:
        mode_result = choose_observation_mode(profile, config)

    if config.bbox_strategy == "binary_dominant":
        if generation_relations is None:
            raise ValueError(
                "bbox_strategy='binary_dominant' requires "
                "mode_strategy='binary_count_majority'."
            )
        bbox, dominant_relations = build_dominant_bbox(
            generation_relations, mode_result.mode, profile, config
        )
    elif config.bbox_strategy == "scene_profile":
        bbox = resolve_generation_bbox(profile, mode_result.mode)
    else:
        raise ValueError(f"Unknown bbox_strategy: {config.bbox_strategy}")

    v33_view_domain = None
    if version.startswith("3.3"):
        from viewpoint_framework.stage2.angular_grid import build_v33_bbox

        config.azimuth_step_deg = float(config.grid.azimuth_step_deg)
        config.elevation_step_deg = float(config.grid.elevation_step_deg)
        bbox, v33_view_domain = build_v33_bbox(
            captured_cameras, bbox, profile, mode_result.mode, config.grid
        )

    view_limits = build_view_limits(
        profile, bbox, config, dominant_relations=dominant_relations
    )
    grid = generate_angular_grid(profile, bbox, config)

    if version.startswith("3.2"):
        from viewpoint_framework.pose_generation_v32 import generate_v32_candidates

        return generate_v32_candidates(
            captured_cameras,
            profile,
            bbox,
            view_limits,
            grid,
            mode_result,
            point_cloud_points,
            depth_probe,
            config,
        )

    if config.v33_placement_strategy == "v3_2":
        from viewpoint_framework.pose_generation_v32 import generate_v32_candidates

        result = generate_v32_candidates(
            captured_cameras,
            profile,
            bbox,
            view_limits,
            grid,
            mode_result,
            point_cloud_points,
            depth_probe,
            config,
        )
        result.diagnostics.update(
            {
                "version": "3.3_grid_v3.2_placement",
                "angular_grid_version": "3.3",
                "placement_version": "3.2",
            }
        )
        result.placement_metadata.update(
            {
                "version": "3.3_grid_v3.2_placement",
                "angular_grid_version": "3.3",
                "placement_version": "3.2",
                "view_domain": to_jsonable(v33_view_domain),
            }
        )
        return result

    if config.v33_placement_strategy == "v3_3":
        from viewpoint_framework.stage2.pipeline import generate_v33_candidates

        return generate_v33_candidates(
            captured_cameras,
            profile,
            bbox,
            view_limits,
            grid,
            mode_result,
            point_cloud_points,
            depth_probe,
            config,
            v33_view_domain,
        )

    raise ValueError(
        "v33_placement_strategy must be 'v3_2' or 'v3_3', got "
        f"{config.v33_placement_strategy!r}"
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
        "diagnostics": to_jsonable(result.diagnostics),
        "renderer": to_jsonable(result.renderer_metadata),
        "skybox": to_jsonable(result.renderer_metadata.get("skybox", {})),
        "placement": to_jsonable(result.placement_metadata),
        "trajectory_columns": to_jsonable(result.placement_metadata.get("trajectory_columns", [])),
        "local_height_columns": to_jsonable(result.placement_metadata.get("local_height_columns", [])),
        "global_height": to_jsonable(result.placement_metadata.get("global_height", {})),
        "candidates": to_jsonable(result.candidates),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "view_limits": str(view_limits_path),
        "gen_cameras": str(gen_cameras_path),
        "gen_cameras_meta": str(meta_path),
    }
