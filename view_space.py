#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Spherical scene frame and per-mode view-space bounding boxes.

Three bbox strategies are provided:

``legacy_minmax``
    Camera-derived ordinary min/max in azimuth/elevation/radius plus a fixed
    angular expansion.  This intentionally preserves the wrap-around weakness
    of the old rectangular representation for ablation.

``legacy_view_limits``
    Read the historical ``view_limits.json`` fields (minPhi/maxPhi,
    minTheta/maxTheta, minRadius/maxRadius) and reproduce the old 0.1-radian
    angular expansion behavior.

``circular_robust``
    Default improved strategy: shortest circular azimuth interval, percentile
    elevation/radius bounds, and data-adaptive angular expansion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from viewpoint_framework.scene_types import (
    CameraMode,
    CameraSceneRelation,
    CircularInterval,
    SphericalFrame,
    ViewBBox,
)


EPS = 1e-10

BBOX_STRATEGIES = (
    "legacy_minmax",
    "legacy_view_limits",
    "circular_robust",
)


@dataclass
class ViewSpaceConfig:
    """Configuration for spherical coordinates and per-mode view bbox."""

    strategy: str = "circular_robust"

    # World-space spherical coordinate frame.  Automatic gravity inference is
    # intentionally not hidden inside this first-stage implementation.
    world_up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    azimuth_zero_reference: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    # Legacy camera-derived rectangular bbox.
    legacy_extension_deg: float = float(np.degrees(0.1))

    # Improved robust bbox.
    elevation_percentiles: Tuple[float, float] = (2.0, 98.0)
    radius_percentiles: Tuple[float, float] = (5.0, 95.0)

    extension_mode: str = "adaptive"  # adaptive | fixed
    fixed_extension_deg: float = 5.0
    adaptive_extension_factor: float = 0.5
    min_extension_deg: float = 5.0
    max_extension_deg: float = 15.0

    # Optional radial bbox expansion after observed bounds are estimated.
    radius_extension_ratio: float = 0.0

    min_support_cameras: int = 1


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        raise ValueError(f"Cannot normalize near-zero vector: {vector}")
    return vector / norm


def build_spherical_frame(
    config: ViewSpaceConfig | None = None,
) -> SphericalFrame:
    """Build an orthonormal scene frame for azimuth/elevation coordinates."""

    config = config or ViewSpaceConfig()

    y_axis = _normalize(np.asarray(config.world_up, dtype=np.float64))
    reference = _normalize(
        np.asarray(config.azimuth_zero_reference, dtype=np.float64)
    )

    # Project azimuth-zero reference onto the horizontal plane.
    z_axis = reference - float(np.dot(reference, y_axis)) * y_axis

    if float(np.linalg.norm(z_axis)) <= EPS:
        # Deterministic fallback if requested zero reference is parallel to up.
        candidates = [
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
        ]
        for candidate in candidates:
            projected = candidate - float(np.dot(candidate, y_axis)) * y_axis
            if float(np.linalg.norm(projected)) > EPS:
                z_axis = projected
                break

    z_axis = _normalize(z_axis)
    x_axis = _normalize(np.cross(y_axis, z_axis))

    # Re-orthogonalize z to avoid accumulated numerical error.
    z_axis = _normalize(np.cross(x_axis, y_axis))

    return SphericalFrame(
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        up_source="config",
        reference_source="config",
    )


def direction_to_azimuth_elevation(
    direction: np.ndarray,
    frame: SphericalFrame,
) -> Tuple[float, float]:
    """Convert one world-space unit direction to azimuth/elevation in degrees."""

    direction = _normalize(direction)

    x = float(np.dot(direction, frame.x_axis))
    y = float(np.dot(direction, frame.y_axis))
    z = float(np.dot(direction, frame.z_axis))

    azimuth_deg = float(np.degrees(np.arctan2(x, z)))
    elevation_deg = float(np.degrees(np.arcsin(np.clip(y, -1.0, 1.0))))
    return normalize_angle_deg(azimuth_deg), elevation_deg


def azimuth_elevation_to_direction(
    azimuth_deg: float,
    elevation_deg: float,
    frame: SphericalFrame,
) -> np.ndarray:
    """Convert azimuth/elevation in scene frame to a world-space unit direction."""

    azimuth = np.radians(float(azimuth_deg))
    elevation = np.radians(float(elevation_deg))

    local_x = np.cos(elevation) * np.sin(azimuth)
    local_y = np.sin(elevation)
    local_z = np.cos(elevation) * np.cos(azimuth)

    direction = (
        local_x * frame.x_axis
        + local_y * frame.y_axis
        + local_z * frame.z_axis
    )
    return _normalize(direction)


def normalize_angle_deg(angle_deg: float) -> float:
    """Normalize degrees to [-180, 180)."""

    return float((float(angle_deg) + 180.0) % 360.0 - 180.0)


def minimal_circular_interval(
    angles_deg: Sequence[float],
) -> CircularInterval:
    """Smallest positive-direction circular arc containing all angles."""

    if len(angles_deg) == 0:
        raise ValueError("At least one angle is required.")

    angles = np.mod(np.asarray(angles_deg, dtype=np.float64), 360.0)

    if len(angles) == 1:
        value = normalize_angle_deg(float(angles[0]))
        return CircularInterval(
            start_deg=value,
            end_deg=value,
            span_deg=0.0,
            wraps=False,
        )

    sorted_angles = np.sort(angles)
    wrapped = np.concatenate([sorted_angles, sorted_angles[:1] + 360.0])
    gaps = np.diff(wrapped)

    largest_gap_index = int(np.argmax(gaps))
    largest_gap = float(gaps[largest_gap_index])

    start_360 = float(sorted_angles[(largest_gap_index + 1) % len(sorted_angles)])
    span = float(max(0.0, 360.0 - largest_gap))
    end_360 = (start_360 + span) % 360.0

    start = normalize_angle_deg(start_360)
    end = normalize_angle_deg(end_360)
    wraps = bool(span > EPS and end < start)

    return CircularInterval(
        start_deg=start,
        end_deg=end,
        span_deg=span,
        wraps=wraps,
    )


def expand_circular_interval(
    interval: CircularInterval,
    extension_deg: float,
) -> CircularInterval:
    """Expand a circular interval equally on both sides."""

    extension_deg = max(float(extension_deg), 0.0)
    new_span = min(360.0, interval.span_deg + 2.0 * extension_deg)

    if new_span >= 360.0 - 1e-9:
        return CircularInterval(
            start_deg=-180.0,
            end_deg=180.0 - 1e-9,
            span_deg=360.0,
            wraps=False,
        )

    start = normalize_angle_deg(interval.start_deg - extension_deg)
    end = normalize_angle_deg(start + new_span)
    wraps = bool(new_span > EPS and end < start)

    return CircularInterval(
        start_deg=start,
        end_deg=end,
        span_deg=float(new_span),
        wraps=wraps,
    )


def circular_interval_contains(
    interval: CircularInterval,
    angle_deg: float,
    atol: float = 1e-8,
) -> bool:
    """Check whether angle lies on the positive arc represented by interval."""

    if interval.span_deg >= 360.0 - atol:
        return True

    angle = normalize_angle_deg(angle_deg)
    start = interval.start_deg
    offset = (angle - start) % 360.0
    return bool(offset <= interval.span_deg + atol)


def sample_circular_interval(
    interval: CircularInterval,
    step_deg: float,
    include_end: bool = True,
) -> np.ndarray:
    """Sample a circular interval along its positive direction."""

    if step_deg <= 0:
        raise ValueError(f"step_deg must be >0, got {step_deg}")

    if interval.span_deg <= EPS:
        return np.asarray([interval.start_deg], dtype=np.float64)

    count = max(1, int(np.floor(interval.span_deg / step_deg)))
    offsets = np.arange(count + 1, dtype=np.float64) * step_deg
    offsets = offsets[offsets <= interval.span_deg + EPS]

    if include_end and (
        len(offsets) == 0
        or abs(float(offsets[-1]) - interval.span_deg) > 1e-8
    ):
        offsets = np.concatenate([offsets, [interval.span_deg]])

    return np.asarray(
        [normalize_angle_deg(interval.start_deg + offset) for offset in offsets],
        dtype=np.float64,
    )


def angular_distance_deg(
    direction_a: np.ndarray,
    direction_b: np.ndarray,
) -> float:
    """Great-circle angular distance between two directions, in degrees."""

    a = _normalize(direction_a)
    b = _normalize(direction_b)
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _median_nearest_angular_spacing(
    directions: np.ndarray,
) -> Optional[float]:
    """Median nearest-neighbor great-circle spacing in degrees."""

    if len(directions) < 2:
        return None

    directions = np.asarray(directions, dtype=np.float64)
    directions = directions / np.maximum(
        np.linalg.norm(directions, axis=1, keepdims=True),
        EPS,
    )

    cosine = np.clip(directions @ directions.T, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosine))
    np.fill_diagonal(angles, np.inf)

    nearest = np.min(angles, axis=1)
    nearest = nearest[np.isfinite(nearest)]
    if len(nearest) == 0:
        return None
    return float(np.median(nearest))


def _relation_subset(
    relations: Sequence[CameraSceneRelation],
    mode: CameraMode,
) -> List[CameraSceneRelation]:
    return [relation for relation in relations if relation.mode == mode]


def attach_spherical_coordinates(
    relations: Sequence[CameraSceneRelation],
    frame: SphericalFrame,
) -> None:
    """Populate azimuth/elevation fields in-place for all non-zero radial dirs."""

    for relation in relations:
        if relation.radius <= EPS:
            relation.azimuth_deg = None
            relation.elevation_deg = None
            continue

        azimuth_deg, elevation_deg = direction_to_azimuth_elevation(
            relation.radial_direction,
            frame,
        )
        relation.azimuth_deg = azimuth_deg
        relation.elevation_deg = elevation_deg


def _percentile_pair(
    values: np.ndarray,
    percentiles: Tuple[float, float],
) -> Tuple[float, float]:
    low, high = percentiles
    if not (0.0 <= low <= high <= 100.0):
        raise ValueError(f"Invalid percentile pair: {percentiles}")

    result = np.percentile(values, [low, high])
    return float(result[0]), float(result[1])


def _legacy_rectangular_azimuth(
    azimuths_deg: np.ndarray,
) -> CircularInterval:
    """Represent ordinary min/max azimuth as a non-circular legacy interval."""

    az_min = float(np.min(azimuths_deg))
    az_max = float(np.max(azimuths_deg))
    return CircularInterval(
        start_deg=az_min,
        end_deg=az_max,
        span_deg=float(az_max - az_min),
        wraps=False,
    )


def _legacy_expand_rectangular_azimuth(
    interval: CircularInterval,
    extension_deg: float,
) -> CircularInterval:
    """Legacy arithmetic expansion; intentionally does not fix angle wrapping."""

    start = float(interval.start_deg - extension_deg)
    end = float(interval.end_deg + extension_deg)
    return CircularInterval(
        start_deg=start,
        end_deg=end,
        span_deg=float(end - start),
        wraps=False,
    )


def load_legacy_view_limits(path: str) -> Dict:
    """Load historical view_limits.json."""

    path_obj = Path(path).expanduser().resolve()
    if not path_obj.is_file():
        raise FileNotFoundError(f"view_limits.json not found: {path_obj}")

    with open(path_obj, "r", encoding="utf-8") as f:
        data = json.load(f)

    required = (
        "minPhi",
        "maxPhi",
        "minTheta",
        "maxTheta",
        "minRadius",
        "maxRadius",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Missing view_limits keys: {missing}")

    return data


def estimate_view_bbox(
    relations: Sequence[CameraSceneRelation],
    mode: CameraMode,
    frame: SphericalFrame,
    config: ViewSpaceConfig | None = None,
    legacy_view_limits: Optional[Dict] = None,
) -> Optional[ViewBBox]:
    """Estimate one mode-specific view bbox using the selected strategy."""

    config = config or ViewSpaceConfig()

    if config.strategy not in BBOX_STRATEGIES:
        raise ValueError(
            f"Unknown bbox strategy {config.strategy!r}; choices={BBOX_STRATEGIES}"
        )

    subset = _relation_subset(relations, mode)
    if len(subset) < config.min_support_cameras:
        return None

    # Ensure spherical coordinates reflect the requested frame.
    for relation in subset:
        az, el = direction_to_azimuth_elevation(relation.radial_direction, frame)
        relation.azimuth_deg = az
        relation.elevation_deg = el

    azimuths = np.asarray([r.azimuth_deg for r in subset], dtype=np.float64)
    elevations = np.asarray([r.elevation_deg for r in subset], dtype=np.float64)
    radii = np.asarray([r.radius for r in subset], dtype=np.float64)
    directions = np.stack([r.radial_direction for r in subset], axis=0)

    notes: List[str] = []

    if config.strategy == "legacy_view_limits":
        if legacy_view_limits is None:
            raise ValueError(
                "bbox strategy 'legacy_view_limits' requires legacy_view_limits data."
            )

        # Historical naming:
        #   Phi   -> yaw / azimuth
        #   Theta -> polar angle from +Y
        # New elevation = 90 - Theta.
        observed_azimuth = CircularInterval(
            start_deg=float(legacy_view_limits["minPhi"]),
            end_deg=float(legacy_view_limits["maxPhi"]),
            span_deg=float(
                legacy_view_limits["maxPhi"] - legacy_view_limits["minPhi"]
            ),
            wraps=False,
        )
        observed_elevation = (
            float(90.0 - legacy_view_limits["maxTheta"]),
            float(90.0 - legacy_view_limits["minTheta"]),
        )
        observed_radius = (
            float(legacy_view_limits["minRadius"]),
            float(legacy_view_limits["maxRadius"]),
        )

        extension = config.legacy_extension_deg
        generation_azimuth = _legacy_expand_rectangular_azimuth(
            observed_azimuth,
            extension,
        )
        generation_elevation = (
            observed_elevation[0] - extension,
            observed_elevation[1] + extension,
        )
        generation_radius = observed_radius
        radius_extension = 0.0
        notes.append(
            "Legacy view_limits fields and fixed historical angular extension used."
        )

    elif config.strategy == "legacy_minmax":
        observed_azimuth = _legacy_rectangular_azimuth(azimuths)
        observed_elevation = (
            float(np.min(elevations)),
            float(np.max(elevations)),
        )
        observed_radius = (
            float(np.min(radii)),
            float(np.max(radii)),
        )

        extension = config.legacy_extension_deg
        generation_azimuth = _legacy_expand_rectangular_azimuth(
            observed_azimuth,
            extension,
        )
        generation_elevation = (
            observed_elevation[0] - extension,
            observed_elevation[1] + extension,
        )

        radius_extension = config.radius_extension_ratio * max(
            float(np.median(radii)),
            EPS,
        )
        generation_radius = (
            max(0.0, observed_radius[0] - radius_extension),
            observed_radius[1] + radius_extension,
        )
        notes.append(
            "Legacy camera min/max bbox intentionally keeps ordinary azimuth min/max."
        )

    else:
        observed_azimuth = minimal_circular_interval(azimuths)
        observed_elevation = _percentile_pair(
            elevations,
            config.elevation_percentiles,
        )
        observed_radius = _percentile_pair(
            radii,
            config.radius_percentiles,
        )

        if config.extension_mode == "fixed":
            extension = config.fixed_extension_deg
        elif config.extension_mode == "adaptive":
            typical_spacing = _median_nearest_angular_spacing(directions)
            if typical_spacing is None:
                extension = config.fixed_extension_deg
                notes.append(
                    "Adaptive extension unavailable with <2 support cameras; fixed extension used."
                )
            else:
                extension = float(
                    np.clip(
                        config.adaptive_extension_factor * typical_spacing,
                        config.min_extension_deg,
                        config.max_extension_deg,
                    )
                )
                notes.append(
                    f"Adaptive extension from median nearest angular spacing={typical_spacing:.3f} deg."
                )
        else:
            raise ValueError(
                f"Unknown extension_mode={config.extension_mode!r}; expected 'fixed' or 'adaptive'."
            )

        generation_azimuth = expand_circular_interval(
            observed_azimuth,
            extension,
        )
        generation_elevation = (
            max(-89.9, observed_elevation[0] - extension),
            min(89.9, observed_elevation[1] + extension),
        )

        radius_extension = config.radius_extension_ratio * max(
            float(np.median(radii)),
            EPS,
        )
        generation_radius = (
            max(0.0, observed_radius[0] - radius_extension),
            observed_radius[1] + radius_extension,
        )

    return ViewBBox(
        mode=mode,
        strategy=config.strategy,
        camera_indices=[int(r.camera_index) for r in subset],
        observed_azimuth=observed_azimuth,
        observed_elevation_deg=observed_elevation,
        observed_radius=observed_radius,
        generation_azimuth=generation_azimuth,
        generation_elevation_deg=generation_elevation,
        generation_radius=generation_radius,
        angular_extension_deg=float(extension),
        radius_extension=float(radius_extension),
        support_camera_count=len(subset),
        notes=notes,
    )


def estimate_mode_view_bboxes(
    relations: Sequence[CameraSceneRelation],
    frame: SphericalFrame,
    config: ViewSpaceConfig | None = None,
    legacy_view_limits: Optional[Dict] = None,
) -> Dict[str, ViewBBox]:
    """Estimate separate outside-in and inside-out view bboxes."""

    config = config or ViewSpaceConfig()
    result: Dict[str, ViewBBox] = {}

    for mode in (CameraMode.OUTSIDE_IN, CameraMode.INSIDE_OUT):
        bbox = estimate_view_bbox(
            relations=relations,
            mode=mode,
            frame=frame,
            config=config,
            legacy_view_limits=legacy_view_limits,
        )
        if bbox is not None:
            result[mode.value] = bbox

    return result
