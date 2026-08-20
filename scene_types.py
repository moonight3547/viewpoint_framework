#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared data models for scene-understanding and viewpoint constraints.

This module deliberately contains *data only* plus light-weight serialization
helpers.  Geometry, fitting, classification, and visualization logic live in
other modules so individual algorithmic pieces can be ablated independently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class CameraMode(str, Enum):
    """Per-camera acquisition mode relative to the fitted scene center."""

    OUTSIDE_IN = "outside_in"
    INSIDE_OUT = "inside_out"
    AMBIGUOUS = "ambiguous"
    OUTLIER = "outlier"


class GlobalCollectionMode(str, Enum):
    """Global acquisition-mode summary for one camera sequence."""

    OUTSIDE_IN = "outside_in"
    INSIDE_OUT = "inside_out"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class CenterFitResult:
    """Result of common sight-line center estimation."""

    center: np.ndarray
    residuals: np.ndarray
    robust_weights: np.ndarray
    inlier_mask: np.ndarray

    strategy: str
    solver: str
    converged: bool
    iterations: int

    condition_number: float
    singular_values: np.ndarray
    median_residual: float
    mad_residual: float

    notes: List[str] = field(default_factory=list)


@dataclass
class CameraSceneRelation:
    """Geometric relation between one captured camera and the scene center."""

    camera_index: int

    position: np.ndarray
    forward: np.ndarray

    radius: float
    radial_direction: np.ndarray

    lambda_center: float
    sight_residual: float
    residual_ratio: float
    alignment_deg: float

    robust_weight: float
    mode: CameraMode
    confidence: float

    azimuth_deg: Optional[float] = None
    elevation_deg: Optional[float] = None


@dataclass
class ModeSummary:
    """Weighted global statistics of per-camera acquisition modes."""

    dominant_mode: GlobalCollectionMode
    dominant_confidence: float

    outside_in_count: int
    inside_out_count: int
    ambiguous_count: int
    outlier_count: int

    outside_in_weight: float
    inside_out_weight: float

    outside_in_ratio: float
    inside_out_ratio: float

    strategy: str


@dataclass
class SphericalFrame:
    """World-space orthonormal basis used for azimuth/elevation coordinates.

    Convention:
        +Y_s = up
        +Z_s = azimuth 0 reference direction on the horizontal plane
        +X_s = +90 degree azimuth direction

    For the common world-up [0, 1, 0] and reference [0, 0, 1], this becomes
    the identity world basis: X_s=[1,0,0], Y_s=[0,1,0], Z_s=[0,0,1].
    """

    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray

    up_source: str = "config"
    reference_source: str = "config"


@dataclass
class CircularInterval:
    """Circular interval in degrees.

    start_deg/end_deg are normalized to [-180, 180).  ``span_deg`` is always
    non-negative.  ``wraps`` indicates that traversing from start to end in
    the positive angular direction crosses +180/-180.
    """

    start_deg: float
    end_deg: float
    span_deg: float
    wraps: bool


@dataclass
class ViewBBox:
    """Observed and generation view-space constraints for one camera mode."""

    mode: CameraMode
    strategy: str
    camera_indices: List[int]

    observed_azimuth: CircularInterval
    observed_elevation_deg: Tuple[float, float]
    observed_radius: Tuple[float, float]

    generation_azimuth: CircularInterval
    generation_elevation_deg: Tuple[float, float]
    generation_radius: Tuple[float, float]

    angular_extension_deg: float
    radius_extension: float

    support_camera_count: int
    notes: List[str] = field(default_factory=list)


@dataclass
class RadiusEstimate:
    """Directional camera-radius estimate for one radial direction."""

    nominal: float
    low: float
    high: float
    confidence: float

    valid: bool
    strategy: str
    source: str

    nearest_support_angle_deg: Optional[float] = None
    effective_neighbors: Optional[float] = None
    neighbor_camera_indices: List[int] = field(default_factory=list)
    neighbor_angles_deg: List[float] = field(default_factory=list)

    geometry_point_count: Optional[int] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class RadiusFieldSample:
    """Serializable directional radius-field sample."""

    mode: CameraMode
    azimuth_deg: float
    elevation_deg: float
    direction: np.ndarray
    position: Optional[np.ndarray]
    estimate: RadiusEstimate


@dataclass
class SceneProfile:
    """Serializable output of the scene-understanding stage."""

    center_fit: CenterFitResult
    mode_summary: ModeSummary
    coordinate_frame: SphericalFrame

    camera_relations: List[CameraSceneRelation]
    view_bboxes: Dict[str, ViewBBox]
    radius_field_samples: Dict[str, List[RadiusFieldSample]]

    strategy_config: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


def to_jsonable(value: Any) -> Any:
    """Recursively convert framework data structures to JSON-compatible data."""

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Enum):
        return value.value

    if is_dataclass(value):
        return {
            key: to_jsonable(val)
            for key, val in asdict(value).items()
        }

    if isinstance(value, dict):
        return {
            str(key): to_jsonable(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]

    return value
