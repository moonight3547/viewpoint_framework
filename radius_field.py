#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Directional camera-radius priors and optional point-cloud geometry bounds.

This module is deliberately strategy-based.  It supports legacy/global camera
radius behavior, local camera interpolation, the previous point-cloud cone
median heuristic, and a hybrid option for ablation.

Important terminology
---------------------
``camera radius prior`` means a radius inferred from observed camera positions.
It is evidence that a nearby captured camera existed safely there, but it does
not prove free space along the whole radial path.

``pointcloud_cone`` reproduces the previous cone-median scene-depth heuristic.
It is retained for comparison and should not be interpreted as an exact ray
first-hit distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from viewpoint_framework.scene_types import (
    CameraMode,
    CameraSceneRelation,
    RadiusEstimate,
)


EPS = 1e-10

RADIUS_STRATEGIES = (
    "legacy_pointcloud_rule",
    "global_median",
    "nearest",
    "angular_knn",
    "pointcloud_cone",
    "hybrid",
)


@dataclass
class RadiusFieldConfig:
    """Configuration for directional radius estimation."""

    strategy: str = "angular_knn"

    # Camera-support interpolation.
    k_neighbors: int = 6
    min_neighbors: int = 3
    sigma_deg: float = 20.0
    max_support_angle_deg: float = 30.0
    low_quantile: float = 0.20
    high_quantile: float = 0.80

    # Confidence mapping.
    support_count_saturation: float = 3.0

    # Legacy point-cloud cone heuristic.
    cone_half_angle_deg: float = 15.0
    cone_min_points: int = 5
    cone_low_quantile: float = 0.20
    cone_high_quantile: float = 0.80

    # Strict-ish previous radius rule after point-cloud cone depth.
    # The old outside-in path used alpha=0.8, then the main loop multiplied
    # radius by another 0.8.  Both factors are retained for ablation.
    legacy_alpha: float = 0.8
    legacy_post_scale: float = 0.8

    # Hybrid camera-prior + geometry-bound behavior.
    hybrid_geometry_alpha: float = 0.8
    hybrid_rule: str = "min"  # min | clamp_high


def _normalize(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= EPS:
        raise ValueError(f"Cannot normalize near-zero direction: {direction}")
    return direction / norm


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    """Weighted scalar quantile with non-negative weights."""

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if len(values) == 0:
        raise ValueError("weighted_quantile requires at least one value.")
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length.")
    if not (0.0 <= quantile <= 1.0):
        raise ValueError(f"quantile must be in [0,1], got {quantile}")

    weights = np.clip(weights, 0.0, None)
    if float(np.sum(weights)) <= EPS:
        return float(np.quantile(values, quantile))

    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]

    cumulative = np.cumsum(weights_sorted)
    threshold = quantile * cumulative[-1]
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    index = min(index, len(values_sorted) - 1)
    return float(values_sorted[index])


def _angular_distances_deg(
    query_direction: np.ndarray,
    support_directions: np.ndarray,
) -> np.ndarray:
    query = _normalize(query_direction)
    support = np.asarray(support_directions, dtype=np.float64)
    support = support / np.maximum(
        np.linalg.norm(support, axis=1, keepdims=True),
        EPS,
    )
    cosine = np.clip(support @ query, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    numerator = float(np.sum(weights)) ** 2
    denominator = float(np.sum(weights ** 2))
    if denominator <= EPS:
        return 0.0
    return numerator / denominator


class DirectionalRadiusField:
    """Query directional radius estimates around one scene center and mode."""

    def __init__(
        self,
        center: np.ndarray,
        mode: CameraMode,
        relations: Sequence[CameraSceneRelation],
        config: RadiusFieldConfig | None = None,
        point_cloud_points: Optional[np.ndarray] = None,
        radius_bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.center = np.asarray(center, dtype=np.float64)
        if self.center.shape != (3,):
            raise ValueError(f"center must have shape [3], got {self.center.shape}")

        self.mode = mode
        self.config = config or RadiusFieldConfig()
        self.radius_bounds = radius_bounds

        if self.config.strategy not in RADIUS_STRATEGIES:
            raise ValueError(
                f"Unknown radius strategy {self.config.strategy!r}; choices={RADIUS_STRATEGIES}"
            )

        support = [relation for relation in relations if relation.mode == mode]
        self.support_relations = support

        if len(support) > 0:
            self.support_directions = np.stack(
                [relation.radial_direction for relation in support],
                axis=0,
            ).astype(np.float64)
            self.support_radii = np.asarray(
                [relation.radius for relation in support],
                dtype=np.float64,
            )
            self.support_confidences = np.asarray(
                [max(relation.confidence, 0.0) for relation in support],
                dtype=np.float64,
            )
            self.support_indices = np.asarray(
                [relation.camera_index for relation in support],
                dtype=np.int64,
            )
        else:
            self.support_directions = np.empty((0, 3), dtype=np.float64)
            self.support_radii = np.empty((0,), dtype=np.float64)
            self.support_confidences = np.empty((0,), dtype=np.float64)
            self.support_indices = np.empty((0,), dtype=np.int64)

        if point_cloud_points is None:
            self.point_cloud_points = None
            self.point_cloud_relative = None
            self.point_cloud_radius = None
        else:
            points = np.asarray(point_cloud_points, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(
                    f"point_cloud_points must have shape [N,3], got {points.shape}"
                )
            finite = np.all(np.isfinite(points), axis=1)
            self.point_cloud_points = points[finite]
            # Cache center-relative geometry once. Point-cloud radius strategies
            # may be queried at many angular samples, so recomputing this per
            # direction is unnecessarily expensive for million-point clouds.
            self.point_cloud_relative = self.point_cloud_points - self.center[None, :]
            self.point_cloud_radius = np.linalg.norm(self.point_cloud_relative, axis=1)

    @property
    def num_support_cameras(self) -> int:
        return int(len(self.support_relations))

    def query(self, direction: np.ndarray) -> RadiusEstimate:
        """Query the configured strategy at one world-space radial direction."""

        direction = _normalize(direction)

        strategy = self.config.strategy

        if strategy == "legacy_pointcloud_rule":
            return self._query_legacy_pointcloud_rule(direction)
        if strategy == "global_median":
            return self._query_global_median(direction)
        if strategy == "nearest":
            return self._query_nearest(direction)
        if strategy == "angular_knn":
            return self._query_angular_knn(direction)
        if strategy == "pointcloud_cone":
            return self._query_pointcloud_cone(direction)
        if strategy == "hybrid":
            return self._query_hybrid(direction)

        raise AssertionError(f"Unhandled radius strategy: {strategy}")

    def _invalid(
        self,
        strategy: str,
        source: str,
        note: str,
    ) -> RadiusEstimate:
        return RadiusEstimate(
            nominal=0.0,
            low=0.0,
            high=0.0,
            confidence=0.0,
            valid=False,
            strategy=strategy,
            source=source,
            notes=[note],
        )

    def _legacy_radius_bounds(self) -> Optional[Tuple[float, float]]:
        if self.radius_bounds is not None:
            return (float(self.radius_bounds[0]), float(self.radius_bounds[1]))
        if len(self.support_radii) == 0:
            return None
        return (float(np.min(self.support_radii)), float(np.max(self.support_radii)))

    def _query_legacy_pointcloud_rule(self, direction: np.ndarray) -> RadiusEstimate:
        """Reproduce previous point-cloud depth -> camera-radius rules.

        Outside-in matches the previous point-cloud branch:
            R = cone median depth from center
            determine_radius_outsidein(R, min_r, max_r, alpha)
            radius *= legacy_post_scale

        Inside-out applies the historical ``determine_radius_insideout`` rule,
        but the old implementation obtained R from a 3DGS render at the center.
        The current lightweight codebase has no gsplat dependency, so this
        strategy uses the same point-cloud cone R as an explicit proxy.  The
        output note records this difference.
        """

        geometry = self._query_pointcloud_cone(direction)
        if not geometry.valid:
            geometry.strategy = "legacy_pointcloud_rule"
            return geometry

        bounds = self._legacy_radius_bounds()
        if bounds is None:
            return self._invalid(
                strategy="legacy_pointcloud_rule",
                source="point_cloud+legacy_rule",
                note="Legacy radius rule requires camera/view-space radius bounds.",
            )

        min_r, max_r = bounds
        R = float(geometry.nominal)
        alpha = max(float(self.config.legacy_alpha), EPS)

        if self.mode == CameraMode.OUTSIDE_IN:
            if R > max_r / alpha:
                radius = max_r
                branch = "R > max_r / alpha -> max_r"
            elif R > min_r:
                radius = alpha * R
                branch = "min_r < R <= max_r / alpha -> alpha * R"
            elif R > 0.1 * min_r:
                # Historical fallback kept exactly, including its potentially
                # out-of-bounds 1.35*max_r value before post scaling.
                radius = 1.35 * max_r
                branch = "0.1*min_r < R <= min_r -> 1.35 * max_r (legacy fallback)"
            else:
                return self._invalid(
                    strategy="legacy_pointcloud_rule",
                    source="point_cloud+legacy_rule",
                    note="Legacy outside-in radius rule rejected R <= 0.1*min_r.",
                )
            notes = [
                branch,
                "Outside-in depth source matches the historical point-cloud cone median branch.",
            ]

        else:
            if len(self.support_radii) == 0:
                return self._invalid(
                    strategy="legacy_pointcloud_rule",
                    source="point_cloud+legacy_rule",
                    note="Inside-out legacy rule requires captured-camera median radius.",
                )
            r_constraint = float(np.median(self.support_radii))
            if R > 1.5 * r_constraint:
                radius = r_constraint
                branch = "R > 1.5*r_constraint -> r_constraint"
            elif R > min_r:
                radius = 0.5 * R
                branch = "min_r < R <= 1.5*r_constraint -> 0.5 * R"
            else:
                return self._invalid(
                    strategy="legacy_pointcloud_rule",
                    source="point_cloud+legacy_rule",
                    note="Legacy inside-out radius rule rejected R <= min_r.",
                )
            notes = [
                branch,
                "Inside-out historical rule retained, but R uses point-cloud cone proxy; old code used 3DGS render depth.",
            ]

        radius *= float(self.config.legacy_post_scale)
        notes.append(
            f"Historical main-loop post scale applied: x{self.config.legacy_post_scale:.3f}."
        )

        return RadiusEstimate(
            nominal=float(radius),
            low=float(radius),
            high=float(radius),
            confidence=geometry.confidence,
            valid=True,
            strategy="legacy_pointcloud_rule",
            source="point_cloud+legacy_rule",
            geometry_point_count=geometry.geometry_point_count,
            notes=notes,
        )

    def _query_global_median(self, direction: np.ndarray) -> RadiusEstimate:
        """Legacy-style direction-independent median camera radius."""

        if len(self.support_radii) == 0:
            return self._invalid(
                strategy="global_median",
                source="captured_cameras",
                note=f"No {self.mode.value} support cameras.",
            )

        angles = _angular_distances_deg(direction, self.support_directions)
        nearest = float(np.min(angles))

        nominal = float(np.median(self.support_radii))
        low = float(np.quantile(self.support_radii, self.config.low_quantile))
        high = float(np.quantile(self.support_radii, self.config.high_quantile))

        # Legacy value is global, but confidence still exposes how far the query
        # lies from observed directions for downstream diagnostics.
        confidence = float(
            np.exp(
                -0.5
                * (nearest / max(self.config.max_support_angle_deg, EPS)) ** 2
            )
        )

        return RadiusEstimate(
            nominal=nominal,
            low=low,
            high=high,
            confidence=confidence,
            valid=True,
            strategy="global_median",
            source="captured_cameras",
            nearest_support_angle_deg=nearest,
            effective_neighbors=float(len(self.support_radii)),
            neighbor_camera_indices=self.support_indices.tolist(),
            neighbor_angles_deg=angles.tolist(),
            notes=[
                "Direction-independent median radius retained as legacy baseline."
            ],
        )

    def _query_nearest(self, direction: np.ndarray) -> RadiusEstimate:
        """Use radius of the nearest captured radial direction."""

        if len(self.support_radii) == 0:
            return self._invalid(
                strategy="nearest",
                source="captured_cameras",
                note=f"No {self.mode.value} support cameras.",
            )

        angles = _angular_distances_deg(direction, self.support_directions)
        nearest_local = int(np.argmin(angles))
        nearest_angle = float(angles[nearest_local])

        valid = nearest_angle <= self.config.max_support_angle_deg
        confidence = float(
            np.exp(
                -0.5
                * (nearest_angle / max(self.config.sigma_deg, EPS)) ** 2
            )
        )

        radius = float(self.support_radii[nearest_local])
        index = int(self.support_indices[nearest_local])

        return RadiusEstimate(
            nominal=radius,
            low=radius,
            high=radius,
            confidence=confidence if valid else 0.0,
            valid=valid,
            strategy="nearest",
            source="captured_cameras",
            nearest_support_angle_deg=nearest_angle,
            effective_neighbors=1.0,
            neighbor_camera_indices=[index],
            neighbor_angles_deg=[nearest_angle],
            notes=(
                []
                if valid
                else [
                    "Nearest captured direction exceeds max_support_angle_deg."
                ]
            ),
        )

    def _query_angular_knn(self, direction: np.ndarray) -> RadiusEstimate:
        """Angular KNN weighted-quantile camera-radius prior."""

        if len(self.support_radii) == 0:
            return self._invalid(
                strategy="angular_knn",
                source="captured_cameras",
                note=f"No {self.mode.value} support cameras.",
            )

        if self.config.k_neighbors <= 0:
            raise ValueError("k_neighbors must be >0")

        angles = _angular_distances_deg(direction, self.support_directions)
        order = np.argsort(angles)
        order = order[: min(self.config.k_neighbors, len(order))]

        selected_angles = angles[order]
        support_mask = selected_angles <= self.config.max_support_angle_deg
        order = order[support_mask]
        selected_angles = selected_angles[support_mask]

        if len(order) < self.config.min_neighbors:
            nearest = float(np.min(angles)) if len(angles) else None
            estimate = self._invalid(
                strategy="angular_knn",
                source="captured_cameras",
                note=(
                    f"Only {len(order)} support cameras within "
                    f"{self.config.max_support_angle_deg:.1f} deg; "
                    f"min_neighbors={self.config.min_neighbors}."
                ),
            )
            estimate.nearest_support_angle_deg = nearest
            return estimate

        selected_radii = self.support_radii[order]
        selected_confidences = self.support_confidences[order]
        selected_indices = self.support_indices[order]

        sigma = max(self.config.sigma_deg, EPS)
        angular_weights = np.exp(-0.5 * (selected_angles / sigma) ** 2)

        # If legacy_sign classification produced confidence=1 this becomes a pure
        # angular interpolation; robust mode confidence naturally down-weights
        # weakly aligned cameras.
        weights = angular_weights * np.maximum(selected_confidences, EPS)

        nominal = _weighted_quantile(selected_radii, weights, 0.50)
        low = _weighted_quantile(
            selected_radii,
            weights,
            self.config.low_quantile,
        )
        high = _weighted_quantile(
            selected_radii,
            weights,
            self.config.high_quantile,
        )

        nearest = float(selected_angles[0])
        effective_neighbors = _effective_sample_size(weights)

        angle_confidence = float(np.exp(-0.5 * (nearest / sigma) ** 2))
        count_confidence = float(
            1.0
            - np.exp(
                -effective_neighbors
                / max(self.config.support_count_saturation, EPS)
            )
        )
        confidence = float(np.clip(angle_confidence * count_confidence, 0.0, 1.0))

        return RadiusEstimate(
            nominal=nominal,
            low=low,
            high=high,
            confidence=confidence,
            valid=True,
            strategy="angular_knn",
            source="captured_cameras",
            nearest_support_angle_deg=nearest,
            effective_neighbors=effective_neighbors,
            neighbor_camera_indices=selected_indices.astype(int).tolist(),
            neighbor_angles_deg=selected_angles.astype(float).tolist(),
        )

    def _query_pointcloud_cone(self, direction: np.ndarray) -> RadiusEstimate:
        """Previous cone-median point-cloud depth heuristic, exposed as strategy."""

        if self.point_cloud_points is None or len(self.point_cloud_points) == 0:
            return self._invalid(
                strategy="pointcloud_cone",
                source="point_cloud",
                note="Point cloud is required for pointcloud_cone radius strategy.",
            )

        direction = _normalize(direction)
        relative = self.point_cloud_relative
        radii = self.point_cloud_radius
        projection = relative @ direction

        front_mask = projection > 0.0
        if int(np.sum(front_mask)) < self.config.cone_min_points:
            return self._invalid(
                strategy="pointcloud_cone",
                source="point_cloud",
                note="Too few point-cloud samples in front of scene center.",
            )

        # For unit direction d:
        #   perpendicular^2 = ||p||^2 - (p dot d)^2
        # This avoids allocating an [M,3] perpendicular-vector array per query.
        projection_front = projection[front_mask]
        radius_front = radii[front_mask]
        perpendicular_sq = np.maximum(
            radius_front ** 2 - projection_front ** 2,
            0.0,
        )

        cone_tangent = np.tan(np.radians(self.config.cone_half_angle_deg))
        cone_mask = perpendicular_sq < (projection_front * cone_tangent) ** 2

        count = int(np.sum(cone_mask))
        if count < self.config.cone_min_points:
            return self._invalid(
                strategy="pointcloud_cone",
                source="point_cloud",
                note=(
                    f"Only {count} points in {self.config.cone_half_angle_deg:.1f} deg cone; "
                    f"min={self.config.cone_min_points}."
                ),
            )

        distances = radius_front[cone_mask]
        nominal = float(np.median(distances))
        low = float(np.quantile(distances, self.config.cone_low_quantile))
        high = float(np.quantile(distances, self.config.cone_high_quantile))

        confidence = float(1.0 - np.exp(-count / 20.0))

        return RadiusEstimate(
            nominal=nominal,
            low=low,
            high=high,
            confidence=confidence,
            valid=True,
            strategy="pointcloud_cone",
            source="point_cloud",
            geometry_point_count=count,
            notes=[
                "Legacy 15-degree-style cone median; not an exact first-hit free-space boundary."
            ],
        )

    def _query_hybrid(self, direction: np.ndarray) -> RadiusEstimate:
        """Combine camera KNN prior with point-cloud cone geometry estimate."""

        # Force the camera component to use the improved KNN implementation even
        # though the top-level configured strategy is 'hybrid'.
        camera_estimate = self._query_angular_knn(direction)
        geometry_estimate = self._query_pointcloud_cone(direction)

        if not camera_estimate.valid and not geometry_estimate.valid:
            return self._invalid(
                strategy="hybrid",
                source="camera+point_cloud",
                note="Neither camera support nor point-cloud geometry is valid.",
            )

        if not camera_estimate.valid:
            geometry_estimate.strategy = "hybrid"
            geometry_estimate.source = "point_cloud_only_fallback"
            geometry_estimate.notes.append("Camera prior unavailable.")
            return geometry_estimate

        if not geometry_estimate.valid:
            camera_estimate.strategy = "hybrid"
            camera_estimate.source = "captured_cameras_only_fallback"
            camera_estimate.notes.append("Point-cloud geometry unavailable.")
            return camera_estimate

        geometry_limit = self.config.hybrid_geometry_alpha * geometry_estimate.nominal

        if self.config.hybrid_rule == "min":
            nominal = min(camera_estimate.nominal, geometry_limit)
            low = min(camera_estimate.low, geometry_limit)
            high = min(camera_estimate.high, geometry_limit)
        elif self.config.hybrid_rule == "clamp_high":
            nominal = min(camera_estimate.nominal, geometry_limit)
            low = min(camera_estimate.low, nominal)
            high = min(camera_estimate.high, geometry_limit)
        else:
            raise ValueError(
                f"Unknown hybrid_rule={self.config.hybrid_rule!r}; expected 'min' or 'clamp_high'."
            )

        confidence = float(
            np.clip(
                np.sqrt(camera_estimate.confidence * geometry_estimate.confidence),
                0.0,
                1.0,
            )
        )

        return RadiusEstimate(
            nominal=float(nominal),
            low=float(low),
            high=float(high),
            confidence=confidence,
            valid=True,
            strategy="hybrid",
            source="camera+point_cloud",
            nearest_support_angle_deg=camera_estimate.nearest_support_angle_deg,
            effective_neighbors=camera_estimate.effective_neighbors,
            neighbor_camera_indices=camera_estimate.neighbor_camera_indices,
            neighbor_angles_deg=camera_estimate.neighbor_angles_deg,
            geometry_point_count=geometry_estimate.geometry_point_count,
            notes=[
                "Hybrid applies point-cloud cone estimate as a conservative camera-radius bound.",
                "Because cone median can hit foreground clutter, keep camera-only strategy for ablation.",
            ],
        )
