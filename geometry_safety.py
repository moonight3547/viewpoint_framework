#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fast point-cloud safety checks for candidate camera placement.

The pose generator uses rendered 3DGS depth to estimate *directional free space*.
This module deliberately solves a different problem: whether the camera center
or its short motion path gets too close to observed geometry.

The implementation is intentionally conservative and small:
- one Open3D KD-tree built once;
- median distance of the k nearest points (less sensitive to one isolated point);
- segment safety approximated by regular samples along the motion path.

The API is kept independent from the pose-generation policy so a future
Gaussian-scale / TSDF / mesh backend can replace it without changing the
placement code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except ImportError:  # pragma: no cover - handled at runtime when used
    o3d = None


EPS = 1e-10


@dataclass
class GeometrySafetyConfig:
    """Point-cloud safety configuration.

    ``clearance_ratio`` is multiplied by the point-cloud AABB diagonal.
    ``clearance_abs`` is an absolute lower bound in scene world units.
    """

    strategy: str = "pointcloud_knn"  # none | pointcloud_knn
    clearance_ratio: float = 0.015
    clearance_abs: float = 0.03
    knn_k: int = 3
    path_step_ratio: float = 0.50  # sample step = clearance * this ratio
    max_path_samples: int = 96
    path_backoff_ratio: float = 0.75


@dataclass
class PathSafetyResult:
    safe_fraction: float
    fully_safe: bool
    min_clearance: float
    num_samples: int


class PointCloudSafety:
    """KD-tree based position and short-path safety evaluator."""

    def __init__(
        self,
        points: np.ndarray,
        config: Optional[GeometrySafetyConfig] = None,
    ) -> None:
        self.config = config or GeometrySafetyConfig()
        self.points = np.asarray(points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must have shape [N,3], got {self.points.shape}")
        if len(self.points) == 0:
            raise ValueError("PointCloudSafety requires at least one point.")

        if self.config.strategy not in ("none", "pointcloud_knn"):
            raise ValueError(f"Unknown geometry safety strategy: {self.config.strategy}")

        self.min_bound = self.points.min(axis=0)
        self.max_bound = self.points.max(axis=0)
        self.scene_diagonal = float(np.linalg.norm(self.max_bound - self.min_bound))
        if self.scene_diagonal <= EPS:
            self.scene_diagonal = 1.0

        self.clearance = max(
            float(self.config.clearance_abs),
            float(self.config.clearance_ratio) * self.scene_diagonal,
        )

        self._tree = None
        if self.config.strategy == "pointcloud_knn":
            if o3d is None:
                raise ImportError(
                    "open3d is required for pointcloud_knn geometry safety."
                )
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            self._tree = o3d.geometry.KDTreeFlann(pcd)

    def point_clearance(self, position: np.ndarray) -> float:
        """Robust distance from a camera position to nearby point geometry."""
        if self.config.strategy == "none":
            return float("inf")

        position = np.asarray(position, dtype=np.float64).reshape(3)
        k = max(1, min(int(self.config.knn_k), len(self.points)))
        count, _, dist2 = self._tree.search_knn_vector_3d(position, k)
        if count <= 0:
            return float("inf")
        distances = np.sqrt(np.asarray(dist2[:count], dtype=np.float64))
        return float(np.median(distances))

    def is_position_safe(self, position: np.ndarray) -> Tuple[bool, float]:
        distance = self.point_clearance(position)
        return bool(distance >= self.clearance), distance

    def safe_path_fraction(
        self,
        start: np.ndarray,
        end: np.ndarray,
    ) -> PathSafetyResult:
        """Return the farthest contiguous safe fraction on ``start -> end``.

        The initial point is assumed to be a useful prior but is still checked.
        When a collision is detected, a small backoff is applied so the adjusted
        camera does not sit exactly at the estimated boundary.
        """
        start = np.asarray(start, dtype=np.float64).reshape(3)
        end = np.asarray(end, dtype=np.float64).reshape(3)

        if self.config.strategy == "none":
            return PathSafetyResult(1.0, True, float("inf"), 2)

        length = float(np.linalg.norm(end - start))
        if length <= EPS:
            safe, distance = self.is_position_safe(start)
            return PathSafetyResult(
                safe_fraction=1.0 if safe else 0.0,
                fully_safe=safe,
                min_clearance=distance,
                num_samples=1,
            )

        step = max(self.clearance * float(self.config.path_step_ratio), 1e-6)
        n = int(np.ceil(length / step)) + 1
        n = max(2, min(n, int(self.config.max_path_samples)))
        ts = np.linspace(0.0, 1.0, n, dtype=np.float64)

        min_clearance = float("inf")
        last_safe_t = 0.0

        for sample_index, t in enumerate(ts):
            position = start + float(t) * (end - start)
            distance = self.point_clearance(position)
            min_clearance = min(min_clearance, distance)

            if distance < self.clearance:
                if sample_index == 0:
                    return PathSafetyResult(0.0, False, min_clearance, n)

                previous_t = float(ts[sample_index - 1])
                current_t = float(t)
                # Stay slightly before the last safe sample.  The backoff is
                # expressed in fractions of one sampling interval, which keeps
                # the behavior scale-independent.
                backoff = float(np.clip(self.config.path_backoff_ratio, 0.0, 1.0))
                safe_t = max(0.0, previous_t - backoff * (current_t - previous_t))
                return PathSafetyResult(
                    safe_fraction=float(np.clip(safe_t, 0.0, 1.0)),
                    fully_safe=False,
                    min_clearance=min_clearance,
                    num_samples=n,
                )

            last_safe_t = float(t)

        return PathSafetyResult(1.0, True, min_clearance, n)
