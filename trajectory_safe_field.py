"""Ordered trajectory support for V3 horizontal camera placement.

Intervals describe a position prior, not proven free space. Every proposed
camera still requires its own point-cloud clearance check.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from viewpoint_framework.cameras_util import Camera
from viewpoint_framework.scene_types import SphericalFrame

EPS = 1e-10


@dataclass
class TrajectorySafeFieldConfig:
    max_step_multiplier: float = 5.0
    resample_step_ratio: float = 0.5
    azimuth_support_deg: float = 10.0
    max_angular_gap_deg: float = 30.0
    tube_radius_ratio: float = 0.04
    unsupported_behavior: str = "nearest_limited"  # nearest_limited | reject
    rho_strategy: str = "v3_1"  # v3_1 | segment_ray_min
    fallback_strategy: str = "v3_1"
    fallback_confidence_threshold: float = 0.2

    def validate(self) -> None:
        for name in ("max_step_multiplier", "resample_step_ratio",
                     "azimuth_support_deg", "max_angular_gap_deg", "tube_radius_ratio"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not self.azimuth_support_deg <= self.max_angular_gap_deg < 180:
            raise ValueError("Require support angle <= max angular gap < 180 degrees.")
        if self.tube_radius_ratio >= 1:
            raise ValueError("tube_radius_ratio must be < 1.")
        if self.unsupported_behavior not in ("nearest_limited", "reject"):
            raise ValueError("Unsupported trajectory fallback policy.")
        if self.rho_strategy not in ("v3_1", "segment_ray_min"):
            raise ValueError("Unsupported trajectory rho strategy.")
        if self.fallback_strategy != "v3_1":
            raise ValueError("Only the V3.1 trajectory fallback is supported.")
        if not 0.0 <= self.fallback_confidence_threshold <= 1.0:
            raise ValueError("fallback_confidence_threshold must be in [0,1].")


@dataclass
class HeightGuardConfig:
    enabled: bool = False  # V3.0 experiment stub; positional elevation stays active.
    margin_ratio: float = 0.05

    def allows(self, height: float, interval: "TrajectorySupportInterval") -> bool:
        if not np.isfinite(self.margin_ratio) or self.margin_ratio < 0:
            raise ValueError("Height margin ratio must be finite and nonnegative.")
        margin = self.margin_ratio * interval.rho_preferred
        return bool(np.isfinite(height) and
                    interval.height_min - margin <= height <= interval.height_max + margin)


@dataclass
class TrajectorySupportInterval:
    branch_id: int
    rho_min: float
    rho_max: float
    rho_preferred: float
    height_min: float
    height_max: float
    height_preferred: float
    confidence: float
    support_camera_indices: List[int] = field(default_factory=list)


@dataclass
class TrajectorySafeEstimate:
    azimuth_deg: float
    intervals: List[TrajectorySupportInterval]
    selected_interval: Optional[TrajectorySupportInterval]
    confidence: float
    source: str
    nearest_support_angle_deg: Optional[float] = None


@dataclass
class AzimuthTrajectorySupport:
    azimuth_deg: float
    rho: Optional[float]
    rho_source: str
    direct_intersection_count: int
    selected_segment_start_index: Optional[int]
    selected_segment_end_index: Optional[int]
    selected_segment_t: Optional[float]
    trajectory_cross_height: Optional[float]
    trajectory_height_min: Optional[float]
    trajectory_height_max: Optional[float]
    fallback_confidence: Optional[float]
    fallback_low_confidence: bool = False
    branch_id: Optional[int] = None
    nearest_support_angle_deg: Optional[float] = None


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    i = np.searchsorted(cumulative, cumulative[-1] * 0.5, side="left")
    return float(values[order[i]])


class TrajectorySafeField:
    def __init__(
        self, cameras: Sequence[Camera], center: np.ndarray, frame: SphericalFrame,
        config: Optional[TrajectorySafeFieldConfig] = None,
    ) -> None:
        self.config = config or TrajectorySafeFieldConfig()
        self.config.validate()
        center = np.asarray(center, dtype=np.float64).reshape(3)
        basis = np.column_stack((frame.x_axis, frame.y_axis, frame.z_axis))
        if (not np.isfinite(center).all() or not np.isfinite(basis).all()
                or not np.allclose(basis.T @ basis, np.eye(3), atol=1e-6)):
            raise ValueError("Trajectory field requires a finite center and orthonormal frame.")
        cameras = list(cameras)
        positions = np.asarray([c.position for c in cameras], dtype=np.float64).reshape(-1, 3)
        finite = np.asarray([np.isfinite(c.c2w).all() for c in cameras], dtype=bool)
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        good_edges = finite[:-1] & finite[1:] & np.isfinite(steps)
        nonzero = steps[good_edges & (steps > EPS)]
        self.typical_step = float(np.median(nonzero)) if len(nonzero) else 0.0
        jumps = good_edges & (steps > self.config.max_step_multiplier * self.typical_step)
        linked = good_edges & ~jumps
        self.jump_rejected_count = int(jumps.sum())
        self.center = center
        self.basis = basis
        self.positions = positions
        self.finite_camera_mask = finite
        self.linked_edges = linked
        self.camera_indices = [int(c.index) for c in cameras]

        # Invalid poses break the sequence; filtering them out must not bridge gaps.
        branch_ids = np.full(len(cameras), -1, dtype=int)
        branch = -1
        for i in range(len(cameras)):
            if finite[i]:
                if i == 0 or not linked[i - 1]:
                    branch += 1
                branch_ids[i] = branch
        self.branch_count = branch + 1
        self.invalid_pose_count = int((~finite).sum())
        self.degenerate_horizontal_count = 0
        samples, sample_branches, support = [], [], []

        def append(position, branch_id, indices):
            local = (position - center) @ basis
            rho = float(np.hypot(local[0], local[2]))
            if rho <= EPS:
                self.degenerate_horizontal_count += 1
                return
            az = float(np.degrees(np.arctan2(local[0], local[2])))
            samples.append((az, rho, float(local[1])))
            sample_branches.append(branch_id)
            support.append(tuple(indices))

        for i, camera in enumerate(cameras):
            if not finite[i]:
                continue
            append(positions[i], int(branch_ids[i]), [int(camera.index)])
            if i + 1 < len(cameras) and linked[i] and steps[i] > EPS:
                step = self.config.resample_step_ratio * self.typical_step
                count = int(np.ceil(steps[i] / max(step, EPS)))
                if count > 10000:
                    raise ValueError("Trajectory resampling requested >10000 divisions per edge.")
                for j in range(1, count):
                    append(positions[i] + j / count * (positions[i + 1] - positions[i]),
                           int(branch_ids[i]), [int(camera.index), int(cameras[i + 1].index)])
        self.samples = np.asarray(samples, dtype=np.float64).reshape(-1, 3)
        self.branches = np.asarray(sample_branches, dtype=int)
        self.support_indices = support
        local_original = (positions[finite] - center) @ basis
        original_rhos = np.hypot(local_original[:, 0], local_original[:, 2])
        original_rhos = original_rhos[original_rhos > EPS]
        self.median_captured_rho = float(np.median(original_rhos)) if len(original_rhos) else 0.0
        self.captured_heights = local_original[:, 1].copy()

    def query(self, azimuth_deg: float) -> TrajectorySafeEstimate:
        if not np.isfinite(azimuth_deg):
            raise ValueError("Azimuth must be finite.")
        az = float((azimuth_deg + 180) % 360 - 180)
        if not len(self.samples):
            return TrajectorySafeEstimate(az, [], None, 0.0, "NO_TRAJECTORY_SUPPORT")
        angles = np.abs((self.samples[:, 0] - az + 180) % 360 - 180)
        nearest = float(angles.min())
        selected = np.flatnonzero(angles <= self.config.azimuth_support_deg)
        source = "trajectory_local"
        if not len(selected):
            if (self.config.unsupported_behavior == "reject"
                    or nearest > self.config.max_angular_gap_deg):
                return TrajectorySafeEstimate(az, [], None, 0.0, "TRAJECTORY_GAP_TOO_LARGE", nearest)
            selected = np.flatnonzero(np.isclose(angles, nearest, atol=1e-8, rtol=0))
            source = "trajectory_nearest_limited"

        intervals = []
        for branch in sorted(set(self.branches[selected])):
            ids = selected[self.branches[selected] == branch]
            ids = ids[np.argsort(self.samples[ids, 1], kind="stable")]
            groups = []
            current, high = [], -np.inf
            for idx in ids:
                rho = self.samples[idx, 1]
                lo, hi = rho * (1 - self.config.tube_radius_ratio), rho * (1 + self.config.tube_radius_ratio)
                if current and lo > high:
                    groups.append(current)
                    current, high = [], -np.inf
                current.append(idx)
                high = max(high, hi)
            if current:
                groups.append(current)
            for group in groups:
                rows = self.samples[group]
                weights = np.exp(-0.5 * (angles[group] / self.config.azimuth_support_deg) ** 2)
                camera_ids = sorted({j for i in group for j in self.support_indices[i]})
                confidence = float(weights.max() * min(1.0, len(camera_ids) / 3.0))
                intervals.append(TrajectorySupportInterval(
                    branch_id=int(branch),
                    rho_min=float(rows[:, 1].min() * (1 - self.config.tube_radius_ratio)),
                    rho_max=float(rows[:, 1].max() * (1 + self.config.tube_radius_ratio)),
                    rho_preferred=_weighted_median(rows[:, 1], weights),
                    height_min=float(rows[:, 2].min()), height_max=float(rows[:, 2].max()),
                    height_preferred=_weighted_median(rows[:, 2], weights),
                    confidence=confidence, support_camera_indices=camera_ids,
                ))
        # Deterministic ties: original branch order, then smaller supported rho.
        best = max(intervals, key=lambda x: (x.confidence, -x.branch_id, -x.rho_preferred))
        return TrajectorySafeEstimate(az, intervals, best, best.confidence, source, nearest)

    def query_segment_ray_min(self, azimuth_deg: float) -> AzimuthTrajectorySupport:
        """Resolve V3.2 rho from continuous segment/ray intersections."""
        if not np.isfinite(azimuth_deg):
            raise ValueError("Azimuth must be finite.")
        az = float((azimuth_deg + 180.0) % 360.0 - 180.0)
        radians = np.radians(az)
        ray = np.array([np.sin(radians), np.cos(radians)], dtype=np.float64)
        local = (self.positions - self.center) @ self.basis
        hits = []
        for i, linked in enumerate(self.linked_edges):
            if not linked:
                continue
            q0 = local[i, [0, 2]]
            q1 = local[i + 1, [0, 2]]
            delta = q1 - q0
            matrix = np.column_stack((delta, -ray))
            determinant = float(np.linalg.det(matrix))
            scale = max(float(np.linalg.norm(delta)), 1.0)
            if abs(determinant) <= EPS * scale:
                continue
            t, rho = np.linalg.solve(matrix, -q0)
            if -1e-9 <= t <= 1.0 + 1e-9 and rho > EPS:
                t = float(np.clip(t, 0.0, 1.0))
                h0, h1 = float(local[i, 1]), float(local[i + 1, 1])
                height = (1.0 - t) * h0 + t * h1
                hits.append((float(rho), i, t, height, h0, h1))

        if hits:
            rho, i, t, height, h0, h1 = min(hits, key=lambda x: (x[0], x[1], x[2]))
            return AzimuthTrajectorySupport(
                azimuth_deg=az, rho=rho, rho_source="segment_ray_min",
                direct_intersection_count=len(hits),
                selected_segment_start_index=i,
                selected_segment_end_index=i + 1,
                selected_segment_t=t,
                trajectory_cross_height=height,
                trajectory_height_min=min(h0, h1, height),
                trajectory_height_max=max(h0, h1, height),
                fallback_confidence=None,
            )

        estimate = self.query(az)
        intervals = list(estimate.intervals)
        if not intervals:
            return AzimuthTrajectorySupport(
                azimuth_deg=az, rho=None, rho_source=estimate.source,
                direct_intersection_count=0,
                selected_segment_start_index=None, selected_segment_end_index=None,
                selected_segment_t=None, trajectory_cross_height=None,
                trajectory_height_min=None, trajectory_height_max=None,
                fallback_confidence=estimate.confidence,
                fallback_low_confidence=True,
                nearest_support_angle_deg=estimate.nearest_support_angle_deg,
            )
        threshold = float(self.config.fallback_confidence_threshold)
        qualified = [item for item in intervals if item.confidence >= threshold]
        low_confidence = not bool(qualified)
        if qualified:
            selected = min(qualified, key=lambda x: (x.rho_preferred, x.branch_id))
        else:
            selected = min(intervals, key=lambda x: (-x.confidence, x.rho_preferred, x.branch_id))
        return AzimuthTrajectorySupport(
            azimuth_deg=az, rho=float(selected.rho_preferred),
            rho_source=("trajectory_fallback_low_confidence" if low_confidence
                        else "trajectory_fallback_v3_1"),
            direct_intersection_count=0,
            selected_segment_start_index=None, selected_segment_end_index=None,
            selected_segment_t=None,
            trajectory_cross_height=float(selected.height_preferred),
            trajectory_height_min=float(selected.height_min),
            trajectory_height_max=float(selected.height_max),
            fallback_confidence=float(selected.confidence),
            fallback_low_confidence=low_confidence,
            branch_id=int(selected.branch_id),
            nearest_support_angle_deg=estimate.nearest_support_angle_deg,
        )
