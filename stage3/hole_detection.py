#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Geometric under-coverage detection and focused hole-view generation.

Important distinction: Stage 2 already densely samples the allowed angular grid,
so Stage 3 does *not* look for holes in that 2-D grid.  Instead, it looks for
3-D reconstruction regions that are poorly observed by the preselected captured
anchor views.

V1 uses sampled Gaussian surface elements.  Visibility counts over the selected
captured views identify under-observed Gaussians, which are spatially clustered.
A focused view reuses the *position* of a Stage-2 geometry-safe candidate, changes
only orientation/intrinsics, and therefore preserves Stage-2 collision safety.
Large clusters are recursively split when one camera cannot fit the region.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from viewpoint_framework.cameras_util import Camera
from viewpoint_framework.gs_renderer import GsplatRenderer
from viewpoint_framework.pose_generation import build_c2w_from_forward
from viewpoint_framework.stage3.types import (
    CandidateOrigin,
    HoleCluster,
    HoleViewRecord,
    SelectionCandidate,
)
from viewpoint_framework.stage3.visibility import VisibilityModel


EPS = 1e-8


@dataclass
class HoleDetectionConfig:
    strategy: str = "gaussian_undercoverage"  # gaussian_undercoverage | pointcloud_gaussian_gap | none
    max_coverage_count: int = 0
    voxel_size_ratio: float = 0.035
    min_cluster_samples: int = 35
    min_severity_fraction: float = 0.002
    max_holes: int = 6
    connectivity: int = 26

    # Focused hole-view generation.
    max_views_per_hole: int = 3
    max_split_depth: int = 2
    fit_margin: float = 0.82
    min_focal_scale: float = 0.55
    max_focal_scale: float = 1.60
    min_positive_fraction: float = 0.90
    fit_quantile: float = 0.98
    max_source_angle_deg: float = 70.0
    min_target_visible_fraction: float = 0.10

    # Optional geometry-gap detector: point cloud has surface support while the
    # aligned 3DGS has no nearby Gaussian support.  This is intentionally not
    # the default because feed-forward point clouds can contain floaters.
    gap_distance_ratio: float = 0.015
    gap_gaussian_scale_multiplier: float = 2.0
    gap_min_gaussian_opacity: float = 0.10
    gap_require_anchor_visibility: bool = True


def _neighbors(connectivity: int) -> list[tuple[int, int, int]]:
    result = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                if connectivity == 6 and abs(dx) + abs(dy) + abs(dz) != 1:
                    continue
                result.append((dx, dy, dz))
    return result


def _cluster_sample_indices(
    visibility_model: VisibilityModel,
    indices: np.ndarray,
    counts: np.ndarray,
    base_weights: np.ndarray,
    scene_scale: float,
    config: HoleDetectionConfig,
) -> List[HoleCluster]:
    """Cluster selected representation samples into 3-D hole regions."""
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        return []
    points = visibility_model.sample_points[indices]
    weights = np.asarray(base_weights, dtype=np.float64).reshape(-1)
    if len(weights) != len(indices):
        raise ValueError("base_weights must align with indices")

    voxel_size = max(float(scene_scale) * float(config.voxel_size_ratio), 1e-6)
    origin = np.min(points, axis=0)
    voxels = np.floor((points - origin[None, :]) / voxel_size).astype(np.int64)
    voxel_to_local: dict[tuple[int, int, int], list[int]] = {}
    for local_idx, voxel in enumerate(voxels):
        voxel_to_local.setdefault(tuple(int(x) for x in voxel), []).append(local_idx)

    occupied = set(voxel_to_local.keys())
    neighbor_offsets = _neighbors(int(config.connectivity))
    clusters_local: List[List[int]] = []
    while occupied:
        seed = occupied.pop()
        stack = [seed]
        component_voxels = [seed]
        while stack:
            current = stack.pop()
            for off in neighbor_offsets:
                nxt = (current[0] + off[0], current[1] + off[1], current[2] + off[2])
                if nxt in occupied:
                    occupied.remove(nxt)
                    stack.append(nxt)
                    component_voxels.append(nxt)
        local_ids: List[int] = []
        for voxel in component_voxels:
            local_ids.extend(voxel_to_local[voxel])
        clusters_local.append(local_ids)

    total_severity = float(np.sum(weights))
    clusters: List[HoleCluster] = []
    for local_ids in clusters_local:
        if len(local_ids) < int(config.min_cluster_samples):
            continue
        local_ids_arr = np.asarray(local_ids, dtype=np.int64)
        global_ids = indices[local_ids_arr]
        pts = visibility_model.sample_points[global_ids]
        w = np.maximum(weights[local_ids_arr], EPS)
        severity = float(np.sum(w))
        if total_severity > EPS and severity / total_severity < float(config.min_severity_fraction):
            continue
        centroid = np.average(pts, axis=0, weights=w)
        extent = np.max(pts, axis=0) - np.min(pts, axis=0)
        clusters.append(
            HoleCluster(
                hole_id=-1,
                sample_indices=global_ids,
                centroid=centroid,
                severity=severity,
                coverage_mean=float(np.mean(counts[global_ids])),
                point_count=len(global_ids),
                extent=extent,
            )
        )

    clusters.sort(key=lambda x: (-x.severity, -x.point_count))
    clusters = clusters[: max(0, int(config.max_holes))]
    for idx, cluster in enumerate(clusters):
        cluster.hole_id = idx
    return clusters


def detect_geometric_holes(
    visibility_model: VisibilityModel,
    anchor_cameras: Sequence[Camera],
    scene_scale: float,
    config: Optional[HoleDetectionConfig] = None,
) -> Tuple[List[HoleCluster], np.ndarray]:
    """Detect under-observed *existing* representation elements.

    This is the robust V1 default: a Gaussian can exist in the reconstruction yet
    be seen by too few of the preselected captured anchor views.
    """
    config = config or HoleDetectionConfig()
    if config.strategy == "none" or len(visibility_model.sample_points) == 0:
        return [], np.zeros(len(visibility_model.sample_points), dtype=np.int32)
    if config.strategy != "gaussian_undercoverage":
        raise ValueError(f"detect_geometric_holes expects gaussian_undercoverage, got: {config.strategy}")

    counts = visibility_model.coverage_counts(anchor_cameras, key_prefix="hole_anchor")
    indices = np.flatnonzero(counts <= int(config.max_coverage_count))
    if len(indices) == 0:
        return [], counts
    base_weights = visibility_model.sample_weights[indices] / (1.0 + counts[indices])
    return _cluster_sample_indices(
        visibility_model, indices, counts, base_weights, scene_scale, config
    ), counts


def detect_pointcloud_gaussian_gaps(
    pointcloud_model: VisibilityModel,
    anchor_cameras: Sequence[Camera],
    renderer: GsplatRenderer,
    scene_scale: float,
    config: Optional[HoleDetectionConfig] = None,
) -> Tuple[List[HoleCluster], np.ndarray]:
    """Detect point-cloud-supported geometry with weak/absent 3DGS support.

    The aligned point cloud acts as an independent geometry hint.  A sampled point
    is considered a gap candidate when it remains farther than a scale-aware
    threshold from opaque Gaussian support.  Optionally require the point to be
    visible from at least one preselected captured anchor, reducing sensitivity to
    isolated feed-forward floaters.
    """
    config = config or HoleDetectionConfig(strategy="pointcloud_gaussian_gap")
    if len(pointcloud_model.sample_points) == 0:
        return [], np.zeros(0, dtype=np.int32)
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError("pointcloud_gaussian_gap requires scipy.spatial.cKDTree") from exc

    counts = pointcloud_model.coverage_counts(anchor_cameras, key_prefix="gap_anchor")
    opacity = np.asarray(renderer.opacities_np, dtype=np.float64)
    gs_mask = opacity >= float(config.gap_min_gaussian_opacity)
    if not np.any(gs_mask):
        return [], counts
    means = np.asarray(renderer.means_np[gs_mask], dtype=np.float64)
    scales = np.asarray(renderer.max_scale_np[gs_mask], dtype=np.float64)
    tree = cKDTree(means)
    k = min(4, len(means))
    distances, nearest = tree.query(pointcloud_model.sample_points, k=k)
    distances = np.asarray(distances, dtype=np.float64)
    nearest = np.asarray(nearest, dtype=np.int64)
    if k == 1:
        distances = distances[:, None]
        nearest = nearest[:, None]
    effective = distances - float(config.gap_gaussian_scale_multiplier) * scales[nearest]
    support_distance = np.min(effective, axis=1)
    threshold = max(float(config.gap_distance_ratio) * float(scene_scale), EPS)
    gap_mask = support_distance > threshold
    if config.gap_require_anchor_visibility:
        gap_mask &= counts > 0
    indices = np.flatnonzero(gap_mask)
    if len(indices) == 0:
        return [], counts

    # Large positive support distance means stronger evidence that point-cloud
    # geometry is missing from the Gaussian representation.  Anchor visibility
    # increases confidence rather than decreasing the score.
    evidence = np.clip(support_distance[indices] / threshold, 1.0, 4.0)
    anchor_conf = np.clip(counts[indices].astype(np.float64), 1.0, 3.0) / 3.0
    base_weights = pointcloud_model.sample_weights[indices] * evidence * anchor_conf
    return _cluster_sample_indices(
        pointcloud_model, indices, counts, base_weights, scene_scale, config
    ), counts


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a /= max(float(np.linalg.norm(a)), EPS)
    b /= max(float(np.linalg.norm(b)), EPS)
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0))))


def _choose_source_candidate(
    centroid: np.ndarray,
    scene_center: np.ndarray,
    candidates: Sequence[SelectionCandidate],
    max_angle_deg: float,
) -> Optional[SelectionCandidate]:
    direction = np.asarray(centroid, dtype=np.float64) - np.asarray(scene_center, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= EPS:
        return None
    direction /= norm

    best = None
    best_score = float("inf")
    for candidate in candidates:
        angle = _angle_deg(candidate.observation_direction, direction)
        # Small distance preference only breaks angular ties; angular sector is the
        # main signal for both outside-in and inside-out Stage-2 grids.
        distance = float(np.linalg.norm(candidate.camera.position - centroid))
        score = angle + 0.01 * distance
        if score < best_score:
            best_score = score
            best = candidate
    if best is None or _angle_deg(best.observation_direction, direction) > float(max_angle_deg):
        return None
    return best


def _camera_from_target(
    source: Camera,
    target: np.ndarray,
    world_up: np.ndarray,
    fallback_axis: np.ndarray,
    focal_scale: float,
    index: int,
) -> Camera:
    forward = np.asarray(target, dtype=np.float64) - source.position
    forward /= max(float(np.linalg.norm(forward)), EPS)
    c2w = build_c2w_from_forward(
        source.position,
        forward,
        world_up=world_up,
        fallback_axis=fallback_axis,
    )
    w2c = np.linalg.inv(c2w)
    return Camera(
        index=index,
        fx=float(source.fx) * float(focal_scale),
        fy=float(source.fy) * float(focal_scale),
        cx=float(source.cx),
        cy=float(source.cy),
        width=int(source.width),
        height=int(source.height),
        w2c=w2c,
        c2w=c2w,
    )


def _fit_focal_scale(
    camera: Camera,
    points: np.ndarray,
    config: HoleDetectionConfig,
) -> tuple[float, bool, float]:
    R = camera.w2c[:3, :3]
    t = camera.w2c[:3, 3]
    pc = points @ R.T + t[None, :]
    positive = pc[:, 2] > EPS
    positive_fraction = float(np.mean(positive)) if len(points) else 0.0
    if positive_fraction < float(config.min_positive_fraction):
        return float(config.min_focal_scale), False, positive_fraction
    pc = pc[positive]
    rx = np.abs(pc[:, 0] / pc[:, 2])
    ry = np.abs(pc[:, 1] / pc[:, 2])
    q = float(np.clip(config.fit_quantile, 0.5, 1.0))
    max_rx = max(float(np.quantile(rx, q)), EPS)
    max_ry = max(float(np.quantile(ry, q)), EPS)
    fmax_x = 0.5 * float(camera.width) * float(config.fit_margin) / max_rx
    fmax_y = 0.5 * float(camera.height) * float(config.fit_margin) / max_ry
    scale_fit = min(fmax_x / float(camera.fx), fmax_y / float(camera.fy))
    fit_ok = scale_fit >= float(config.min_focal_scale)
    scale = float(np.clip(scale_fit, config.min_focal_scale, config.max_focal_scale))
    return scale, bool(fit_ok), positive_fraction


def _split_points_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0)
    centered = points - center[None, :]
    if len(points) < 2:
        return points, np.empty((0, 3), dtype=points.dtype)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    values = centered @ axis
    median = float(np.median(values))
    left = points[values <= median]
    right = points[values > median]
    if len(left) == 0 or len(right) == 0:
        half = len(points) // 2
        return points[:half], points[half:]
    return left, right


def generate_focused_hole_views(
    holes: Sequence[HoleCluster],
    visibility_model: VisibilityModel,
    grid_candidates: Sequence[SelectionCandidate],
    scene_center: np.ndarray,
    world_up: np.ndarray,
    fallback_axis: np.ndarray,
    *,
    next_candidate_id: int,
    config: Optional[HoleDetectionConfig] = None,
) -> tuple[List[SelectionCandidate], List[HoleViewRecord]]:
    config = config or HoleDetectionConfig()
    generated: List[SelectionCandidate] = []
    records: List[HoleViewRecord] = []
    candidate_id = int(next_candidate_id)

    for hole in holes:
        root_points = visibility_model.sample_points[hole.sample_indices]
        queue: List[tuple[np.ndarray, int]] = [(root_points, 0)]
        hole_views = 0
        while queue and hole_views < int(config.max_views_per_hole):
            points, split_depth = queue.pop(0)
            if len(points) < int(config.min_cluster_samples):
                continue
            centroid = np.mean(points, axis=0)
            source_candidate = _choose_source_candidate(
                centroid,
                scene_center,
                grid_candidates,
                max_angle_deg=config.max_source_angle_deg,
            )
            if source_candidate is None:
                continue

            probe_camera = _camera_from_target(
                source_candidate.camera,
                centroid,
                world_up,
                fallback_axis,
                focal_scale=1.0,
                index=candidate_id,
            )
            scale, fit_ok, positive_fraction = _fit_focal_scale(probe_camera, points, config)

            if (
                not fit_ok
                and split_depth < int(config.max_split_depth)
                and len(points) >= 2 * int(config.min_cluster_samples)
            ):
                left, right = _split_points_pca(points)
                queue.insert(0, (right, split_depth + 1))
                queue.insert(0, (left, split_depth + 1))
                continue

            if not fit_ok:
                # The region is not a reasonable single-view hole target under the
                # configured minimum FOV; do not force-select an invalid close-up.
                continue

            camera = _camera_from_target(
                source_candidate.camera,
                centroid,
                world_up,
                fallback_axis,
                focal_scale=scale,
                index=candidate_id,
            )
            observation = camera.forward.copy()
            generated.append(
                SelectionCandidate(
                    candidate_id=candidate_id,
                    camera=camera,
                    origin=CandidateOrigin.HOLE,
                    grid_id=None,
                    row=source_candidate.row,
                    col=source_candidate.col,
                    azimuth_deg=source_candidate.azimuth_deg,
                    elevation_deg=source_candidate.elevation_deg,
                    observation_direction=observation,
                    source_candidate_id=source_candidate.candidate_id,
                    hole_id=hole.hole_id,
                    forced_select=True,
                    signed_radius=source_candidate.signed_radius,
                    crossed_center=source_candidate.crossed_center,
                    safety_clearance=source_candidate.safety_clearance,
                    radius_confidence=source_candidate.radius_confidence,
                    depth_confidence=source_candidate.depth_confidence,
                    notes=[
                        "focused geometric-hole view; Stage-2 safe position reused",
                        f"positive_fraction={positive_fraction:.3f}",
                    ],
                )
            )
            records.append(
                HoleViewRecord(
                    hole_id=hole.hole_id,
                    candidate_id=candidate_id,
                    source_grid_candidate_id=source_candidate.candidate_id,
                    split_depth=split_depth,
                    focal_scale=scale,
                    fit_ok=True,
                    severity=float(hole.severity),
                )
            )
            candidate_id += 1
            hole_views += 1

    return generated, records


if __name__ == "__main__":
    # Connectivity test independent of renderer/camera code.
    pts = np.array([[0, 0, 0], [0.01, 0, 0], [1, 1, 1], [1.01, 1, 1]], dtype=float)
    left, right = _split_points_pca(pts)
    assert len(left) + len(right) == len(pts)
    print("stage3.hole_detection self-test passed")
