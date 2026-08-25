#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reusable captured-reference selection for denoising targets.

Inputs
------
* original captured cameras (indices must match train_cameras.json)
* a reference candidate pool, normally --select_view_dir/selection.json
* final pano target cameras
* optional visibility model for representation-aware covisibility coverage

Output
------
``ReferenceSelectionResult.original_indices`` contains indices in the *original*
train_cameras.json sequence and is written as ``[indices]`` to traj_refs.json by
Stage 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from viewpoint_framework.cameras_util import Camera
from viewpoint_framework.stage3.selection import angular_distance_rad
from viewpoint_framework.stage3.types import ReferenceSelectionResult, SelectedViewSet
from viewpoint_framework.stage3.visibility import NullVisibilityModel, VisibilityModel


EPS = 1e-10


@dataclass
class ReferenceSelectionConfig:
    strategy: str = "legacy_global_fps"  # legacy_global_fps | target_coverage_greedy | artifixer_style_covisibility
    target_angle_sigma_deg: float = 25.0
    target_position_sigma_ratio: float = 0.35
    artifixer_diminishing_return: bool = True


def _pool(selected_views: SelectedViewSet, captured_cameras: Sequence[Camera]):
    if selected_views.original_indices:
        return list(selected_views.original_indices), list(selected_views.cameras)
    return [int(c.index) for c in captured_cameras], list(captured_cameras)


def legacy_global_fps(
    selected_views: SelectedViewSet,
    captured_cameras: Sequence[Camera],
    num_refs: int,
) -> ReferenceSelectionResult:
    """Match the previous panorama generator's position-FPS reference baseline."""
    indices, cameras = _pool(selected_views, captured_cameras)
    num_refs = max(0, min(int(num_refs), len(cameras)))
    if num_refs == 0:
        return ReferenceSelectionResult("legacy_global_fps", [], indices)
    if len(cameras) <= num_refs:
        chosen = sorted(indices)
        return ReferenceSelectionResult("legacy_global_fps", chosen, indices)

    positions = np.stack([c.position for c in cameras], axis=0)
    # Previous code forces original frame 0 if present; otherwise candidate closest
    # to frame 0's captured position.
    if 0 in indices:
        first = indices.index(0)
    else:
        frame0 = captured_cameras[0].position
        first = int(np.argmin(np.linalg.norm(positions - frame0[None, :], axis=1)))

    chosen_local = [first]
    min_dist = np.linalg.norm(positions - positions[first][None, :], axis=1)
    for _ in range(1, num_refs):
        nxt = int(np.argmax(min_dist))
        chosen_local.append(nxt)
        min_dist = np.minimum(
            min_dist,
            np.linalg.norm(positions - positions[nxt][None, :], axis=1),
        )
    chosen = sorted(int(indices[i]) for i in chosen_local)
    return ReferenceSelectionResult(
        strategy="legacy_global_fps",
        original_indices=chosen,
        candidate_pool_indices=indices,
    )


def _target_support_matrix(
    refs: Sequence[Camera],
    targets: Sequence[Camera],
    scene_scale: float,
    config: ReferenceSelectionConfig,
) -> np.ndarray:
    """Pose-support proxy; intentionally independent of spherical scene-center assumptions."""
    if not refs or not targets:
        return np.zeros((len(refs), len(targets)), dtype=np.float64)
    angle_sigma = np.radians(max(float(config.target_angle_sigma_deg), 1e-3))
    pos_sigma = max(float(config.target_position_sigma_ratio) * max(scene_scale, EPS), EPS)
    support = np.zeros((len(refs), len(targets)), dtype=np.float64)
    for i, ref in enumerate(refs):
        for j, target in enumerate(targets):
            angle = angular_distance_rad(ref.forward, target.forward)
            pos = float(np.linalg.norm(ref.position - target.position))
            support[i, j] = np.exp(-0.5 * (angle / angle_sigma) ** 2) * np.exp(
                -0.5 * (pos / pos_sigma) ** 2
            )
    return support


def target_coverage_greedy(
    selected_views: SelectedViewSet,
    captured_cameras: Sequence[Camera],
    target_cameras: Sequence[Camera],
    num_refs: int,
    scene_scale: float,
    config: ReferenceSelectionConfig,
) -> ReferenceSelectionResult:
    """Greedy facility-location support of final pano targets.

    Objective: maximize sum_t max_{r in R} support(r, t).  It is a robust V1
    target-conditioned reference selector and can later swap in render/content
    similarity without changing the greedy optimizer.
    """
    indices, refs = _pool(selected_views, captured_cameras)
    num_refs = max(0, min(int(num_refs), len(refs)))
    support = _target_support_matrix(refs, target_cameras, scene_scale, config)
    best_support = np.zeros(len(target_cameras), dtype=np.float64)
    remaining = set(range(len(refs)))
    chosen_local: List[int] = []
    trace = []

    for _ in range(num_refs):
        best = None
        best_gain = -float("inf")
        for i in sorted(remaining):
            new_support = np.maximum(best_support, support[i])
            gain = float(np.sum(new_support - best_support))
            if gain > best_gain + 1e-12:
                best = i
                best_gain = gain
        if best is None:
            break
        chosen_local.append(best)
        remaining.remove(best)
        best_support = np.maximum(best_support, support[best])
        trace.append({"original_index": int(indices[best]), "gain": best_gain})

    chosen = sorted(int(indices[i]) for i in chosen_local)
    return ReferenceSelectionResult(
        strategy="target_coverage_greedy",
        original_indices=chosen,
        candidate_pool_indices=indices,
        scores={
            "objective": float(np.sum(best_support)),
            "mean_target_support": float(np.mean(best_support)) if len(best_support) else 0.0,
            "trace": trace,
        },
    )


def artifixer_style_covisibility(
    selected_views: SelectedViewSet,
    captured_cameras: Sequence[Camera],
    num_refs: int,
    visibility_model: VisibilityModel,
    config: ReferenceSelectionConfig,
) -> ReferenceSelectionResult:
    """ArtiFixer-inspired sparse-view covisibility coverage experiment.

    This is intentionally named ``style`` rather than claiming exact reproduction.
    Public ArtiFixer preparation describes half-covisibility sparse camera splits and
    explicitly prepares 2/3/6/12-view reconstructions.  Here we implement the same
    high-level goal with the aligned representation already available in this
    project: greedily cover sampled surface elements visible from the captured pool.
    """
    indices, refs = _pool(selected_views, captured_cameras)
    num_refs = max(0, min(int(num_refs), len(refs)))
    if isinstance(visibility_model, NullVisibilityModel) or len(visibility_model.sample_points) == 0:
        return legacy_global_fps(selected_views, captured_cameras, num_refs)

    matrix = visibility_model.visibility_matrix(refs, key_prefix="ref_artifixer_style")
    counts = np.zeros(matrix.shape[1], dtype=np.int32)
    remaining = set(range(len(refs)))
    chosen_local: List[int] = []
    trace = []

    for _ in range(num_refs):
        best = None
        best_gain = -float("inf")
        for i in sorted(remaining):
            if config.artifixer_diminishing_return:
                gain = visibility_model.marginal_gain(matrix[i], counts)
            else:
                gain = float(np.sum(visibility_model.sample_weights[matrix[i] & (counts == 0)]))
            if gain > best_gain + 1e-12:
                best = i
                best_gain = gain
        if best is None:
            break
        chosen_local.append(best)
        remaining.remove(best)
        counts += matrix[best].astype(np.int32)
        trace.append({"original_index": int(indices[best]), "gain": float(best_gain)})

    chosen = sorted(int(indices[i]) for i in chosen_local)
    covered = counts > 0
    return ReferenceSelectionResult(
        strategy="artifixer_style_covisibility",
        original_indices=chosen,
        candidate_pool_indices=indices,
        scores={
            "surface_coverage_ratio": float(np.mean(covered)) if len(covered) else 0.0,
            "trace": trace,
            "note": "ArtiFixer-inspired covisibility coverage; not an exact code reproduction.",
        },
    )


def select_references(
    selected_views: SelectedViewSet,
    captured_cameras: Sequence[Camera],
    target_cameras: Sequence[Camera],
    *,
    num_refs: int,
    scene_scale: float,
    config: Optional[ReferenceSelectionConfig] = None,
    visibility_model: Optional[VisibilityModel] = None,
) -> ReferenceSelectionResult:
    config = config or ReferenceSelectionConfig()
    visibility_model = visibility_model or NullVisibilityModel()
    if config.strategy == "legacy_global_fps":
        return legacy_global_fps(selected_views, captured_cameras, num_refs)
    if config.strategy == "target_coverage_greedy":
        return target_coverage_greedy(
            selected_views,
            captured_cameras,
            target_cameras,
            num_refs,
            scene_scale,
            config,
        )
    if config.strategy == "artifixer_style_covisibility":
        return artifixer_style_covisibility(
            selected_views,
            captured_cameras,
            num_refs,
            visibility_model,
            config,
        )
    raise ValueError(f"Unknown reference selection strategy: {config.strategy}")


if __name__ == "__main__":
    print("stage3.reference_selection: import OK")
