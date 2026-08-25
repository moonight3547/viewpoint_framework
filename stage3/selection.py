#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Final target-view selection: FPS, IG-aware greedy coverage and ordering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from viewpoint_framework.cameras_util import Camera
from viewpoint_framework.scene_types import CameraMode
from viewpoint_framework.stage3.types import CandidateOrigin, SelectionCandidate
from viewpoint_framework.stage3.visibility import NullVisibilityModel, VisibilityModel


EPS = 1e-10


@dataclass
class SelectionConfig:
    strategy: str = "angular_fps"  # legacy_position_fps | angular_fps | utility_angular_fps | greedy_coverage
    reference: str = "generated_only"  # generated_only | captured_seeded
    information_gain_strategy: str = "none"  # none | gaussian_visibility | pointcloud_visibility
    coverage_include_captured: bool = True
    ig_tie_ratio: float = 0.97

    # Denoising quality V1: do not guess quality except remove near-duplicates.
    quality_strategy: str = "none"  # none | render_quality_band (interface reserved)
    near_duplicate_filter: bool = True
    near_duplicate_forward_deg: float = 2.0
    near_duplicate_position_ratio: float = 0.01

    # Output ordering remains separate from selection.
    ordering_strategy: str = "grid_order"  # grid_order | nearest_neighbor | selection_order
    position_distance_weight: float = 0.20


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v / max(float(np.linalg.norm(v)), EPS)


def angular_distance_rad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arccos(np.clip(float(np.dot(normalize(a), normalize(b))), -1.0, 1.0)))


def angular_distance_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(angular_distance_rad(a, b)))


def captured_coverage_direction(
    camera: Camera,
    scene_center: np.ndarray,
    mode: CameraMode,
) -> np.ndarray:
    """Experimental descriptor for captured-seeded angular FPS.

    Outside-in uses camera location sector relative to the center; inside-out uses
    actual forward direction.  Captured cameras need not lie on a perfect sphere,
    so this is deliberately exposed only as an ablation option.  Gaussian coverage
    is the more robust representation-aware alternative.
    """
    if mode == CameraMode.OUTSIDE_IN:
        radial = camera.position - np.asarray(scene_center, dtype=np.float64)
        if float(np.linalg.norm(radial)) > EPS:
            return normalize(radial)
    return normalize(camera.forward)


def filter_near_captured_duplicates(
    candidates: Sequence[SelectionCandidate],
    captured_cameras: Sequence[Camera],
    scene_scale: float,
    config: SelectionConfig,
) -> tuple[List[SelectionCandidate], List[int]]:
    if not config.near_duplicate_filter or len(captured_cameras) == 0:
        return list(candidates), []

    position_threshold = max(
        EPS,
        float(config.near_duplicate_position_ratio) * max(float(scene_scale), EPS),
    )
    kept: List[SelectionCandidate] = []
    removed: List[int] = []
    captured_positions = np.stack([c.position for c in captured_cameras], axis=0)
    captured_forwards = np.stack([normalize(c.forward) for c in captured_cameras], axis=0)

    for candidate in candidates:
        # Hole views are intentionally generated from under-covered geometry and are
        # forced targets; never remove them with a cheap pose proxy.
        if candidate.forced_select:
            kept.append(candidate)
            continue
        distances = np.linalg.norm(captured_positions - candidate.camera.position[None, :], axis=1)
        close = np.flatnonzero(distances <= position_threshold)
        duplicate = False
        if len(close):
            f = normalize(candidate.camera.forward)
            dots = np.clip(captured_forwards[close] @ f, -1.0, 1.0)
            angles = np.degrees(np.arccos(dots))
            duplicate = bool(np.any(angles <= float(config.near_duplicate_forward_deg)))
        if duplicate:
            removed.append(candidate.candidate_id)
        else:
            kept.append(candidate)
    return kept, removed


def _initial_seed_directions(
    forced: Sequence[SelectionCandidate],
    captured_cameras: Sequence[Camera],
    scene_center: np.ndarray,
    mode: CameraMode,
    reference: str,
) -> List[np.ndarray]:
    seeds = [normalize(c.observation_direction) for c in forced]
    if reference == "captured_seeded":
        seeds.extend(
            captured_coverage_direction(c, scene_center, mode)
            for c in captured_cameras
        )
    elif reference != "generated_only":
        raise ValueError(f"Unknown selection reference: {reference}")
    return seeds


def _min_angular_distance(direction: np.ndarray, seeds: Sequence[np.ndarray]) -> float:
    if not seeds:
        return np.pi
    d = normalize(direction)
    dots = [float(np.dot(d, normalize(seed))) for seed in seeds]
    return float(np.arccos(np.clip(max(dots), -1.0, 1.0)))


def _min_position_distance(position: np.ndarray, seed_positions: Sequence[np.ndarray]) -> float:
    if not seed_positions:
        return float("inf")
    return min(float(np.linalg.norm(position - p)) for p in seed_positions)


def _candidate_visibility_rows(
    candidates: Sequence[SelectionCandidate],
    visibility_model: VisibilityModel,
    cache_prefix: str,
) -> Dict[int, np.ndarray]:
    return {
        c.candidate_id: visibility_model.visibility(
            c.camera,
            cache_key=f"{cache_prefix}:{c.candidate_id}",
        )
        for c in candidates
    }


def select_views(
    candidates: Sequence[SelectionCandidate],
    *,
    num_panos: int,
    captured_seed_cameras: Sequence[Camera],
    scene_center: np.ndarray,
    scene_scale: float,
    mode: CameraMode,
    config: Optional[SelectionConfig] = None,
    visibility_model: Optional[VisibilityModel] = None,
) -> tuple[List[SelectionCandidate], Dict[str, object]]:
    config = config or SelectionConfig()
    visibility_model = visibility_model or NullVisibilityModel()
    num_panos = max(0, int(num_panos))

    filtered, removed_near = filter_near_captured_duplicates(
        candidates,
        captured_seed_cameras,
        scene_scale,
        config,
    )
    forced = [c for c in filtered if c.forced_select]
    normal = [c for c in filtered if not c.forced_select]

    # Reasonable hole views are reserved before generic FPS/coverage selection.
    forced = sorted(forced, key=lambda c: (c.hole_id if c.hole_id is not None else 10**9, c.candidate_id))
    if len(forced) > num_panos:
        forced = forced[:num_panos]
    selected: List[SelectionCandidate] = list(forced)
    remaining_budget = max(0, num_panos - len(selected))
    debug: Dict[str, object] = {
        "removed_near_captured": removed_near,
        "forced_hole_ids": [c.candidate_id for c in forced],
        "selection_trace": [],
    }

    if remaining_budget == 0 or not normal:
        for c in selected:
            c.selected = True
        return selected, debug

    if config.quality_strategy == "render_quality_band":
        raise NotImplementedError(
            "render_quality_band is intentionally reserved for a later experiment; "
            "V1 uses quality_strategy='none'."
        )

    vis_rows: Dict[int, np.ndarray] = {}
    counts = np.zeros(len(visibility_model.sample_points), dtype=np.int32)
    if config.information_gain_strategy != "none" or config.strategy == "greedy_coverage":
        vis_rows = _candidate_visibility_rows(normal + forced, visibility_model, "stage3_select")
        if config.coverage_include_captured and len(captured_seed_cameras):
            counts += visibility_model.coverage_counts(
                captured_seed_cameras,
                key_prefix="stage3_captured_seed",
            )
        for c in forced:
            if c.candidate_id in vis_rows:
                counts += vis_rows[c.candidate_id].astype(np.int32)

    if config.strategy == "greedy_coverage":
        pool = list(normal)
        for _ in range(min(remaining_budget, len(pool))):
            best = None
            best_gain = -float("inf")
            for c in pool:
                gain = visibility_model.marginal_gain(vis_rows[c.candidate_id], counts)
                if gain > best_gain + 1e-12 or (
                    abs(gain - best_gain) <= 1e-12 and (best is None or c.candidate_id < best.candidate_id)
                ):
                    best = c
                    best_gain = gain
            assert best is not None
            best.information_gain = float(best_gain)
            selected.append(best)
            counts += vis_rows[best.candidate_id].astype(np.int32)
            pool.remove(best)
            debug["selection_trace"].append({"candidate_id": best.candidate_id, "gain": best_gain})
    elif config.strategy in ("angular_fps", "utility_angular_fps"):
        pool = list(normal)
        seed_dirs = _initial_seed_directions(
            forced,
            captured_seed_cameras,
            scene_center,
            mode,
            config.reference,
        )
        # Deterministic first sample: if there is no seed, use the candidate most
        # central in grid order, rather than random initialization.
        for _ in range(min(remaining_budget, len(pool))):
            distances = np.array(
                [_min_angular_distance(c.observation_direction, seed_dirs) for c in pool],
                dtype=np.float64,
            )
            max_distance = float(np.max(distances))
            eligible = np.flatnonzero(distances >= float(config.ig_tie_ratio) * max_distance - 1e-12)

            best_local = int(eligible[0])
            best_secondary = -float("inf")
            for local_idx in eligible:
                c = pool[int(local_idx)]
                secondary = float(c.quality_score)
                if config.information_gain_strategy != "none" and c.candidate_id in vis_rows:
                    gain = visibility_model.marginal_gain(vis_rows[c.candidate_id], counts)
                    c.information_gain = gain
                    secondary = gain if config.strategy == "angular_fps" else gain * c.quality_score
                elif config.strategy == "utility_angular_fps":
                    secondary = max_distance * c.quality_score
                if secondary > best_secondary + 1e-12 or (
                    abs(secondary - best_secondary) <= 1e-12
                    and c.candidate_id < pool[best_local].candidate_id
                ):
                    best_local = int(local_idx)
                    best_secondary = secondary

            best = pool.pop(best_local)
            selected.append(best)
            seed_dirs.append(normalize(best.observation_direction))
            if best.candidate_id in vis_rows:
                counts += vis_rows[best.candidate_id].astype(np.int32)
            debug["selection_trace"].append(
                {
                    "candidate_id": best.candidate_id,
                    "min_angular_deg": float(np.degrees(distances[best_local])),
                    "secondary": float(best_secondary),
                }
            )
    elif config.strategy == "legacy_position_fps":
        pool = list(normal)
        seed_positions = [c.camera.position.copy() for c in forced]
        if config.reference == "captured_seeded":
            seed_positions.extend(c.position.copy() for c in captured_seed_cameras)
        elif config.reference != "generated_only":
            raise ValueError(f"Unknown selection reference: {config.reference}")

        for _ in range(min(remaining_budget, len(pool))):
            if not seed_positions:
                positions = np.stack([c.camera.position for c in pool], axis=0)
                centroid = np.mean(positions, axis=0)
                distances = np.linalg.norm(positions - centroid[None, :], axis=1)
            else:
                distances = np.array(
                    [_min_position_distance(c.camera.position, seed_positions) for c in pool]
                )
            best_local = int(np.argmax(distances))
            best = pool.pop(best_local)
            selected.append(best)
            seed_positions.append(best.camera.position.copy())
            debug["selection_trace"].append(
                {"candidate_id": best.candidate_id, "position_distance": float(distances[best_local])}
            )
    else:
        raise ValueError(f"Unknown selection strategy: {config.strategy}")

    for c in selected:
        c.selected = True
    debug["num_filtered_candidates"] = len(filtered)
    debug["num_selected"] = len(selected)
    return selected, debug


def order_selected_views(
    selected: Sequence[SelectionCandidate],
    config: Optional[SelectionConfig] = None,
) -> List[SelectionCandidate]:
    config = config or SelectionConfig()
    selected = list(selected)
    if config.ordering_strategy == "selection_order":
        return selected
    if config.ordering_strategy == "grid_order":
        return sorted(
            selected,
            key=lambda c: (
                c.row if c.row is not None else 10**9,
                c.col if c.col is not None else 10**9,
                0 if c.origin == CandidateOrigin.HOLE else 1,
                c.candidate_id,
            ),
        )
    if config.ordering_strategy == "nearest_neighbor":
        if len(selected) <= 2:
            return selected
        remaining = list(selected)
        # Stable start from smallest grid order.
        remaining.sort(
            key=lambda c: (
                c.row if c.row is not None else 10**9,
                c.col if c.col is not None else 10**9,
                c.candidate_id,
            )
        )
        ordered = [remaining.pop(0)]
        while remaining:
            prev = ordered[-1]
            best_idx = 0
            best_score = float("inf")
            for i, c in enumerate(remaining):
                angle = angular_distance_rad(prev.camera.forward, c.camera.forward)
                pos = float(np.linalg.norm(prev.camera.position - c.camera.position))
                score = angle + float(config.position_distance_weight) * pos
                if score < best_score:
                    best_score = score
                    best_idx = i
            ordered.append(remaining.pop(best_idx))
        return ordered
    raise ValueError(f"Unknown ordering_strategy: {config.ordering_strategy}")


if __name__ == "__main__":
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert abs(angular_distance_deg(a, b) - 90.0) < 1e-8
    print("stage3.selection self-test passed")
