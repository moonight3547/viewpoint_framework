#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Deterministic, capacity-balanced content blocks for V3.4 outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Any, Sequence

import numpy as np

from viewpoint_framework.stage3.selection import angular_distance_rad
from viewpoint_framework.stage3.types import BlockPlan, PanoBlock, SelectionCandidate
from viewpoint_framework.stage3.visibility import NullVisibilityModel, VisibilityModel


@dataclass
class BlockReferenceConfig:
    strategy: str = "block_target_coverage"
    max_refs: int = 6
    pool: str = "selected_only"
    fill_to_max_refs: bool = False
    min_reference_count: int = 1
    low_support_action: str = "warn_and_record"


@dataclass
class BlockConfig:
    mode: str = "off"
    strategy: str = "balanced_content_visibility"
    trunk_frames: int = 7
    target_chunks: int = 4
    min_chunks: int = 2
    max_chunks: int = 6
    frame_count_policy: str = "trim_low_value"
    descriptor: str = "gaussian_visibility"
    content_weight: float = 0.75
    references: BlockReferenceConfig = field(default_factory=BlockReferenceConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BlockConfig":
        payload = dict(data)
        references = BlockReferenceConfig(**payload.pop("references", {}))
        cfg = cls(**payload)
        cfg.references = references
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.mode not in ("off", "content"):
            raise ValueError("blocks.mode must be 'off' or 'content'")
        if int(self.trunk_frames) <= 0:
            raise ValueError("blocks.trunk_frames must be positive")
        if not (1 <= int(self.min_chunks) <= int(self.target_chunks) <= int(self.max_chunks)):
            raise ValueError("blocks chunk limits must satisfy 1 <= min <= target <= max")
        if self.frame_count_policy not in ("trim_low_value", "strict"):
            raise ValueError("blocks.frame_count_policy must be trim_low_value or strict")
        if not (0.0 <= float(self.content_weight) <= 1.0):
            raise ValueError("blocks.content_weight must be in [0, 1]")
        if not (1 <= int(self.references.max_refs) <= 6):
            raise ValueError("blocks.references.max_refs must be in [1, 6]")


def _pose_affinity(candidates: Sequence[SelectionCandidate], scene_scale: float) -> np.ndarray:
    n = len(candidates)
    affinity = np.eye(n, dtype=np.float64)
    positions = np.stack([c.camera.position for c in candidates], axis=0)
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    positive = distances[distances > 1e-8]
    position_sigma = float(np.median(positive)) if len(positive) else max(scene_scale * 0.1, 1e-6)
    angle_sigma = np.radians(25.0)
    for i in range(n):
        for j in range(i + 1, n):
            angle = angular_distance_rad(candidates[i].camera.forward, candidates[j].camera.forward)
            value = np.exp(-0.5 * (angle / angle_sigma) ** 2) * np.exp(
                -0.5 * (distances[i, j] / max(position_sigma, 1e-8)) ** 2
            )
            affinity[i, j] = affinity[j, i] = float(value)
    return affinity


def _content_affinity(rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n = len(rows)
    affinity = np.eye(n, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            union = rows[i] | rows[j]
            denom = float(np.sum(weights[union]))
            value = 0.0 if denom <= 1e-12 else float(np.sum(weights[rows[i] & rows[j]]) / denom)
            affinity[i, j] = affinity[j, i] = value
    return affinity


def _planned_sizes(frame_count: int, config: BlockConfig) -> list[int]:
    trunk = int(config.trunk_frames)
    chunks = frame_count // trunk
    if chunks == 0:
        return []
    if chunks == 1:
        return [trunk]
    minimum = int(config.min_chunks)
    maximum = int(config.max_chunks)
    k_low = max(1, int(ceil(chunks / maximum)))
    k_high = max(1, chunks // minimum)
    target_k = int(ceil(chunks / int(config.target_chunks)))
    block_count = min(max(target_k, k_low), k_high)
    base, remainder = divmod(chunks, block_count)
    return [trunk * (base + (1 if i < remainder else 0)) for i in range(block_count)]


def _trim_to_trunks(
    candidates: list[SelectionCandidate],
    affinity: np.ndarray,
    config: BlockConfig,
) -> tuple[list[SelectionCandidate], np.ndarray, list[int]]:
    remainder = len(candidates) % int(config.trunk_frames)
    if remainder == 0:
        return candidates, affinity, []
    if config.frame_count_policy == "strict":
        raise ValueError(
            f"Block mode requires a multiple of {config.trunk_frames} frames; "
            f"received {len(candidates)}"
        )
    active = list(range(len(candidates)))
    dropped: list[int] = []
    for _ in range(remainder):
        # Remove the most redundant target, preserving rare/content-boundary views.
        sub = affinity[np.ix_(active, active)].copy()
        np.fill_diagonal(sub, -np.inf)
        redundancy = np.max(sub, axis=1)
        local = max(
            range(len(active)),
            key=lambda i: (float(redundancy[i]), -int(candidates[active[i]].candidate_id)),
        )
        dropped.append(int(candidates[active[local]].candidate_id))
        active.pop(local)
    return [candidates[i] for i in active], affinity[np.ix_(active, active)], dropped


def _choose_seeds(affinity: np.ndarray, count: int) -> list[int]:
    first = int(np.argmax(np.mean(affinity, axis=1)))
    seeds = [first]
    while len(seeds) < count:
        similarity = np.max(affinity[:, seeds], axis=1)
        similarity[seeds] = np.inf
        seeds.append(int(np.argmin(similarity)))
    return seeds


def _balanced_assign(affinity: np.ndarray, sizes: Sequence[int]) -> list[list[int]]:
    seeds = _choose_seeds(affinity, len(sizes))
    groups = [[seed] for seed in seeds]
    remaining_capacity = [int(size) - 1 for size in sizes]
    unassigned = set(range(len(affinity))) - set(seeds)
    while unassigned:
        choices = []
        for index in sorted(unassigned):
            ranked = sorted(
                (float(affinity[index, seeds[b]]), b)
                for b in range(len(groups)) if remaining_capacity[b] > 0
            )
            score, block_id = ranked[-1]
            choices.append((score, -index, block_id, index))
        _, _, block_id, index = max(choices)
        groups[block_id].append(index)
        remaining_capacity[block_id] -= 1
        unassigned.remove(index)
    return groups


def _order_group(group: Sequence[int], affinity: np.ndarray) -> tuple[list[int], dict[str, float]]:
    sub = affinity[np.ix_(group, group)]
    medoid_local = int(np.argmax(np.mean(sub, axis=1)))
    remaining = set(group)
    ordered = [int(group[medoid_local])]
    remaining.remove(ordered[0])
    jumps = []
    while remaining:
        previous = ordered[-1]
        nxt = max(sorted(remaining), key=lambda index: (float(affinity[previous, index]), -index))
        jumps.append(1.0 - float(affinity[previous, nxt]))
        ordered.append(int(nxt))
        remaining.remove(nxt)
    return ordered, {
        "mean_neighbor_jump": float(np.mean(jumps)) if jumps else 0.0,
        "max_neighbor_jump": float(np.max(jumps)) if jumps else 0.0,
    }


def plan_content_blocks(
    candidates: Sequence[SelectionCandidate],
    *,
    scene_scale: float,
    visibility_model: VisibilityModel,
    config: BlockConfig,
) -> BlockPlan:
    config.validate()
    candidates = list(candidates)
    if not candidates:
        raise ValueError("Cannot build blocks from an empty target set")

    pose = _pose_affinity(candidates, scene_scale)
    descriptor_strategy = "pose_fallback"
    descriptor_error = None
    affinity = pose
    if not isinstance(visibility_model, NullVisibilityModel) and len(visibility_model.sample_points):
        try:
            rows = visibility_model.visibility_matrix(
                [c.camera for c in candidates], key_prefix="v34_block_target")
            content = _content_affinity(rows, visibility_model.sample_weights)
            weight = float(config.content_weight)
            affinity = weight * content + (1.0 - weight) * pose
            descriptor_strategy = config.descriptor
        except Exception as exc:  # recorded, never presented as geometry content
            descriptor_error = f"{type(exc).__name__}: {exc}"

    retained, affinity, dropped = _trim_to_trunks(candidates, affinity, config)
    sizes = _planned_sizes(len(retained), config)
    if not sizes:
        raise ValueError(
            f"Block mode needs at least {config.trunk_frames} selected targets"
        )
    groups = _balanced_assign(affinity, sizes)
    blocks = []
    start = 0
    for block_id, group in enumerate(groups):
        ordered_indices, ordering_metrics = _order_group(group, affinity)
        ordered = [retained[i] for i in ordered_indices]
        cohesion = affinity[np.ix_(group, group)]
        blocks.append(PanoBlock(
            block_id=block_id,
            candidates=ordered,
            start_index=start,
            descriptor_strategy=descriptor_strategy,
            coverage_metrics={
                "mean_internal_affinity": float(np.mean(cohesion)),
            },
            ordering_metrics=ordering_metrics,
        ))
        start += len(ordered)
    return BlockPlan(
        blocks=blocks,
        dropped_candidate_ids=dropped,
        frame_count_policy=config.frame_count_policy,
        config={
            "mode": config.mode,
            "strategy": config.strategy,
            "trunk_frames": int(config.trunk_frames),
            "planned_sizes": sizes,
            "descriptor_strategy": descriptor_strategy,
            "descriptor_error": descriptor_error,
        },
    )


__all__ = [
    "BlockConfig",
    "BlockReferenceConfig",
    "plan_content_blocks",
]
