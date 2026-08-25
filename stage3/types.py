#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Data models for Stage-3 denoising-view selection.

The module intentionally contains no renderer / optimizer logic.  Keeping the
contracts here makes FPS, coverage, hole detection and reference selection
replaceable without changing the outer pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from viewpoint_framework.cameras_util import Camera


class CandidateOrigin(str, Enum):
    GRID = "grid"
    HOLE = "hole"


@dataclass
class SelectionCandidate:
    candidate_id: int
    camera: Camera
    origin: CandidateOrigin

    grid_id: Optional[int]
    row: Optional[int]
    col: Optional[int]
    azimuth_deg: Optional[float]
    elevation_deg: Optional[float]
    observation_direction: np.ndarray

    source_candidate_id: Optional[int] = None
    hole_id: Optional[int] = None
    forced_select: bool = False

    # Stage-2 cheap provenance used by later ablations.
    signed_radius: Optional[float] = None
    crossed_center: bool = False
    safety_clearance: Optional[float] = None
    radius_confidence: Optional[float] = None
    depth_confidence: Optional[float] = None

    # Stage-3 scores.  Expensive entries are filled only when requested.
    quality_score: float = 1.0
    information_gain: float = 0.0
    selected: bool = False
    notes: List[str] = field(default_factory=list)


@dataclass
class SelectedViewSet:
    """Preselected captured views provided by --select_view_dir."""

    original_indices: List[int]
    cameras: List[Camera]
    source_dir: Optional[str] = None
    image_paths: List[str] = field(default_factory=list)


@dataclass
class HoleCluster:
    hole_id: int
    sample_indices: np.ndarray
    centroid: np.ndarray
    severity: float
    coverage_mean: float
    point_count: int
    extent: np.ndarray
    notes: List[str] = field(default_factory=list)


@dataclass
class HoleViewRecord:
    hole_id: int
    candidate_id: int
    source_grid_candidate_id: int
    split_depth: int
    focal_scale: float
    fit_ok: bool
    severity: float
    note: str = ""


@dataclass
class ReferenceSelectionResult:
    strategy: str
    original_indices: List[int]
    candidate_pool_indices: List[int]
    scores: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage3Result:
    selected_candidates: List[SelectionCandidate]
    selected_cameras: List[Camera]
    reference_result: ReferenceSelectionResult
    all_candidates: List[SelectionCandidate]
    holes: List[HoleCluster]
    hole_views: List[HoleViewRecord]
    debug: Dict[str, Any] = field(default_factory=dict)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Camera):
        return {
            "index": int(value.index),
            "fx": float(value.fx),
            "fy": float(value.fy),
            "cx": float(value.cx),
            "cy": float(value.cy),
            "width": int(value.width),
            "height": int(value.height),
            "w2c": value.w2c.tolist(),
            "c2w": value.c2w.tolist(),
        }
    if is_dataclass(value):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


if __name__ == "__main__":
    print("stage3.types: import OK")
