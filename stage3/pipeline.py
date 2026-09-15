#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage-3 orchestration: geometric holes -> final panos -> references -> render."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from viewpoint_framework.cameras_util import Camera, load_cameras_json
from viewpoint_framework.gs_renderer import GsplatRenderer
from viewpoint_framework.pose_generation import CandidateStatus, PoseGenerationResult
from viewpoint_framework.scene_types import CameraMode
from viewpoint_framework.stage3.hole_detection import (
    HoleDetectionConfig,
    detect_geometric_holes,
    detect_pointcloud_gaussian_gaps,
    generate_focused_hole_views,
)
from viewpoint_framework.stage3.reference_selection import (
    ReferenceSelectionConfig,
    select_references,
)
from viewpoint_framework.stage3.render_output import save_stage3_outputs
from viewpoint_framework.stage3.selected_views import load_selected_view_set
from viewpoint_framework.stage3.selection import (
    SelectionConfig,
    order_selected_views,
    select_views,
)
from viewpoint_framework.stage3.types import (
    CandidateOrigin,
    SelectionCandidate,
    Stage3Result,
)
from viewpoint_framework.stage3.visibility import (
    NullVisibilityModel,
    VisibilityConfig,
    VisibilityModel,
    build_visibility_model,
)


EPS = 1e-8


@dataclass
class Stage3Config:
    num_panos: int = 49
    num_refs: int = 12
    debug_mode: bool = False

    # The selected 40 captured views are expected in --select_view_dir.  This only
    # controls standalone fallback when that directory is not supplied.
    selected_view_fallback_count: int = 40

    selection: SelectionConfig = field(default_factory=SelectionConfig)
    visibility: VisibilityConfig = field(default_factory=VisibilityConfig)
    holes: HoleDetectionConfig = field(default_factory=HoleDetectionConfig)
    references: ReferenceSelectionConfig = field(default_factory=ReferenceSelectionConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Stage3Config":
        payload = dict(data)
        selection = SelectionConfig(**payload.pop("selection", {}))
        visibility = VisibilityConfig(**payload.pop("visibility", {}))
        holes = HoleDetectionConfig(**payload.pop("holes", {}))
        references = ReferenceSelectionConfig(**payload.pop("references", {}))
        cfg = cls(**payload)
        cfg.selection = selection
        cfg.visibility = visibility
        cfg.holes = holes
        cfg.references = references
        return cfg


def estimate_scene_scale(
    point_cloud_points: Optional[np.ndarray],
    captured_cameras: Sequence[Camera],
    center: np.ndarray,
) -> float:
    if point_cloud_points is not None:
        points = np.asarray(point_cloud_points, dtype=np.float64).reshape(-1, 3)
        finite = points[np.all(np.isfinite(points), axis=1)]
        if len(finite):
            extent = np.max(finite, axis=0) - np.min(finite, axis=0)
            diag = float(np.linalg.norm(extent))
            if diag > EPS:
                return diag
    if captured_cameras:
        radii = [float(np.linalg.norm(c.position - center)) for c in captured_cameras]
        med = float(np.median(radii))
        if med > EPS:
            return 2.0 * med
    return 1.0


def candidates_from_pose_result(result: PoseGenerationResult) -> List[SelectionCandidate]:
    output: List[SelectionCandidate] = []
    valid_index = 0
    for candidate in result.candidates:
        if candidate.status != CandidateStatus.VALID or candidate.camera is None:
            continue
        output.append(
            SelectionCandidate(
                candidate_id=valid_index,
                camera=candidate.camera,
                origin=CandidateOrigin.GRID,
                grid_id=int(candidate.grid_id),
                row=int(candidate.row),
                col=int(candidate.col),
                azimuth_deg=float(candidate.azimuth_deg),
                elevation_deg=float(candidate.elevation_deg),
                observation_direction=np.asarray(candidate.direction, dtype=np.float64),
                signed_radius=float(candidate.final_signed_radius),
                crossed_center=bool(candidate.crossed_center),
                safety_clearance=(
                    None if candidate.final_clearance is None else float(candidate.final_clearance)
                ),
                radius_confidence=float(candidate.initial_radius_confidence),
                depth_confidence=(
                    None
                    if candidate.depth_probe is None
                    else float(candidate.depth_probe.confidence)
                ),
            )
        )
        valid_index += 1
    return output


def candidates_from_files(gen_cameras_path: str, gen_meta_path: str) -> tuple[List[SelectionCandidate], CameraMode]:
    cameras = load_cameras_json(gen_cameras_path)
    with open(Path(gen_meta_path).expanduser().resolve(), "r", encoding="utf-8") as f:
        meta = json.load(f)
    valid_meta = [c for c in meta.get("candidates", []) if str(c.get("status")) == "valid"]
    if len(valid_meta) != len(cameras):
        raise ValueError(
            "gen_cameras.json / gen_cameras_meta.json mismatch: "
            f"{len(cameras)} cameras vs {len(valid_meta)} valid metadata records"
        )
    mode_value = meta.get("mode", {}).get("mode", "outside_in")
    mode = CameraMode(mode_value)
    output: List[SelectionCandidate] = []
    for i, (camera, record) in enumerate(zip(cameras, valid_meta)):
        depth = record.get("depth_probe") or {}
        geometry = record.get("geometry_metadata") or {}
        fps_direction = geometry.get(
            "fps_direction", geometry.get("position_direction", record.get("direction", camera.forward))
        )
        output.append(
            SelectionCandidate(
                candidate_id=i,
                camera=camera,
                origin=CandidateOrigin.GRID,
                grid_id=int(record.get("grid_id", i)),
                row=int(record.get("row", 0)),
                col=int(record.get("col", i)),
                azimuth_deg=float(geometry.get("position_azimuth_deg", record.get("azimuth_deg", 0.0))),
                elevation_deg=float(geometry.get("position_elevation_deg", record.get("elevation_deg", 0.0))),
                observation_direction=np.asarray(fps_direction, dtype=np.float64),
                signed_radius=float(record.get("final_signed_radius", 0.0)),
                crossed_center=bool(record.get("crossed_center", False)),
                safety_clearance=record.get("final_clearance"),
                radius_confidence=record.get("initial_radius_confidence"),
                depth_confidence=depth.get("confidence") if isinstance(depth, dict) else None,
            )
        )
    return output, mode


def _visibility_model_for_strategy(
    strategy: str,
    *,
    renderer: Optional[GsplatRenderer],
    point_cloud_points: Optional[np.ndarray],
    scene_scale: float,
    config: VisibilityConfig,
) -> VisibilityModel:
    cfg = VisibilityConfig(**asdict(config))
    cfg.strategy = strategy
    return build_visibility_model(
        strategy,
        renderer=renderer,
        point_cloud_points=point_cloud_points,
        scene_scale=scene_scale,
        config=cfg,
    )


def _validate_hole_views(
    hole_views: List[SelectionCandidate],
    holes,
    model: VisibilityModel,
    min_fraction: float,
) -> tuple[List[SelectionCandidate], Dict[int, float]]:
    hole_map = {h.hole_id: h for h in holes}
    kept = []
    fractions: Dict[int, float] = {}
    for candidate in hole_views:
        hole = hole_map.get(candidate.hole_id)
        if hole is None:
            continue
        vis = model.visibility(candidate.camera, cache_key=f"hole_view:{candidate.candidate_id}")
        target = hole.sample_indices
        if len(target) == 0:
            continue
        fraction = float(np.mean(vis[target]))
        fractions[candidate.candidate_id] = fraction
        if fraction >= float(min_fraction):
            candidate.notes.append(f"hole_visible_fraction={fraction:.3f}")
            kept.append(candidate)
    return kept, fractions


def run_stage3(
    *,
    captured_cameras: Sequence[Camera],
    stage2_candidates: Sequence[SelectionCandidate],
    mode: CameraMode,
    scene_center: np.ndarray,
    world_up: np.ndarray,
    fallback_axis: np.ndarray,
    renderer: GsplatRenderer,
    output_dir: str,
    point_cloud_points: Optional[np.ndarray] = None,
    select_view_dir: Optional[str] = None,
    config: Optional[Stage3Config] = None,
) -> tuple[Stage3Result, Dict[str, str]]:
    config = config or Stage3Config()
    stage2_candidates = list(stage2_candidates)
    captured_cameras = list(captured_cameras)
    if not stage2_candidates:
        raise ValueError("Stage 3 requires at least one Stage-2 valid candidate.")

    scene_scale = estimate_scene_scale(point_cloud_points, captured_cameras, scene_center)
    selected_views = load_selected_view_set(
        select_view_dir,
        captured_cameras,
        fallback_count=config.selected_view_fallback_count,
    )
    anchor_cameras = selected_views.cameras

    # Build only the representation-aware models actually required by configured
    # algorithms.  This keeps normal-mode latency predictable.
    shared_gaussian_model: Optional[VisibilityModel] = None
    hole_model: VisibilityModel = NullVisibilityModel()
    if config.holes.strategy == "gaussian_undercoverage":
        shared_gaussian_model = _visibility_model_for_strategy(
            "gaussian_visibility",
            renderer=renderer,
            point_cloud_points=point_cloud_points,
            scene_scale=scene_scale,
            config=config.visibility,
        )
        hole_model = shared_gaussian_model
    elif config.holes.strategy == "pointcloud_gaussian_gap":
        hole_model = _visibility_model_for_strategy(
            "pointcloud_visibility",
            renderer=renderer,
            point_cloud_points=point_cloud_points,
            scene_scale=scene_scale,
            config=config.visibility,
        )
    elif config.holes.strategy != "none":
        raise ValueError(f"Unknown hole strategy: {config.holes.strategy}")

    ig_strategy = config.selection.information_gain_strategy
    if ig_strategy == "none":
        selection_model: VisibilityModel = NullVisibilityModel()
    elif ig_strategy == "gaussian_visibility" and shared_gaussian_model is not None:
        selection_model = shared_gaussian_model
    else:
        selection_model = _visibility_model_for_strategy(
            ig_strategy,
            renderer=renderer,
            point_cloud_points=point_cloud_points,
            scene_scale=scene_scale,
            config=config.visibility,
        )

    # Geometric hole detection is based on 3-D representation coverage by the
    # already-selected captured anchors, not on Stage-2 angular-grid gaps.
    if config.holes.strategy == "gaussian_undercoverage":
        holes, anchor_counts = detect_geometric_holes(
            hole_model,
            anchor_cameras,
            scene_scale,
            config=config.holes,
        )
    elif config.holes.strategy == "pointcloud_gaussian_gap":
        holes, anchor_counts = detect_pointcloud_gaussian_gaps(
            hole_model,
            anchor_cameras,
            renderer,
            scene_scale,
            config=config.holes,
        )
    else:
        holes = []
        anchor_counts = np.zeros(len(hole_model.sample_points), dtype=np.int32)
    hole_views: List[SelectionCandidate] = []
    hole_records = []
    hole_visibility_fraction: Dict[int, float] = {}
    if holes:
        hole_views, hole_records = generate_focused_hole_views(
            holes,
            hole_model,
            stage2_candidates,
            scene_center,
            world_up,
            fallback_axis,
            next_candidate_id=max(c.candidate_id for c in stage2_candidates) + 1,
            config=config.holes,
        )
        hole_views, hole_visibility_fraction = _validate_hole_views(
            hole_views,
            holes,
            hole_model,
            config.holes.min_target_visible_fraction,
        )
        valid_ids = {c.candidate_id for c in hole_views}
        hole_records = [r for r in hole_records if r.candidate_id in valid_ids]

    all_candidates = list(stage2_candidates) + list(hole_views)

    selected, selection_debug = select_views(
        all_candidates,
        num_panos=config.num_panos,
        captured_seed_cameras=anchor_cameras,
        scene_center=scene_center,
        scene_scale=scene_scale,
        mode=mode,
        config=config.selection,
        visibility_model=selection_model,
    )
    selected = order_selected_views(selected, config.selection)
    selected_cameras = [c.camera for c in selected]

    # Reference selector can reuse the Gaussian visibility model already computed
    # for holes.  Create it lazily only for the ArtiFixer-style strategy.
    ref_model: VisibilityModel = NullVisibilityModel()
    if config.references.strategy == "artifixer_style_covisibility":
        if shared_gaussian_model is None:
            shared_gaussian_model = _visibility_model_for_strategy(
                "gaussian_visibility",
                renderer=renderer,
                point_cloud_points=point_cloud_points,
                scene_scale=scene_scale,
                config=config.visibility,
            )
        ref_model = shared_gaussian_model

    ref_result = select_references(
        selected_views,
        captured_cameras,
        selected_cameras,
        num_refs=config.num_refs,
        scene_scale=scene_scale,
        config=config.references,
        visibility_model=ref_model,
    )

    debug = {
        "scene_scale": scene_scale,
        "selected_view_indices": selected_views.original_indices,
        "selection": selection_debug,
        "hole_visibility_fraction": hole_visibility_fraction,
        "anchor_gaussian_coverage": (
            {
                "mean": float(np.mean(anchor_counts)),
                "unseen_ratio": float(np.mean(anchor_counts == 0)),
                "max": int(np.max(anchor_counts)) if len(anchor_counts) else 0,
            }
            if len(anchor_counts)
            else {}
        ),
        "config": asdict(config),
    }

    result = Stage3Result(
        selected_candidates=selected,
        selected_cameras=selected_cameras,
        reference_result=ref_result,
        all_candidates=all_candidates,
        holes=holes,
        hole_views=hole_records,
        debug=debug,
    )
    paths = save_stage3_outputs(
        result,
        renderer,
        output_dir,
        debug_mode=config.debug_mode,
        stage2_grid_candidates=stage2_candidates,
    )
    return result, paths


if __name__ == "__main__":
    print("stage3.pipeline: import OK")
