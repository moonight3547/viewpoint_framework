#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Standalone Stage-3 CLI.

Consumes Stage-2 ``gen_cameras.json`` / ``gen_cameras_meta.json`` and writes the
denoising input contract under --output_dir:
    pano_cameras.json
    pano_images/frame_XXXX.png
    traj_refs.json          # [[original captured indices]]
    traj_lens.json          # [num_panos actually selected]

Debug-only outputs are placed strictly under ``output_dir/debug``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from viewpoint_framework.cameras_util import load_cameras_json
from viewpoint_framework.gs_renderer import GsplatRenderer
from viewpoint_framework.points_util import load_ply_point_cloud
from viewpoint_framework.stage3.pipeline import (
    Stage3Config,
    candidates_from_files,
    run_stage3,
)


def _load_json(path: str) -> dict:
    with open(Path(path).expanduser().resolve(), "r", encoding="utf-8") as f:
        return json.load(f)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage-3 denoising-view selection and rendering")
    p.add_argument("--cameras", required=True, help="Original train_cameras.json")
    p.add_argument("--point_cloud", required=True, help="Aligned point cloud PLY")
    p.add_argument("--gaussian_ply", required=True, help="Aligned high-quality 3DGS PLY")
    p.add_argument("--select_view_dir", default=None, help="Directory containing selection.json / selected_cameras.json")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gen_cameras", default=None, help="Defaults to output_dir/gen_cameras.json")
    p.add_argument("--gen_meta", default=None, help="Defaults to output_dir/gen_cameras_meta.json")
    p.add_argument("--view_limits", default=None, help="Defaults to output_dir/view_limits.json")
    p.add_argument("--config_json", default=None)

    p.add_argument("--num_panos", type=int, default=49)
    p.add_argument("--num_refs", type=int, default=12)
    p.add_argument("--debug-mode", action="store_true")

    p.add_argument(
        "--selection-strategy",
        choices=("legacy_position_fps", "angular_fps", "utility_angular_fps", "greedy_coverage"),
        default=None,
    )
    p.add_argument(
        "--selection-reference",
        choices=("generated_only", "captured_seeded"),
        default=None,
    )
    p.add_argument(
        "--information-gain",
        choices=("none", "gaussian_visibility", "pointcloud_visibility"),
        default=None,
    )
    p.add_argument(
        "--hole-strategy",
        choices=("none", "gaussian_undercoverage", "pointcloud_gaussian_gap"),
        default=None,
    )
    p.add_argument(
        "--reference-strategy",
        choices=("legacy_global_fps", "target_coverage_greedy", "artifixer_style_covisibility"),
        default=None,
    )
    p.add_argument(
        "--ordering-strategy",
        choices=("grid_order", "nearest_neighbor", "selection_order"),
        default=None,
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--visibility-samples", type=int, default=None)
    p.add_argument("--visibility-max-dim", type=int, default=None)
    p.add_argument("--pointcloud_max_points", type=int, default=0)
    return p


def _config(args: argparse.Namespace) -> Stage3Config:
    cfg = Stage3Config()
    if args.config_json:
        cfg = Stage3Config.from_dict(_load_json(args.config_json))
    cfg.num_panos = int(args.num_panos)
    cfg.num_refs = int(args.num_refs)
    cfg.debug_mode = bool(args.debug_mode)
    if args.selection_strategy:
        cfg.selection.strategy = args.selection_strategy
    if args.selection_reference:
        cfg.selection.reference = args.selection_reference
    if args.information_gain:
        cfg.selection.information_gain_strategy = args.information_gain
    if args.hole_strategy:
        cfg.holes.strategy = args.hole_strategy
    if args.reference_strategy:
        cfg.references.strategy = args.reference_strategy
    if args.ordering_strategy:
        cfg.selection.ordering_strategy = args.ordering_strategy
    if args.visibility_samples is not None:
        cfg.visibility.max_samples = int(args.visibility_samples)
    if args.visibility_max_dim is not None:
        cfg.visibility.max_image_dim = int(args.visibility_max_dim)
    return cfg


def main() -> None:
    args = build_argparser().parse_args()
    cfg = _config(args)
    output = Path(args.output_dir).expanduser().resolve()
    gen_cameras = Path(args.gen_cameras).expanduser().resolve() if args.gen_cameras else output / "gen_cameras.json"
    gen_meta = Path(args.gen_meta).expanduser().resolve() if args.gen_meta else output / "gen_cameras_meta.json"
    view_limits = Path(args.view_limits).expanduser().resolve() if args.view_limits else output / "view_limits.json"

    captured = load_cameras_json(args.cameras)
    point_cloud = load_ply_point_cloud(args.point_cloud, max_points=args.pointcloud_max_points)
    candidates, mode = candidates_from_files(str(gen_cameras), str(gen_meta))
    limits = _load_json(str(view_limits))
    center = np.asarray(limits["target"], dtype=np.float64)
    gravity = np.asarray(limits.get("gravityCoordinate", np.eye(3)), dtype=np.float64)
    if gravity.shape == (3, 3):
        world_up = gravity[:, 1]
        fallback_axis = gravity[:, 2]
    else:
        world_up = np.array([0.0, 1.0, 0.0])
        fallback_axis = np.array([0.0, 0.0, 1.0])

    renderer = GsplatRenderer(args.gaussian_ply, device=args.device)
    result, paths = run_stage3(
        captured_cameras=captured,
        stage2_candidates=candidates,
        mode=mode,
        scene_center=center,
        world_up=world_up,
        fallback_axis=fallback_axis,
        renderer=renderer,
        output_dir=str(output),
        point_cloud_points=point_cloud.points,
        select_view_dir=args.select_view_dir,
        config=cfg,
    )

    print("=" * 72)
    print("Stage 3: Denoising View Selection")
    print("=" * 72)
    print(f"candidates : {len(candidates)}")
    print(f"holes      : {len(result.holes)}")
    print(f"hole views : {len(result.hole_views)}")
    print(f"panos      : {len(result.selected_cameras)} / requested {cfg.num_panos}")
    print(f"refs       : {len(result.reference_result.original_indices)} / requested {cfg.num_refs}")
    for key, value in paths.items():
        print(f"{key:12s}: {value}")


if __name__ == "__main__":
    main()
