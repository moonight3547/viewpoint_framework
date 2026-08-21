#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""CLI for stage-2 candidate viewpoint pose generation.

Recommended first-pass usage
----------------------------
python -m viewpoint_framework.generate_poses \
    --cameras train_cameras.json \
    --point_cloud pi3_init_aligned.ply \
    --gaussian_ply point_cloud_final.ply \
    --output_dir outputs/pose_generation

The command writes:
    view_limits.json
    gen_cameras.json
    gen_cameras_meta.json

``gen_cameras.json`` is the same 18-D protocol consumed by visualize_cameras.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from viewpoint_framework.cameras_util import load_cameras_json
from viewpoint_framework.gs_depth_probe import (
    DepthProbeConfig,
    GsplatDepthProbe,
    NullDepthProbe,
)
from viewpoint_framework.points_util import load_ply_point_cloud
from viewpoint_framework.pose_generation import (
    PoseGenerationConfig,
    generate_candidate_poses,
    save_pose_generation_result,
)
from viewpoint_framework.scene_understanding import (
    SceneUnderstandingConfig,
    understand_scene,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate all geometry-safe candidate cameras on a spherical angular grid."
    )

    parser.add_argument("--cameras", required=True, type=str)
    parser.add_argument("--point_cloud", required=True, type=str)
    parser.add_argument(
        "--gaussian_ply",
        type=str,
        default=None,
        help="Aligned 3DGS PLY. If omitted, placement keeps the radius prior.",
    )
    parser.add_argument("--output_dir", type=str, default="pose_generation")

    parser.add_argument(
        "--scene_config_json",
        type=str,
        default=None,
        help="Optional stage-1 SceneUnderstandingConfig JSON.",
    )
    parser.add_argument(
        "--config_json",
        type=str,
        default=None,
        help="Optional PoseGenerationConfig JSON.",
    )

    # High-frequency experiment overrides.
    parser.add_argument("--mode", choices=("auto", "outside_in", "inside_out"), default="auto")
    parser.add_argument("--grid_gap", type=float, default=None)
    parser.add_argument("--focal_ratio", type=float, default=None)
    parser.add_argument(
        "--radius_strategy",
        choices=("directional", "azimuth_only", "bbox_median"),
        default=None,
    )
    parser.add_argument(
        "--outside_placement",
        choices=("prior_only", "depth_backoff"),
        default=None,
    )
    parser.add_argument(
        "--inside_placement",
        choices=("prior_only", "no_crossing", "center_crossing_depth"),
        default=None,
    )
    parser.add_argument("--safety_ratio", type=float, default=None)
    parser.add_argument("--safety_abs", type=float, default=None)
    parser.add_argument("--disable_path_safety", action="store_true")

    # 3DGS depth probe.
    parser.add_argument("--no_gs_depth", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--probe_max_dim", type=int, default=256)
    parser.add_argument("--probe_crop_ratio", type=float, default=0.35)
    parser.add_argument("--probe_quantile", type=float, default=0.10)
    parser.add_argument("--probe_alpha_threshold", type=float, default=0.05)
    parser.add_argument(
        "--depth_strategy",
        choices=("central_low_quantile", "median", "mean"),
        default="central_low_quantile",
    )

    parser.add_argument(
        "--pointcloud_max_points",
        type=int,
        default=0,
        help="<=0 keeps all points for scene analysis and safety.",
    )
    return parser


def _load_json(path: str) -> dict:
    with open(Path(path).expanduser().resolve(), "r", encoding="utf-8") as f:
        return json.load(f)


def _build_pose_config(args: argparse.Namespace) -> PoseGenerationConfig:
    config = PoseGenerationConfig()
    if args.config_json:
        config = PoseGenerationConfig.from_dict(_load_json(args.config_json))

    if args.mode != "auto":
        config.mode_strategy = "forced"
        config.forced_mode = args.mode
    if args.grid_gap is not None:
        config.azimuth_step_deg = args.grid_gap
        config.elevation_step_deg = args.grid_gap
    if args.focal_ratio is not None:
        config.focal_ratio = args.focal_ratio
    if args.radius_strategy is not None:
        config.radius_strategy = args.radius_strategy
    if args.outside_placement is not None:
        config.outside_placement_strategy = args.outside_placement
    if args.inside_placement is not None:
        config.inside_placement_strategy = args.inside_placement
    if args.safety_ratio is not None:
        config.geometry.clearance_ratio = args.safety_ratio
    if args.safety_abs is not None:
        config.geometry.clearance_abs = args.safety_abs
    if args.disable_path_safety:
        config.use_path_safety = False

    return config


def main() -> None:
    args = build_argparser().parse_args()
    pose_config = _build_pose_config(args)

    cameras = load_cameras_json(args.cameras)
    point_cloud = load_ply_point_cloud(
        args.point_cloud,
        max_points=args.pointcloud_max_points,
    )

    scene_config = SceneUnderstandingConfig.default()
    if args.scene_config_json:
        scene_config = SceneUnderstandingConfig.from_dict(
            _load_json(args.scene_config_json)
        )

    scene_result = understand_scene(
        cameras=cameras,
        config=scene_config,
        point_cloud_points=point_cloud.points,
        metadata={
            "source_cameras": str(Path(args.cameras).expanduser().resolve()),
            "source_point_cloud": str(Path(args.point_cloud).expanduser().resolve()),
        },
    )

    if args.gaussian_ply and not args.no_gs_depth:
        probe_config = DepthProbeConfig(
            strategy=args.depth_strategy,
            max_image_dim=args.probe_max_dim,
            central_crop_ratio=args.probe_crop_ratio,
            depth_quantile=args.probe_quantile,
            alpha_threshold=args.probe_alpha_threshold,
        )
        depth_probe = GsplatDepthProbe(
            gaussian_ply=args.gaussian_ply,
            config=probe_config,
            device=args.device,
        )
    else:
        depth_probe = NullDepthProbe()

    result = generate_candidate_poses(
        captured_cameras=cameras,
        scene_result=scene_result,
        point_cloud_points=point_cloud.points,
        depth_probe=depth_probe,
        config=pose_config,
    )
    paths = save_pose_generation_result(result, args.output_dir)

    print("=" * 72)
    print("Pose Generation")
    print("=" * 72)
    print(f"default mode : {result.mode.mode.value} (confidence={result.mode.confidence:.3f})")
    print(f"grid         : {len(result.candidates)} candidates")
    print(f"valid        : {len(result.valid_cameras)}")
    print(f"rejected     : {len(result.candidates) - len(result.valid_cameras)}")
    print(f"view limits  : {paths['view_limits']}")
    print(f"cameras      : {paths['gen_cameras']}")
    print(f"metadata     : {paths['gen_cameras_meta']}")
    print()
    print("Visualize with:")
    print(
        "python -m viewpoint_framework.visualize_cameras "
        f"--point_cloud {args.point_cloud} "
        f"--captured_cameras {args.cameras} "
        f"--generated_cameras {paths['gen_cameras']} "
        f"--output {Path(args.output_dir) / 'camera_vis' / 'index.html'}"
    )


if __name__ == "__main__":
    main()
