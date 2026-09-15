#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""End-to-end Stage 1 -> Stage 2 -> Stage 3 viewpoint pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from viewpoint_framework.cameras_util import load_cameras_json
from viewpoint_framework.gs_depth_probe import DepthProbeConfig, GsplatDepthProbe
from viewpoint_framework.gs_renderer import (
    GaussianRendererConfig,
    GsplatRenderer,
    resolve_renderer_near_plane,
)
from viewpoint_framework.points_util import load_ply_point_cloud
from viewpoint_framework.pose_generation import (
    PoseGenerationConfig,
    generate_candidate_poses,
    save_pose_generation_result,
)
from viewpoint_framework.scene_understanding import SceneUnderstandingConfig, understand_scene
from viewpoint_framework.scene_types import to_jsonable
from viewpoint_framework.stage3.pipeline import (
    Stage3Config,
    candidates_from_pose_result,
    run_stage3,
)


def _load_json(path: str) -> dict:
    with open(Path(path).expanduser().resolve(), "r", encoding="utf-8") as f:
        return json.load(f)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run viewpoint framework Stage 1/2/3 end to end")
    p.add_argument("--cameras", required=True)
    p.add_argument("--point_cloud", required=True)
    p.add_argument("--gaussian_ply", required=True)
    p.add_argument("--select_view_dir", default=None)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--scene_config_json", default=None)
    p.add_argument("--pose_config_json", default=None)
    p.add_argument("--stage3_config_json", default=None)

    p.add_argument("--num_panos", type=int, default=49)
    p.add_argument("--num_refs", type=int, default=12)
    p.add_argument("--debug-mode", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--pointcloud_max_points", type=int, default=0)

    # Stage-2 high-value overrides.
    p.add_argument("--mode", choices=("auto", "outside_in", "inside_out"), default="auto")
    p.add_argument("--grid_gap", type=float, default=None)
    p.add_argument("--focal_ratio", type=float, default=None)
    p.add_argument("--depth_strategy", choices=("central_low_quantile", "median", "mean"), default="central_low_quantile")
    p.add_argument("--probe_max_dim", type=int, default=256)
    p.add_argument("--probe_crop_ratio", type=float, default=0.35)
    p.add_argument("--probe_quantile", type=float, default=0.10)
    p.add_argument("--probe_alpha_threshold", type=float, default=0.05)

    # Stage-3 experiment overrides.
    p.add_argument(
        "--selection-strategy",
        choices=("legacy_position_fps", "angular_fps", "utility_angular_fps", "greedy_coverage"),
        default=None,
    )
    p.add_argument("--selection-reference", choices=("generated_only", "captured_seeded"), default=None)
    p.add_argument("--information-gain", choices=("none", "gaussian_visibility", "pointcloud_visibility"), default=None)
    p.add_argument("--hole-strategy", choices=("none", "gaussian_undercoverage", "pointcloud_gaussian_gap"), default=None)
    p.add_argument(
        "--reference-strategy",
        choices=("legacy_global_fps", "target_coverage_greedy", "artifixer_style_covisibility"),
        default=None,
    )
    p.add_argument("--ordering-strategy", choices=("grid_order", "nearest_neighbor", "selection_order"), default=None)
    return p


def _build_configs(args):
    scene_cfg = SceneUnderstandingConfig.default()
    if args.scene_config_json:
        scene_cfg = SceneUnderstandingConfig.from_dict(_load_json(args.scene_config_json))

    pose_cfg = PoseGenerationConfig()
    if args.pose_config_json:
        pose_cfg = PoseGenerationConfig.from_dict(_load_json(args.pose_config_json))
    if args.mode != "auto":
        pose_cfg.mode_strategy = "forced"
        pose_cfg.forced_mode = args.mode
    if args.grid_gap is not None:
        pose_cfg.azimuth_step_deg = float(args.grid_gap)
        pose_cfg.elevation_step_deg = float(args.grid_gap)
    if args.focal_ratio is not None:
        pose_cfg.focal_ratio = float(args.focal_ratio)

    stage3_cfg = Stage3Config()
    if args.stage3_config_json:
        stage3_cfg = Stage3Config.from_dict(_load_json(args.stage3_config_json))
    stage3_cfg.num_panos = int(args.num_panos)
    stage3_cfg.num_refs = int(args.num_refs)
    stage3_cfg.debug_mode = bool(args.debug_mode)
    if args.selection_strategy:
        stage3_cfg.selection.strategy = args.selection_strategy
    if args.selection_reference:
        stage3_cfg.selection.reference = args.selection_reference
    if args.information_gain:
        stage3_cfg.selection.information_gain_strategy = args.information_gain
    if args.hole_strategy:
        stage3_cfg.holes.strategy = args.hole_strategy
    if args.reference_strategy:
        stage3_cfg.references.strategy = args.reference_strategy
    if args.ordering_strategy:
        stage3_cfg.selection.ordering_strategy = args.ordering_strategy
    return scene_cfg, pose_cfg, stage3_cfg


def main() -> None:
    args = build_argparser().parse_args()
    scene_cfg, pose_cfg, stage3_cfg = _build_configs(args)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    cameras = load_cameras_json(args.cameras)
    point_cloud = load_ply_point_cloud(args.point_cloud, max_points=args.pointcloud_max_points)

    # Stage 1: in-memory scene understanding.  Debug serialization lives under
    # output/debug only; normal mode avoids extra Stage-1 artifacts.
    scene_result = understand_scene(
        cameras=cameras,
        config=scene_cfg,
        point_cloud_points=point_cloud.points,
        metadata={
            "source_cameras": str(Path(args.cameras).expanduser().resolve()),
            "source_point_cloud": str(Path(args.point_cloud).expanduser().resolve()),
        },
    )
    if args.debug_mode:
        debug_dir = output / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        with open(debug_dir / "stage1_scene_profile.json", "w", encoding="utf-8") as f:
            json.dump(to_jsonable(scene_result.profile), f, indent=2)

    # Resolve z-near from captured trajectory scale after Stage 1 establishes
    # center/up. Load and split Gaussian tensors once for Stage 2 and Stage 3.
    frame = scene_result.profile.coordinate_frame
    near_plane = resolve_renderer_near_plane(
        cameras, scene_result.profile.center_fit.center, frame.y_axis,
        pose_cfg.renderer_near_plane,
    )
    renderer = GsplatRenderer(
        args.gaussian_ply,
        config=GaussianRendererConfig(
            near_plane=near_plane,
            skybox=pose_cfg.skybox,
        ),
        device=args.device,
    )
    depth_probe = GsplatDepthProbe(
        renderer=renderer,
        config=DepthProbeConfig(
            strategy=args.depth_strategy,
            max_image_dim=args.probe_max_dim,
            central_crop_ratio=args.probe_crop_ratio,
            depth_quantile=args.probe_quantile,
            alpha_threshold=args.probe_alpha_threshold,
        ),
    )

    # Stage 2: all geometry-safe grid candidates + endpoint view_limits.
    pose_result = generate_candidate_poses(
        captured_cameras=cameras,
        scene_result=scene_result,
        point_cloud_points=point_cloud.points,
        depth_probe=depth_probe,
        config=pose_cfg,
    )
    pose_result.renderer_metadata = {
        "near_plane": near_plane,
        "near_plane_strategy": pose_cfg.renderer_near_plane.strategy,
        "skybox": renderer.skybox_metadata,
    }
    pose_result.diagnostics.update({
        "renderer_near_plane": near_plane,
        "skybox_gaussian_count": renderer.skybox_metadata["skybox_gaussians"],
        "geometry_gaussian_count": renderer.skybox_metadata["geometry_gaussians"],
        "skybox_fraction": renderer.skybox_metadata["skybox_fraction"],
        "skybox_detection_confidence": renderer.skybox_metadata["detection_confidence"],
    })
    stage2_paths = save_pose_generation_result(pose_result, str(output))

    # Stage 3: geometric-hole mandatory views + final selection + references + RGB.
    stage2_candidates = candidates_from_pose_result(pose_result)
    stage3_result, stage3_paths = run_stage3(
        captured_cameras=cameras,
        stage2_candidates=stage2_candidates,
        mode=pose_result.mode.mode,
        scene_center=scene_result.profile.center_fit.center,
        world_up=frame.y_axis,
        fallback_axis=frame.z_axis,
        renderer=renderer,
        output_dir=str(output),
        point_cloud_points=point_cloud.points,
        select_view_dir=args.select_view_dir,
        config=stage3_cfg,
    )

    print("=" * 80)
    print("Viewpoint Framework End-to-End")
    print("=" * 80)
    print(f"Stage 1 mode        : {pose_result.mode.mode.value}")
    print(f"Stage 2 valid views : {len(stage2_candidates)}")
    print(f"Crossing capped     : {pose_result.diagnostics.get('inside_out_crossing_radius_capped_count', 0)}")
    print(f"Crossing fallback   : {pose_result.diagnostics.get('inside_out_crossing_fallback_initial_count', 0)}")
    print(f"GS geometry/skybox  : {len(renderer.geometry_means_np)} / {len(renderer.skybox_means_np)}")
    print(f"Stage 3 panos       : {len(stage3_result.selected_cameras)} / {stage3_cfg.num_panos}")
    print(f"Stage 3 refs        : {len(stage3_result.reference_result.original_indices)} / {stage3_cfg.num_refs}")
    print("-- Stage 2 --")
    for key, value in stage2_paths.items():
        print(f"  {key:20s} {value}")
    print("-- Stage 3 --")
    for key, value in stage3_paths.items():
        print(f"  {key:20s} {value}")


if __name__ == "__main__":
    main()
