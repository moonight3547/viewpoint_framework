#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""CLI entry for first-stage scene input understanding.

Example
-------
python -m viewpoint_framework.analyze_scene \
    --cameras train_cameras.json \
    --point_cloud pi3_init_aligned.ply \
    --output_dir outputs/scene_analysis \
    --no_serve

Legacy camera-only ablation:
python -m viewpoint_framework.analyze_scene \
    --cameras train_cameras.json \
    --preset legacy \
    --output_dir outputs/scene_analysis_legacy \
    --no_serve
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from viewpoint_framework.cameras_util import load_cameras_json
from viewpoint_framework.points_util import load_ply_point_cloud
from viewpoint_framework.radius_field import RadiusFieldConfig, RADIUS_STRATEGIES
from viewpoint_framework.scene_analysis import (
    CENTER_STRATEGIES,
    GLOBAL_MODE_STRATEGIES,
    MODE_STRATEGIES,
    CenterEstimationConfig,
    ModeAnalysisConfig,
)
from viewpoint_framework.scene_types import to_jsonable
from viewpoint_framework.scene_understanding import (
    RadiusSamplingConfig,
    SceneUnderstandingConfig,
    understand_scene,
)
from viewpoint_framework.util import serve_html
from viewpoint_framework.view_space import (
    BBOX_STRATEGIES,
    ViewSpaceConfig,
    load_legacy_view_limits,
)
from viewpoint_framework.visualize_scene_analysis import visualize_scene_analysis


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze captured camera geometry: scene center, per-camera collection "
            "mode, mode-specific spherical bbox, and directional radius field."
        )
    )

    # ------------------------------------------------------------------
    # IO / preset.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--cameras",
        type=str,
        required=True,
        help="Captured camera JSON in the framework's unified 18-D format.",
    )
    parser.add_argument(
        "--point_cloud",
        type=str,
        default=None,
        help=(
            "Optional PLY. Required by pointcloud_cone/hybrid radius strategies; "
            "also used by the HTML visualization when provided."
        ),
    )
    parser.add_argument(
        "--view_limits",
        type=str,
        default=None,
        help="Historical view_limits.json; required by legacy_view_limits bbox strategy.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="scene_analysis",
    )
    parser.add_argument(
        "--config_json",
        type=str,
        default=None,
        help=(
            "Optional nested strategy config JSON. Precedence: preset < config_json < explicit CLI args."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=("default", "legacy"),
        default="default",
        help="default=improved strategies; legacy=previous camera-only baselines.",
    )

    # ------------------------------------------------------------------
    # Center strategy.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--center_strategy",
        choices=CENTER_STRATEGIES,
        default=None,
    )
    parser.add_argument("--robust_loss", choices=("huber", "tukey"), default=None)
    parser.add_argument("--robust_delta", type=float, default=None)
    parser.add_argument("--center_max_iters", type=int, default=None)
    parser.add_argument("--center_inlier_weight", type=float, default=None)

    # ------------------------------------------------------------------
    # Camera mode analysis.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--mode_strategy",
        choices=MODE_STRATEGIES,
        default=None,
    )
    parser.add_argument(
        "--global_mode_strategy",
        choices=GLOBAL_MODE_STRATEGIES,
        default=None,
    )
    parser.add_argument("--max_alignment_deg", type=float, default=None)
    parser.add_argument("--max_residual_ratio", type=float, default=None)
    parser.add_argument("--dominant_ratio", type=float, default=None)

    # ------------------------------------------------------------------
    # View bbox / spherical frame.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--bbox_strategy",
        choices=BBOX_STRATEGIES,
        default=None,
    )
    parser.add_argument(
        "--world_up",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
    )
    parser.add_argument(
        "--azimuth_zero_reference",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
    )
    parser.add_argument(
        "--extension_mode",
        choices=("fixed", "adaptive"),
        default=None,
    )
    parser.add_argument("--fixed_extension_deg", type=float, default=None)
    parser.add_argument("--min_extension_deg", type=float, default=None)
    parser.add_argument("--max_extension_deg", type=float, default=None)
    parser.add_argument("--adaptive_extension_factor", type=float, default=None)
    parser.add_argument("--radius_extension_ratio", type=float, default=None)

    # ------------------------------------------------------------------
    # Directional radius field.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--radius_strategy",
        choices=RADIUS_STRATEGIES,
        default=None,
    )
    parser.add_argument("--radius_k_neighbors", type=int, default=None)
    parser.add_argument("--radius_min_neighbors", type=int, default=None)
    parser.add_argument("--radius_sigma_deg", type=float, default=None)
    parser.add_argument("--radius_max_support_angle_deg", type=float, default=None)
    parser.add_argument("--cone_half_angle_deg", type=float, default=None)
    parser.add_argument("--legacy_alpha", type=float, default=None)
    parser.add_argument("--legacy_post_scale", type=float, default=None)
    parser.add_argument("--hybrid_geometry_alpha", type=float, default=None)
    parser.add_argument(
        "--hybrid_rule",
        choices=("min", "clamp_high"),
        default=None,
    )
    parser.add_argument(
        "--radius_pointcloud_max_points",
        type=int,
        default=0,
        help="<=0 keeps all points for radius geometry analysis.",
    )
    parser.add_argument("--radius_grid_az_step", type=float, default=None)
    parser.add_argument("--radius_grid_el_step", type=float, default=None)

    # ------------------------------------------------------------------
    # Visualization.
    # ------------------------------------------------------------------
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--camera_stride", type=int, default=4)
    parser.add_argument("--sight_line_stride", type=int, default=4)
    parser.add_argument("--max_points", type=int, default=100000)
    parser.add_argument("--point_size", type=float, default=1.5)
    parser.add_argument("--point_opacity", type=float, default=0.65)

    # HTTP server, aligned with current visualize_cameras.py behavior.
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open_browser", action="store_true")
    parser.add_argument("--no_serve", action="store_true")

    return parser


def _make_base_config(preset: str) -> SceneUnderstandingConfig:
    if preset == "legacy":
        return SceneUnderstandingConfig.legacy_camera_only()
    return SceneUnderstandingConfig.default()


def _override_if_not_none(obj, name: str, value) -> None:
    if value is not None:
        setattr(obj, name, value)


def build_config_from_args(args: argparse.Namespace) -> SceneUnderstandingConfig:
    config = _make_base_config(args.preset)

    if args.config_json:
        config_path = Path(args.config_json).expanduser().resolve()
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        config = SceneUnderstandingConfig.from_dict(config_data)

    _override_if_not_none(config.center, "strategy", args.center_strategy)
    _override_if_not_none(config.center, "robust_loss", args.robust_loss)
    _override_if_not_none(config.center, "robust_delta", args.robust_delta)
    _override_if_not_none(config.center, "max_iters", args.center_max_iters)
    _override_if_not_none(
        config.center,
        "inlier_weight_threshold",
        args.center_inlier_weight,
    )

    _override_if_not_none(config.mode, "strategy", args.mode_strategy)
    _override_if_not_none(
        config.mode,
        "global_strategy",
        args.global_mode_strategy,
    )
    _override_if_not_none(
        config.mode,
        "max_alignment_deg",
        args.max_alignment_deg,
    )
    _override_if_not_none(
        config.mode,
        "max_residual_ratio",
        args.max_residual_ratio,
    )
    _override_if_not_none(config.mode, "dominant_ratio", args.dominant_ratio)

    _override_if_not_none(config.view_space, "strategy", args.bbox_strategy)
    if args.world_up is not None:
        config.view_space.world_up = tuple(float(x) for x in args.world_up)
    if args.azimuth_zero_reference is not None:
        config.view_space.azimuth_zero_reference = tuple(
            float(x) for x in args.azimuth_zero_reference
        )
    _override_if_not_none(
        config.view_space,
        "extension_mode",
        args.extension_mode,
    )
    _override_if_not_none(
        config.view_space,
        "fixed_extension_deg",
        args.fixed_extension_deg,
    )
    _override_if_not_none(
        config.view_space,
        "min_extension_deg",
        args.min_extension_deg,
    )
    _override_if_not_none(
        config.view_space,
        "max_extension_deg",
        args.max_extension_deg,
    )
    _override_if_not_none(
        config.view_space,
        "adaptive_extension_factor",
        args.adaptive_extension_factor,
    )
    _override_if_not_none(
        config.view_space,
        "radius_extension_ratio",
        args.radius_extension_ratio,
    )

    _override_if_not_none(config.radius, "strategy", args.radius_strategy)
    _override_if_not_none(config.radius, "k_neighbors", args.radius_k_neighbors)
    _override_if_not_none(
        config.radius,
        "min_neighbors",
        args.radius_min_neighbors,
    )
    _override_if_not_none(config.radius, "sigma_deg", args.radius_sigma_deg)
    _override_if_not_none(
        config.radius,
        "max_support_angle_deg",
        args.radius_max_support_angle_deg,
    )
    _override_if_not_none(
        config.radius,
        "cone_half_angle_deg",
        args.cone_half_angle_deg,
    )
    _override_if_not_none(config.radius, "legacy_alpha", args.legacy_alpha)
    _override_if_not_none(
        config.radius,
        "legacy_post_scale",
        args.legacy_post_scale,
    )
    _override_if_not_none(
        config.radius,
        "hybrid_geometry_alpha",
        args.hybrid_geometry_alpha,
    )
    _override_if_not_none(config.radius, "hybrid_rule", args.hybrid_rule)

    _override_if_not_none(
        config.radius_sampling,
        "azimuth_step_deg",
        args.radius_grid_az_step,
    )
    _override_if_not_none(
        config.radius_sampling,
        "elevation_step_deg",
        args.radius_grid_el_step,
    )

    return config


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            to_jsonable(payload),
            f,
            indent=2,
            ensure_ascii=False,
        )


def _print_summary(result) -> None:
    profile = result.profile
    fit = profile.center_fit
    summary = profile.mode_summary

    print("==========================================================")
    print("[SceneAnalysis] Done")
    print(f"  center strategy : {fit.strategy}")
    print(f"  center          : {fit.center}")
    print(f"  median residual : {fit.median_residual:.6f}")
    print(f"  condition       : {fit.condition_number:.6g}")
    print(f"  dominant mode   : {summary.dominant_mode.value}")
    print(f"  mode confidence : {summary.dominant_confidence:.4f}")
    print(
        "  camera modes    : "
        f"outside={summary.outside_in_count}, "
        f"inside={summary.inside_out_count}, "
        f"ambiguous={summary.ambiguous_count}, "
        f"outlier={summary.outlier_count}"
    )
    for key, bbox in profile.view_bboxes.items():
        print(
            f"  bbox[{key}]     : az span={bbox.generation_azimuth.span_deg:.2f} deg, "
            f"el={bbox.generation_elevation_deg}, r={bbox.generation_radius}, "
            f"support={bbox.support_camera_count}"
        )
    print("==========================================================")


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    config = build_config_from_args(args)

    if config.view_space.strategy == "legacy_view_limits" and not args.view_limits:
        parser.error("--view_limits is required for --bbox_strategy legacy_view_limits")

    if config.radius.strategy in ("legacy_pointcloud_rule", "pointcloud_cone", "hybrid") and not args.point_cloud:
        parser.error(
            f"--point_cloud is required for radius strategy {config.radius.strategy}"
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras = load_cameras_json(args.cameras)

    legacy_view_limits: Optional[dict] = None
    if args.view_limits:
        legacy_view_limits = load_legacy_view_limits(args.view_limits)

    point_cloud_points = None
    if args.point_cloud and config.radius.strategy in ("legacy_pointcloud_rule", "pointcloud_cone", "hybrid"):
        analysis_cloud = load_ply_point_cloud(
            args.point_cloud,
            max_points=args.radius_pointcloud_max_points,
        )
        point_cloud_points = analysis_cloud.points

    result = understand_scene(
        cameras=cameras,
        config=config,
        point_cloud_points=point_cloud_points,
        legacy_view_limits=legacy_view_limits,
        metadata={
            "camera_json": str(Path(args.cameras).expanduser().resolve()),
            "point_cloud": (
                str(Path(args.point_cloud).expanduser().resolve())
                if args.point_cloud
                else None
            ),
            "view_limits": (
                str(Path(args.view_limits).expanduser().resolve())
                if args.view_limits
                else None
            ),
            "preset": args.preset,
        },
    )

    _print_summary(result)

    profile_path = output_dir / "scene_profile.json"
    relations_path = output_dir / "camera_relations.json"
    config_path = output_dir / "strategy_config.json"

    _write_json(profile_path, result.profile)
    _write_json(relations_path, result.profile.camera_relations)
    _write_json(config_path, result.profile.strategy_config)

    print(f"[SceneAnalysis] scene profile    : {profile_path}")
    print(f"[SceneAnalysis] camera relations : {relations_path}")
    print(f"[SceneAnalysis] strategy config  : {config_path}")

    html_path = None
    if not args.no_visualize:
        html_path = visualize_scene_analysis(
            result=result,
            cameras=cameras,
            output_path=str(output_dir / "index.html"),
            point_cloud_path=args.point_cloud,
            max_points=args.max_points,
            point_size=args.point_size,
            point_opacity=args.point_opacity,
            camera_stride=args.camera_stride,
            sight_line_stride=args.sight_line_stride,
        )
        print(f"[SceneAnalysis] visualization    : {html_path}")

    if html_path is not None and not args.no_serve:
        serve_html(
            html_path=html_path,
            host=args.host,
            port=args.port,
            open_browser=args.open_browser,
        )


if __name__ == "__main__":
    main()
