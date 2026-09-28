"""Render one fixed camera sequence with one backend for parity diagnosis.

Run this command twice with the same camera JSON and different ``--backend``
values.  Keeping camera poses fixed separates renderer disagreement from
Stage-2 placement changes caused by depth probing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from viewpoint_framework.cameras_util import load_cameras_json
from viewpoint_framework.gs_renderer import GaussianRendererConfig
from viewpoint_framework.pose_generation import PoseGenerationConfig, save_cameras_json
from viewpoint_framework.renderer import create_renderer
from viewpoint_framework.renderer.scene_loader import activate_scales
from viewpoint_framework.stage3.render_output import render_camera_sequence


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Render identical cameras with gsplat or gs_render")
    parser.add_argument("--backend", required=True, choices=("gsplat", "gs_render"))
    parser.add_argument("--gaussian-ply", required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--pose-config-json", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-image-dim", type=int, default=None)
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--render-depths", action="store_true")
    parser.add_argument(
        "--gaussian-subset", choices=("geometry", "full"), default="geometry",
        help="Render the skybox-filtered geometry or the complete canonical PLY.")
    return parser


def _renderer_config(path, near_plane):
    if path:
        with open(Path(path).expanduser().resolve(), "r", encoding="utf-8") as handle:
            pose = PoseGenerationConfig.from_dict(json.load(handle))
    else:
        pose = PoseGenerationConfig()
    return GaussianRendererConfig(
        scale_activation="exp",
        opacity_activation="sigmoid",
        max_sh_degree=None,
        background=tuple(pose.renderer.background),
        skybox=pose.skybox,
        near_plane=float(near_plane),
    )


def _projection_diagnostics(renderer, cameras, subset, near_plane):
    """Describe canonical Gaussians near each fixed camera before rasterization."""
    data = renderer.scene_data
    mask = (data.geometry_mask if subset == "geometry"
            else np.ones(len(data.means), dtype=bool))
    means = np.asarray(data.means[mask], dtype=np.float64)
    scales = activate_scales(
        data.scales[mask], renderer.config.scale_activation).astype(np.float64)
    max_scale = np.max(scales, axis=1)
    rows = []
    for index, camera in enumerate(cameras):
        camera_xyz = means @ camera.w2c[:3, :3].T + camera.w2c[:3, 3]
        z = camera_xyz[:, 2]
        front = np.isfinite(z) & (z > 0.0)
        near_center = front & (z <= float(near_plane))
        # Conservative 3-sigma sphere test. It deliberately ignores quaternion
        # orientation and therefore cannot miss a potentially near-camera splat.
        near_support = front & (z - 3.0 * max_scale <= float(near_plane))
        positive = z[front]
        rows.append({
            "frame": index,
            "gaussian_count": int(len(z)),
            "behind_or_on_camera_count": int(np.count_nonzero(z <= 0.0)),
            "front_count": int(np.count_nonzero(front)),
            "center_at_or_inside_near_count": int(np.count_nonzero(near_center)),
            "three_sigma_reaches_near_count": int(np.count_nonzero(near_support)),
            "front_z_min": None if not len(positive) else float(np.min(positive)),
            "front_z_p01": None if not len(positive) else float(np.quantile(positive, .01)),
            "front_z_median": None if not len(positive) else float(np.median(positive)),
        })
    return rows


def main():
    args = build_argparser().parse_args()
    root = Path(args.output_dir).expanduser().resolve()
    image_dir = root / "images"
    alpha_dir = root / "alphas"
    depth_dir = root / "depths" if args.render_depths else None
    root.mkdir(parents=True, exist_ok=True)

    cameras = load_cameras_json(args.cameras)
    renderer = create_renderer(
        args.gaussian_ply,
        backend=args.backend,
        config=_renderer_config(args.pose_config_json, args.near_plane),
        device=args.device,
    )
    paths = render_camera_sequence(
        renderer,
        cameras,
        image_dir,
        alpha_dir=alpha_dir,
        depth_dir=depth_dir,
        max_image_dim=args.max_image_dim,
        gaussian_subset=args.gaussian_subset,
    )
    save_cameras_json(cameras, str(root / "cameras.json"))
    metadata = {
        "backend": args.backend,
        "backend_version": getattr(renderer, "backend_version", "unknown"),
        "source_cameras": str(Path(args.cameras).expanduser().resolve()),
        "source_gaussian_ply": str(Path(args.gaussian_ply).expanduser().resolve()),
        "frame_count": len(cameras),
        "max_image_dim": args.max_image_dim,
        "near_plane": args.near_plane,
        "gaussian_subset": args.gaussian_subset,
        "geometry_only": args.gaussian_subset == "geometry",
        "canonical_gaussian_count": int(len(renderer.scene_data.means)),
        "rendered_gaussian_count": int(
            np.count_nonzero(renderer.scene_data.geometry_mask)
            if args.gaussian_subset == "geometry"
            else len(renderer.scene_data.means)),
        "skybox_gaussian_count": int(np.count_nonzero(
            renderer.scene_data.skybox_mask)),
        "depth_rendered": bool(args.render_depths),
        "depth_convention": "camera_z_planar_depth" if args.render_depths else None,
        "near_plane_consumed_by_backend": args.backend == "gsplat",
        "projection_diagnostics": _projection_diagnostics(
            renderer, cameras, args.gaussian_subset, args.near_plane),
        "images": paths,
    }
    with open(root / "diagnostic_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(
        f"[GS:SEQUENCE_DIAGNOSTIC] backend={args.backend} "
        f"frames={len(cameras)} output={root}")


if __name__ == "__main__":
    main()
