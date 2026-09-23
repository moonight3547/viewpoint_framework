#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage-3 output contract and Gaussian RGB rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from viewpoint_framework.cameras_util import Camera
from viewpoint_framework.gs_renderer import GsplatRenderer
from viewpoint_framework.pose_generation import save_cameras_json
from viewpoint_framework.stage3.types import SelectionCandidate, Stage3Result, to_jsonable


def _write_rgb_png(path: Path, rgb: np.ndarray) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Rendering PNG outputs requires opencv-python (cv2).") from exc
    rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    image = (rgb * 255.0 + 0.5).astype(np.uint8)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), bgr):
        raise IOError(f"Failed to write image: {path}")


def _write_alpha_png(path: Path, alpha: np.ndarray) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Rendering PNG outputs requires opencv-python (cv2).") from exc
    image = (np.clip(np.asarray(alpha, dtype=np.float32), 0., 1.)*255.+.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Failed to write alpha: {path}")


def render_camera_sequence(
    renderer: GsplatRenderer,
    cameras: Sequence[Camera],
    output_dir: Path,
    *,
    prefix: str = "frame",
    alpha_dir: Path | None = None,
    depth_dir: Path | None = None,
    max_image_dim: int | None = None,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, camera in enumerate(cameras):
        result = renderer.render_geometry(
            camera, max_image_dim=max_image_dim, need_rgb=True, need_alpha=True,
            need_depth=depth_dir is not None)
        rgb = result.rgb
        path = output_dir / f"{prefix}_{i:04d}.png"
        _write_rgb_png(path, rgb)
        if alpha_dir is not None:
            _write_alpha_png(alpha_dir / f"{prefix}_{i:04d}.png", result.alpha)
        if depth_dir is not None:
            depth_dir.mkdir(parents=True, exist_ok=True)
            np.save(depth_dir / f"{prefix}_{i:04d}.npy",
                    np.asarray(result.depth, dtype=np.float32))
        paths.append(str(path))
    return paths


def build_frame_manifest(result: Stage3Result) -> list[dict]:
    """Return a stable frame-to-candidate mapping for backend comparisons."""
    if len(result.selected_candidates) != len(result.selected_cameras):
        raise ValueError(
            "selected_candidates and selected_cameras must have identical lengths")
    rows = []
    for output_index, (candidate, camera) in enumerate(zip(
            result.selected_candidates, result.selected_cameras)):
        rows.append({
            "output_index": output_index,
            "image": f"frame_{output_index:04d}.png",
            "candidate_id": int(candidate.candidate_id),
            "grid_id": None if candidate.grid_id is None else int(candidate.grid_id),
            "row": None if candidate.row is None else int(candidate.row),
            "col": None if candidate.col is None else int(candidate.col),
            "azimuth_deg": candidate.azimuth_deg,
            "elevation_deg": candidate.elevation_deg,
            "signed_radius": candidate.signed_radius,
            "camera_index": int(camera.index),
            "camera": to_jsonable(camera),
        })
    return rows


def save_stage3_outputs(
    result: Stage3Result,
    renderer: GsplatRenderer,
    output_dir: str,
    *,
    debug_mode: bool,
    render_pano_depths: bool = False,
    geometry_output_contract: bool = False,
    stage2_grid_candidates: Sequence[SelectionCandidate],
) -> Dict[str, str]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    pano_cameras_path = root / "pano_cameras.json"
    frame_manifest_path = root / "pano_frame_manifest.json"
    pano_images_dir = root / "pano_images"
    pano_alphas_dir = root / "pano_alphas" if geometry_output_contract else None
    pano_depths_dir = (root / "pano_depths"
                       if geometry_output_contract and render_pano_depths else None)
    traj_refs_path = root / "traj_refs.json"
    traj_lens_path = root / "traj_lens.json"

    save_cameras_json(result.selected_cameras, str(pano_cameras_path))
    with open(frame_manifest_path, "w", encoding="utf-8") as f:
        json.dump(build_frame_manifest(result), f, indent=2)
    if geometry_output_contract:
        render_camera_sequence(renderer, result.selected_cameras, pano_images_dir,
                               prefix="frame", alpha_dir=pano_alphas_dir,
                               depth_dir=pano_depths_dir)
    else:
        # V2/V3.0-V3.2 retain the original full-RGB render path and outputs.
        pano_images_dir.mkdir(parents=True, exist_ok=True)
        for i, camera in enumerate(result.selected_cameras):
            _write_rgb_png(pano_images_dir / f"frame_{i:04d}.png",
                           renderer.render_rgb(camera, max_image_dim=None))
    if pano_depths_dir is not None:
        with open(pano_depths_dir / "depth_meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "convention": "camera_z_planar_depth", "dtype": "float32",
                "invalid_value": 0.0, "skybox_included": False,
                "renderer_backend": getattr(renderer, "backend_name", "gsplat"),
            }, f, indent=2)

    # Keep the previous nested contract: [[idx0, idx1, ...]].
    with open(traj_refs_path, "w", encoding="utf-8") as f:
        json.dump([result.reference_result.original_indices], f, separators=(",", ":"))
    with open(traj_lens_path, "w", encoding="utf-8") as f:
        json.dump([len(result.selected_cameras)], f, separators=(",", ":"))

    paths = {
        "pano_cameras": str(pano_cameras_path),
        "pano_frame_manifest": str(frame_manifest_path),
        "pano_images": str(pano_images_dir),
        "traj_refs": str(traj_refs_path),
        "traj_lens": str(traj_lens_path),
    }
    if pano_alphas_dir is not None:
        paths["pano_alphas"] = str(pano_alphas_dir)
    if pano_depths_dir is not None:
        paths["pano_depths"] = str(pano_depths_dir)

    if debug_mode:
        debug_dir = root / "debug"
        candidate_dir = debug_dir / "candidate_images"
        debug_dir.mkdir(parents=True, exist_ok=True)
        candidate_dir.mkdir(parents=True, exist_ok=True)

        # Requirement: every Stage-2 valid candidate RGB is materialized only in
        # debug mode.  Full resolution is intentional for visual verification.
        for candidate in stage2_grid_candidates:
            rgb = renderer.render_rgb(candidate.camera, max_image_dim=None)
            grid = candidate.grid_id if candidate.grid_id is not None else -1
            path = candidate_dir / f"candidate_{candidate.candidate_id:04d}_grid_{grid:04d}.png"
            _write_rgb_png(path, rgb)

        with open(debug_dir / "stage3_metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "all_candidates": to_jsonable(result.all_candidates),
                    "selected_candidates": to_jsonable(result.selected_candidates),
                    "holes": to_jsonable(result.holes),
                    "hole_views": to_jsonable(result.hole_views),
                    "reference_selection": to_jsonable(result.reference_result),
                    "debug": to_jsonable(result.debug),
                },
                f,
                indent=2,
            )
        paths["debug"] = str(debug_dir)

    return paths


if __name__ == "__main__":
    print("stage3.render_output: import OK; runtime rendering requires a GsplatRenderer.")


__all__ = ["build_frame_manifest", "render_camera_sequence", "save_stage3_outputs"]
