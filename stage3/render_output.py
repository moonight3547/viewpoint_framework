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


def render_camera_sequence(
    renderer: GsplatRenderer,
    cameras: Sequence[Camera],
    output_dir: Path,
    *,
    prefix: str = "frame",
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, camera in enumerate(cameras):
        rgb = renderer.render_rgb(camera, max_image_dim=None)
        path = output_dir / f"{prefix}_{i:04d}.png"
        _write_rgb_png(path, rgb)
        paths.append(str(path))
    return paths


def save_stage3_outputs(
    result: Stage3Result,
    renderer: GsplatRenderer,
    output_dir: str,
    *,
    debug_mode: bool,
    stage2_grid_candidates: Sequence[SelectionCandidate],
) -> Dict[str, str]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    pano_cameras_path = root / "pano_cameras.json"
    pano_images_dir = root / "pano_images"
    traj_refs_path = root / "traj_refs.json"
    traj_lens_path = root / "traj_lens.json"

    save_cameras_json(result.selected_cameras, str(pano_cameras_path))
    render_camera_sequence(renderer, result.selected_cameras, pano_images_dir, prefix="frame")

    # Keep the previous nested contract: [[idx0, idx1, ...]].
    with open(traj_refs_path, "w", encoding="utf-8") as f:
        json.dump([result.reference_result.original_indices], f, separators=(",", ":"))
    with open(traj_lens_path, "w", encoding="utf-8") as f:
        json.dump([len(result.selected_cameras)], f, separators=(",", ":"))

    paths = {
        "pano_cameras": str(pano_cameras_path),
        "pano_images": str(pano_images_dir),
        "traj_refs": str(traj_refs_path),
        "traj_lens": str(traj_lens_path),
    }

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
