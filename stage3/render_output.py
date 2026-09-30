#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage-3 output contract and Gaussian RGB rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from viewpoint_framework.utils.cameras import Camera
from viewpoint_framework.utils.camera_output_transform import (
    PortraitOutputMode,
    transform_camera_for_output,
)
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
    gaussian_subset: str = "geometry",
    portrait_output: PortraitOutputMode = "off",
) -> list[str]:
    if gaussian_subset not in ("geometry", "full"):
        raise ValueError("gaussian_subset must be 'geometry' or 'full'")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, camera in enumerate(cameras):
        output_camera, _ = transform_camera_for_output(camera, portrait_output)
        if gaussian_subset == "geometry":
            result = renderer.render_geometry(
                output_camera, max_image_dim=max_image_dim, need_rgb=True,
                need_alpha=True, need_depth=depth_dir is not None)
        else:
            result = renderer.render(
                output_camera, max_image_dim=max_image_dim, need_rgb=True,
                need_depth=depth_dir is not None, include_skybox=True)
        path = output_dir / f"{prefix}_{i:04d}.png"
        _write_rgb_png(path, result.rgb)
        if alpha_dir is not None:
            _write_alpha_png(alpha_dir / f"{prefix}_{i:04d}.png", result.alpha)
        if depth_dir is not None:
            depth_dir.mkdir(parents=True, exist_ok=True)
            np.save(depth_dir / f"{prefix}_{i:04d}.npy",
                    np.asarray(result.depth, dtype=np.float32))
        paths.append(str(path))
    return paths


def build_frame_manifest(
    result: Stage3Result,
    output_cameras: Sequence[Camera] | None = None,
    applied_transforms: Sequence[str] | None = None,
) -> list[dict]:
    """Return a stable frame-to-candidate mapping for backend comparisons."""
    if len(result.selected_candidates) != len(result.selected_cameras):
        raise ValueError(
            "selected_candidates and selected_cameras must have identical lengths")
    if output_cameras is None:
        output_cameras = result.selected_cameras
    if len(output_cameras) != len(result.selected_cameras):
        raise ValueError("output_cameras and selected_cameras must have identical lengths")
    if applied_transforms is not None and len(applied_transforms) != len(output_cameras):
        raise ValueError("applied_transforms and output_cameras must have identical lengths")
    block_lookup = {}
    if result.block_plan is not None:
        for block in result.block_plan.blocks:
            for local_index, candidate in enumerate(block.candidates):
                block_lookup[int(candidate.candidate_id)] = {
                    "block_id": int(block.block_id),
                    "block_local_index": int(local_index),
                    "trunk_index": int(local_index // int(
                        result.block_plan.config.get("trunk_frames", 7))),
                }
    rows = []
    for output_index, (candidate, source_camera, camera) in enumerate(zip(
            result.selected_candidates, result.selected_cameras, output_cameras)):
        row = {
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
        }
        if applied_transforms is not None:
            row["source_camera"] = to_jsonable(source_camera)
            row["output_transform"] = {
                "applied_transform": applied_transforms[output_index]
            }
        if int(candidate.candidate_id) in block_lookup:
            row.update(block_lookup[int(candidate.candidate_id)])
        rows.append(row)
    return rows


def save_stage3_outputs(
    result: Stage3Result,
    renderer: GsplatRenderer,
    output_dir: str,
    *,
    debug_mode: bool,
    render_pano_depths: bool = False,
    geometry_output_contract: bool = False,
    portrait_output: PortraitOutputMode = "off",
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
    pano_blocks_path = root / "pano_blocks.json"

    transformed = [
        transform_camera_for_output(camera, portrait_output)
        for camera in result.selected_cameras
    ]
    output_cameras = [camera for camera, _ in transformed]
    applied_transforms = [transform for _, transform in transformed]
    save_cameras_json(output_cameras, str(pano_cameras_path))
    with open(frame_manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            build_frame_manifest(
                result,
                output_cameras=output_cameras,
                applied_transforms=(
                    applied_transforms if portrait_output != "off" else None
                ),
            ),
            f,
            indent=2,
        )
    if geometry_output_contract:
        render_camera_sequence(renderer, result.selected_cameras, pano_images_dir,
                               prefix="frame", alpha_dir=pano_alphas_dir,
                               depth_dir=pano_depths_dir,
                               portrait_output=portrait_output)
    else:
        # V3.2 retains the full-RGB render path and outputs.
        pano_images_dir.mkdir(parents=True, exist_ok=True)
        for i, camera in enumerate(result.selected_cameras):
            output_camera, _ = transform_camera_for_output(camera, portrait_output)
            rgb = renderer.render_rgb(output_camera, max_image_dim=None)
            _write_rgb_png(pano_images_dir / f"frame_{i:04d}.png", rgb)
    if pano_depths_dir is not None:
        with open(pano_depths_dir / "depth_meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "convention": "camera_z_planar_depth", "dtype": "float32",
                "invalid_value": 0.0, "skybox_included": False,
                "renderer_backend": getattr(renderer, "backend_name", "gsplat"),
            }, f, indent=2)

    if portrait_output != "off":
        output_transform_path = root / "output_transform.json"
        with open(output_transform_path, "w", encoding="utf-8") as f:
            json.dump({
                "portrait_output": portrait_output,
                "scope": "final_pano_outputs_only",
                "reference_alignment": {
                    "traj_refs_index_space": "original_train_cameras",
                    "reference_transform": "none",
                    "note": (
                        "Captured references remain in their original resolution "
                        "and orientation; downstream cross-attention owns their use."
                    ),
                },
                "applied_transforms": applied_transforms,
            }, f, indent=2)

    # Block mode extends the previous one-row contract to one row per block.
    if result.block_plan is None:
        trajectory_refs = [result.reference_result.original_indices]
        trajectory_lengths = [len(result.selected_cameras)]
    else:
        trajectory_refs = [
            block.reference_original_indices for block in result.block_plan.blocks
        ]
        trajectory_lengths = [len(block.candidates) for block in result.block_plan.blocks]
        block_payload = {
            "schema_version": "3.4",
            "block_mode": "content",
            "trunk_frames": int(result.block_plan.config.get("trunk_frames", 7)),
            "portrait_output": portrait_output,
            "input_target_count": (
                len(result.selected_cameras) + len(result.block_plan.dropped_candidate_ids)
            ),
            "output_target_count": len(result.selected_cameras),
            "dropped_candidate_ids": result.block_plan.dropped_candidate_ids,
            "reference_index_space": "train_cameras_original_index",
            "frame_count_policy": result.block_plan.frame_count_policy,
            "config": result.block_plan.config,
            "blocks": [
                {
                    "block_id": int(block.block_id),
                    "start_index": int(block.start_index),
                    "length": len(block.candidates),
                    "num_trunks": len(block.candidates) // int(
                        result.block_plan.config.get("trunk_frames", 7)),
                    "candidate_ids_in_output_order": [
                        int(candidate.candidate_id) for candidate in block.candidates
                    ],
                    "reference_original_indices": block.reference_original_indices,
                    "descriptor_strategy": block.descriptor_strategy,
                    "coverage_metrics": to_jsonable(block.coverage_metrics),
                    "ordering_metrics": to_jsonable(block.ordering_metrics),
                }
                for block in result.block_plan.blocks
            ],
        }
        with open(pano_blocks_path, "w", encoding="utf-8") as f:
            json.dump(block_payload, f, indent=2)

    with open(traj_refs_path, "w", encoding="utf-8") as f:
        json.dump(trajectory_refs, f, separators=(",", ":"))
    with open(traj_lens_path, "w", encoding="utf-8") as f:
        json.dump(trajectory_lengths, f, separators=(",", ":"))

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
    if portrait_output != "off":
        paths["output_transform"] = str(output_transform_path)
    if result.block_plan is not None:
        paths["pano_blocks"] = str(pano_blocks_path)

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
