"""Compare two fixed-camera renderer diagnostic outputs frame by frame."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from viewpoint_framework.cameras_util import load_cameras_json


def _array_metrics(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        return {"shape_match": False,
                "left_shape": list(left.shape), "right_shape": list(right.shape)}
    diff = left - right
    mse = float(np.mean(diff * diff))
    return {
        "shape_match": True,
        "exact": bool(mse == 0.),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(mse)),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.,
        "psnr": None if mse == 0. else float(10. * np.log10(1. / mse)),
    }


def _rotation_delta_deg(left, right):
    relative = np.asarray(left).T @ np.asarray(right)
    cosine = np.clip((float(np.trace(relative)) - 1.) / 2., -1., 1.)
    return float(np.degrees(np.arccos(cosine)))


def _depth_metrics(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        return {"shape_match": False,
                "left_shape": list(left.shape), "right_shape": list(right.shape)}
    left_valid = np.isfinite(left) & (left > 0.)
    right_valid = np.isfinite(right) & (right > 0.)
    union = left_valid | right_valid
    both = left_valid & right_valid
    result = {
        "shape_match": True,
        "left_valid_ratio": float(np.mean(left_valid)),
        "right_valid_ratio": float(np.mean(right_valid)),
        "valid_iou": (float(np.count_nonzero(both) / np.count_nonzero(union))
                      if np.any(union) else 1.),
        "overlap_pixels": int(np.count_nonzero(both)),
    }
    if np.any(both):
        scale = np.maximum(np.maximum(np.abs(left[both]), np.abs(right[both])), 1e-8)
        result.update({
            "mae": float(np.mean(np.abs(left[both] - right[both]))),
            "mean_relative_abs_error": float(
                np.mean(np.abs(left[both] - right[both]) / scale)),
            "median_depth_ratio": float(np.median(left[both] / right[both])),
        })
    return result


def _summary(rows, section, field):
    values = [row[section][field] for row in rows
              if row.get(section, {}).get(field) is not None]
    if not values:
        return None
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "max": float(np.max(values))}


def compare_outputs(left_root, right_root):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Sequence comparison requires opencv-python (cv2).") from exc
    left_root, right_root = Path(left_root), Path(right_root)
    left_cameras = load_cameras_json(left_root / "cameras.json")
    right_cameras = load_cameras_json(right_root / "cameras.json")
    count = min(len(left_cameras), len(right_cameras))
    rows = []
    for index in range(count):
        left_camera, right_camera = left_cameras[index], right_cameras[index]
        row = {
            "frame": index,
            "camera": {
                "position_delta": float(np.linalg.norm(
                    left_camera.position - right_camera.position)),
                "rotation_delta_deg": _rotation_delta_deg(
                    left_camera.rotation_c2w, right_camera.rotation_c2w),
                "intrinsic_max_abs": float(np.max(np.abs(np.array([
                    left_camera.fx-right_camera.fx, left_camera.fy-right_camera.fy,
                    left_camera.cx-right_camera.cx, left_camera.cy-right_camera.cy,
                ])))),
            },
        }
        for folder, key, grayscale in (
                ("images", "rgb", False), ("alphas", "alpha", True)):
            left_path = left_root / folder / f"frame_{index:04d}.png"
            right_path = right_root / folder / f"frame_{index:04d}.png"
            if left_path.is_file() and right_path.is_file():
                flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
                left = cv2.imread(str(left_path), flag)
                right = cv2.imread(str(right_path), flag)
                if left is None or right is None:
                    raise IOError(f"Failed to read comparison images for frame {index}")
                left = left.astype(np.float32) / 255.
                right = right.astype(np.float32) / 255.
                row[key] = _array_metrics(left, right)
        left_depth = left_root / "depths" / f"frame_{index:04d}.npy"
        right_depth = right_root / "depths" / f"frame_{index:04d}.npy"
        if left_depth.is_file() and right_depth.is_file():
            row["depth"] = _depth_metrics(np.load(left_depth), np.load(right_depth))
        rows.append(row)
    return {
        "left_frame_count": len(left_cameras),
        "right_frame_count": len(right_cameras),
        "compared_frame_count": count,
        "camera_sequence_aligned": bool(
            len(left_cameras) == len(right_cameras)
            and all(row["camera"]["position_delta"] <= 1e-7
                    and row["camera"]["rotation_delta_deg"] <= 1e-5
                    and row["camera"]["intrinsic_max_abs"] <= 1e-7
                    for row in rows)),
        "summary": {
            "rgb_mae": _summary(rows, "rgb", "mae"),
            "alpha_mae": _summary(rows, "alpha", "mae"),
            "depth_valid_iou": _summary(rows, "depth", "valid_iou"),
            "depth_relative_abs_error": _summary(
                rows, "depth", "mean_relative_abs_error"),
        },
        "frames": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = compare_outputs(args.left, args.right)
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print("[GS:SEQUENCE_COMPARE] " + json.dumps(report["summary"]))
    print(f"[GS:SEQUENCE_COMPARE] cameras_aligned={report['camera_sequence_aligned']}")


if __name__ == "__main__":
    main()
