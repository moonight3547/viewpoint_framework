"""Independent-process gs_render contract isolation for V3.3.

Each rasterization combination runs in a fresh process so a CUDA illegal access
cannot poison any later diagnostic case.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from viewpoint_framework.cameras_util import load_cameras_json
from viewpoint_framework.gs_renderer import GaussianRendererConfig
from viewpoint_framework.pose_generation import PoseGenerationConfig
from viewpoint_framework.renderer.contract_diagnostics import print_contract
from viewpoint_framework.renderer.gs_render_backend import GsRenderRenderer


CASES = (
    "full_render",
    "geometry_render",
    "geometry_distance",
    "probe_render",
    "probe_distance",
)


def build_argparser():
    parser = argparse.ArgumentParser(description="Isolate gs_render CUDA contract failures")
    parser.add_argument("--gaussian-ply", required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--pose-config-json", default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--probe-max-dim", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-use-bucket-false-ab", action="store_true",
                        help="Diagnostic only; never changes production configuration")
    parser.add_argument("--_worker-case", choices=CASES, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_use-bucket", choices=("true", "false"), default="true",
                        help=argparse.SUPPRESS)
    return parser


def _load_configs(path):
    config_path = (
        Path(path).expanduser().resolve()
        if path
        else Path(__file__).resolve().parents[1] / "configs" / "v3_3_pose_generation.json"
    )
    with open(config_path, "r", encoding="utf-8") as handle:
        pose = PoseGenerationConfig.from_dict(json.load(handle))
    renderer = GaussianRendererConfig(
        scale_activation="exp", opacity_activation="sigmoid",
        max_sh_degree=None, background=tuple(pose.renderer.background),
        skybox=pose.skybox,
    )
    return pose, renderer


def _case_spec(name):
    return {
        "full_render": (True, False, False),
        "geometry_render": (False, False, False),
        "geometry_distance": (False, True, False),
        "probe_render": (False, False, True),
        "probe_distance": (False, True, True),
    }[name]


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=str)


def _worker(args):
    import importlib
    import torch
    from viewpoint_framework.renderer.contract_diagnostics import tensor_summary

    case = args._worker_case
    include_skybox, with_distance, resized = _case_spec(case)
    use_bucket = args._use_bucket == "true"
    suffix = "__bucket_false" if not use_bucket else ""
    output = Path(args.output_dir).expanduser().resolve() / f"{case}{suffix}"
    output.mkdir(parents=True, exist_ok=True)

    module = importlib.import_module("gs_render")
    print(
        f"[GS:SMOKE_BACKEND] backend=gs_render "
        f"version={getattr(module, '__version__', 'unknown')} device={args.device}",
        flush=True,
    )
    _, renderer_config = _load_configs(args.pose_config_json)
    renderer = GsRenderRenderer(
        args.gaussian_ply, module, renderer_config, device=args.device)
    cameras = load_cameras_json(args.cameras)
    if not 0 <= args.camera_index < len(cameras):
        raise IndexError(f"camera-index {args.camera_index} outside [0,{len(cameras)-1}]")
    camera = cameras[args.camera_index]
    max_dim = args.probe_max_dim if resized else None
    width, height = renderer.resolve_size(camera, max_dim)
    camera_data = renderer._camera_data(camera, width, height)
    gaussian_data = renderer._full_data if include_skybox else renderer._geometry_data
    config_data = module.GsRenderConfigData(
        degree=int(renderer.scene_data.sh_degree),
        bg_color=renderer._tensor(renderer_config.background),
        render_depth=False, render_normal=with_distance,
        clamp_color_min=False, return_abs_grad=False, use_bucket=use_bucket)

    contract = print_contract(case, gaussian_data, camera_data, config_data)
    contract.update({
        "case": case, "include_skybox": include_skybox,
        "with_distance": with_distance, "resized": resized,
        "use_bucket": use_bucket,
        "camera_source": {"index": args.camera_index,
                          "original_width": camera.width, "original_height": camera.height,
                          "render_width": width, "render_height": height},
        "loader_policy": {
            "scale_activation": renderer_config.scale_activation,
            "opacity_activation": renderer_config.opacity_activation,
            "tensor_representation": "activated",
            "degree": int(renderer.scene_data.sh_degree),
            "full_count": int(len(renderer.scene_data.means)),
            "geometry_count": int(renderer.scene_data.geometry_mask.sum()),
            "skybox_count": int(renderer.scene_data.skybox_mask.sum()),
        },
    })
    _write_json(output / "pre_raster_contract.json", contract)
    if renderer.device.startswith("cuda"):
        torch.cuda.synchronize()
    print(f"[GS:SMOKE_PRE_RASTER_OK] case={case} use_bucket={use_bucket}", flush=True)

    with torch.no_grad():
        if with_distance:
            results = module.GsRenderer.render_with_distance(
                gaussian_data, camera_data, config_data)
        else:
            results = module.GsRenderer.render(
                gaussian_data, camera_data, config_data)
    if renderer.device.startswith("cuda"):
        torch.cuda.synchronize()
    result_contract = {
        "case": case, "result_count": len(results),
        "results": [tensor_summary(value) for value in results],
    }
    _write_json(output / "post_raster_contract.json", result_contract)
    print("[GS:SMOKE_RASTER_OK] " + json.dumps(result_contract, default=str), flush=True)
    return 0


def _child_command(args, case, use_bucket):
    command = [
        sys.executable, "-m", "viewpoint_framework.renderer.gs_render_contract_smoke",
        "--gaussian-ply", str(Path(args.gaussian_ply).expanduser().resolve()),
        "--cameras", str(Path(args.cameras).expanduser().resolve()),
        "--camera-index", str(args.camera_index),
        "--probe-max-dim", str(args.probe_max_dim),
        "--device", args.device,
        "--output-dir", str(Path(args.output_dir).expanduser().resolve()),
        "--_worker-case", case,
        "--_use-bucket", "true" if use_bucket else "false",
    ]
    if args.pose_config_json:
        command.extend(["--pose-config-json",
                        str(Path(args.pose_config_json).expanduser().resolve())])
    return command


def _supervisor(args):
    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    runs = [(case, True) for case in CASES]
    if args.include_use_bucket_false_ab:
        runs.extend((case, False) for case in ("geometry_distance", "probe_distance"))
    summary = []
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    for index, (case, use_bucket) in enumerate(runs, 1):
        label = case + ("__bucket_false" if not use_bucket else "")
        print(f"[GS:SMOKE_CASE] {index}/{len(runs)} {label}", flush=True)
        completed = subprocess.run(
            _child_command(args, case, use_bucket), env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", check=False)
        log_path = root / f"{label}.log"
        log_path.write_text(completed.stdout, encoding="utf-8")
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        row = {"case": case, "use_bucket": use_bucket,
               "returncode": completed.returncode,
               "status": "PASS" if completed.returncode == 0 else "FAIL",
               "log": str(log_path)}
        summary.append(row)
        print(f"[GS:SMOKE_CASE_RESULT] {label} {row['status']} rc={completed.returncode}",
              flush=True)
    core = summary[:len(CASES)]
    first_failure = next((row["case"] for row in core if row["returncode"]), None)
    payload = {"cases": summary, "first_core_failure": first_failure,
               "all_core_passed": first_failure is None}
    _write_json(root / "summary.json", payload)
    print("[GS:SMOKE_SUMMARY] " + json.dumps(payload, ensure_ascii=False), flush=True)
    return 0 if first_failure is None else 1


def main():
    args = build_argparser().parse_args()
    if args._worker_case:
        try:
            return _worker(args)
        except Exception:
            traceback.print_exc()
            return 2
    return _supervisor(args)


if __name__ == "__main__":
    raise SystemExit(main())
