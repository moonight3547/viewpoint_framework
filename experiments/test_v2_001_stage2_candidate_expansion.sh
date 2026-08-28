#!/usr/bin/env bash
set -euo pipefail

# V1/V2 end-to-end comparison. Only server paths below (or positional args)
# need to be changed. Outputs are isolated so neither experiment is overwritten.
if [[ $# -ne 5 ]]; then
    echo "Usage: $0 TRAIN_CAMERAS POINT_CLOUD GAUSSIAN_PLY SELECT_VIEW_DIR OUTPUT_ROOT"
    exit 2
fi

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PARENT_DIR="$(dirname "$REPO_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python}"

TRAIN_CAMERAS="$1"
POINT_CLOUD="$2"
GAUSSIAN_PLY="$3"
SELECT_VIEW_DIR="$4"
OUTPUT_ROOT="$5"

resolve_path() {
    "$PYTHON_BIN" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$1"
}

TRAIN_CAMERAS="$(resolve_path "$TRAIN_CAMERAS")"
POINT_CLOUD="$(resolve_path "$POINT_CLOUD")"
GAUSSIAN_PLY="$(resolve_path "$GAUSSIAN_PLY")"
SELECT_VIEW_DIR="$(resolve_path "$SELECT_VIEW_DIR")"
OUTPUT_ROOT="$(resolve_path "$OUTPUT_ROOT")"

mkdir -p "$OUTPUT_ROOT/logs"
cd "$PARENT_DIR"

run_case() {
    local version="$1"
    local scene_config="$2"
    local pose_config="$3"
    local output_dir="$OUTPUT_ROOT/$version"
    local log_path="$OUTPUT_ROOT/logs/$version.log"

    echo "[$version] output=$output_dir log=$log_path"
    set +e
    "$PYTHON_BIN" -m viewpoint_framework.run_pipeline \
        --cameras "$TRAIN_CAMERAS" \
        --point_cloud "$POINT_CLOUD" \
        --gaussian_ply "$GAUSSIAN_PLY" \
        --select_view_dir "$SELECT_VIEW_DIR" \
        --output_dir "$output_dir" \
        --scene_config_json "$scene_config" \
        --pose_config_json "$pose_config" \
        --stage3_config_json "$REPO_DIR/configs/default_stage3.json" \
        --num_panos 49 \
        --num_refs 12 \
        --device "${DEVICE:-auto}" \
        2>&1 | tee "$log_path"
    local run_status="${PIPESTATUS[0]}"
    set -e
    echo "[$version] exit_status=$run_status"
    return "$run_status"
}

v1_status=0
run_case \
    "v1_baseline" \
    "$REPO_DIR/configs/default_scene_understanding.json" \
    "$REPO_DIR/configs/default_pose_generation.json" || v1_status=$?

v2_status=0
run_case \
    "v2_stage2" \
    "$REPO_DIR/configs/v2_scene_understanding.json" \
    "$REPO_DIR/configs/v2_pose_generation.json" || v2_status=$?

echo "Comparison complete: $OUTPUT_ROOT (v1=$v1_status, v2=$v2_status)"
exit "$v2_status"
