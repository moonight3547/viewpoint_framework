#!/usr/bin/env bash
set -euo pipefail

# This script lives in the viewpoint_framework package directory.  It changes to
# the parent directory so `python -m viewpoint_framework.run_pipeline` resolves
# exactly like the existing project commands.

if [[ $# -lt 5 ]]; then
    echo "Usage: $0 TRAIN_CAMERAS POINT_CLOUD GAUSSIAN_PLY SELECT_VIEW_DIR OUTPUT_DIR"
    exit 2
fi

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$REPO_DIR")"

TRAIN_CAMERAS="$1"
POINT_CLOUD="$2"
GAUSSIAN_PLY="$3"
SELECT_VIEW_DIR="$4"
OUTPUT_DIR="$5"

# Resolve path-like positional arguments before changing working directory.
resolve_path() {
    "$PYTHON_BIN" - "$1" <<'PY'
import os, sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
}

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_CAMERAS="$(resolve_path "$TRAIN_CAMERAS")"
POINT_CLOUD="$(resolve_path "$POINT_CLOUD")"
GAUSSIAN_PLY="$(resolve_path "$GAUSSIAN_PLY")"
SELECT_VIEW_DIR="$(resolve_path "$SELECT_VIEW_DIR")"
OUTPUT_DIR="$(resolve_path "$OUTPUT_DIR")"

NUM_PANOS="${NUM_PANOS:-}"
NUM_REFS="${NUM_REFS:-}"
DEBUG_MODE="${DEBUG_MODE:-0}"
DEVICE="${DEVICE:-auto}"

SCENE_CONFIG="${SCENE_CONFIG:-$REPO_DIR/configs/stage1_scene_understanding.json}"
POSE_CONFIG="${POSE_CONFIG:-$REPO_DIR/configs/v3_3_pose_generation.json}"
STAGE3_CONFIG="${STAGE3_CONFIG:-$REPO_DIR/configs/stage3_v3_3.json}"

MODE="${MODE:-auto}"
GRID_GAP="${GRID_GAP:-}"
SELECTION_STRATEGY="${SELECTION_STRATEGY:-}"
SELECTION_REFERENCE="${SELECTION_REFERENCE:-}"
INFORMATION_GAIN="${INFORMATION_GAIN:-}"
HOLE_STRATEGY="${HOLE_STRATEGY:-}"
REFERENCE_STRATEGY="${REFERENCE_STRATEGY:-}"
ORDERING_STRATEGY="${ORDERING_STRATEGY:-}"
PORTRAIT_OUTPUT="${PORTRAIT_OUTPUT:-}"
BLOCK_MODE="${BLOCK_MODE:-}"

cd "$PARENT_DIR"

CMD=(
    "$PYTHON_BIN" -m viewpoint_framework.run_pipeline
    --cameras "$TRAIN_CAMERAS"
    --point_cloud "$POINT_CLOUD"
    --gaussian_ply "$GAUSSIAN_PLY"
    --select_view_dir "$SELECT_VIEW_DIR"
    --output_dir "$OUTPUT_DIR"
    --scene_config_json "$SCENE_CONFIG"
    --pose_config_json "$POSE_CONFIG"
    --stage3_config_json "$STAGE3_CONFIG"
    --device "$DEVICE"
    --mode "$MODE"
)

[[ -n "$NUM_PANOS" ]] && CMD+=(--num_panos "$NUM_PANOS")
[[ -n "$NUM_REFS" ]] && CMD+=(--num_refs "$NUM_REFS")
[[ -n "$GRID_GAP" ]] && CMD+=(--grid_gap "$GRID_GAP")
[[ -n "$SELECTION_STRATEGY" ]] && CMD+=(--selection-strategy "$SELECTION_STRATEGY")
[[ -n "$SELECTION_REFERENCE" ]] && CMD+=(--selection-reference "$SELECTION_REFERENCE")
[[ -n "$INFORMATION_GAIN" ]] && CMD+=(--information-gain "$INFORMATION_GAIN")
[[ -n "$HOLE_STRATEGY" ]] && CMD+=(--hole-strategy "$HOLE_STRATEGY")
[[ -n "$REFERENCE_STRATEGY" ]] && CMD+=(--reference-strategy "$REFERENCE_STRATEGY")
[[ -n "$ORDERING_STRATEGY" ]] && CMD+=(--ordering-strategy "$ORDERING_STRATEGY")
[[ -n "$PORTRAIT_OUTPUT" ]] && CMD+=(--portrait-output "$PORTRAIT_OUTPUT")
[[ -n "$BLOCK_MODE" ]] && CMD+=(--block-mode "$BLOCK_MODE")

if [[ "${ALL_GENERATED_FRAMES:-0}" == "1" || "${ALL_GENERATED_FRAMES:-}" == "true" ]]; then
    CMD+=(--all-generated-frames)
fi

if [[ "$DEBUG_MODE" == "1" || "$DEBUG_MODE" == "true" || "$DEBUG_MODE" == "TRUE" ]]; then
    CMD+=(--debug-mode)
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARRAY=( $EXTRA_ARGS )
    CMD+=("${EXTRA_ARRAY[@]}")
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
