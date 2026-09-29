# Viewpoint Framework V3.4

V3.4 builds on the frozen V3.3 expanded-view candidate set. It does not add a
new pose-generation version and does not change Stage 1 or Stage 2 geometry,
depth safety, skybox handling, or renderer activation contracts.

The first implemented V3.4 experiment is optional portrait output. Content
blocks and block-local references remain a later experiment and are not
silently approximated by sequential slicing in this version.

## Version boundaries

- V3.2 pose generation remains directly selectable with
  `configs/v3_2_pose_generation.json`.
- V3.3 pose generation remains directly selectable with
  `configs/v3_3_pose_generation.json`. Its default is the V3.3 angular grid
  with validated V3.2 placement; native V3.3 placement remains available via
  `v33_placement_strategy=v3_3`.
- V3.4 is currently a Stage 3 output experiment. The portrait CLI override can
  be combined with either version's own Stage 3 config. The bundled
  `stage3_v3_4*.json` configs specifically extend the V3.3 Stage 3 baseline.

No `pose_generation_v34.py` exists by design.

## Portrait output

`output_transform.portrait_output` accepts:

- `off`: exact legacy output behavior.
- `auto_cw90`: rotate only landscape frames clockwise.
- `auto_ccw90`: rotate only landscape frames counter-clockwise.

The transform is applied after final selection and ordering. Each original
camera is rendered once, then RGB, alpha, and optional depth arrays are rotated
together. `pano_cameras.json` records the matching rotated image-plane camera.
Camera position and world-space forward direction do not change.

When portrait mode is enabled, `pano_frame_manifest.json` contains both the
source camera and the output camera plus each frame's applied rotation.
`output_transform.json` records the output contract. `traj_refs.json` only
selects original `train_cameras.json` indices. Captured references keep their
original resolution and orientation; this framework neither rotates nor
rewrites them. Their use is owned by the downstream cross-attention pipeline.

## Configs

- `configs/stage3_v3_4.json`: compatibility baseline, portrait disabled.
- `configs/stage3_v3_4_portrait.json`: clockwise portrait experiment.

The CLI can override either config with:

```text
--portrait-output {off,auto_cw90,auto_ccw90}
```

## Examples

V3.3 candidate generation with V3.4 portrait output:

```bash
python -m viewpoint_framework.run_pipeline \
  --cameras train_cameras.json \
  --point_cloud pi3_init_aligned.ply \
  --gaussian_ply point_cloud_final.ply \
  --pose_config_json viewpoint_framework/configs/v3_3_pose_generation.json \
  --stage3_config_json viewpoint_framework/configs/stage3_v3_4.json \
  --portrait-output auto_cw90 \
  --output_dir outputs/v3_4_portrait
```

The same output experiment over the V3.2 strategy:

```bash
python -m viewpoint_framework.run_pipeline \
  --cameras train_cameras.json \
  --point_cloud pi3_init_aligned.ply \
  --gaussian_ply point_cloud_final.ply \
  --pose_config_json viewpoint_framework/configs/v3_2_pose_generation.json \
  --stage3_config_json viewpoint_framework/configs/stage3_v3_2_grid_only.json \
  --portrait-output auto_cw90 \
  --output_dir outputs/v3_2_v3_4_portrait
```

For an off/off V3.3 baseline, keep using `configs/stage3_v3_3.json`, or use
`stage3_v3_4.json` without a portrait override. This preserves the existing
single reference row and single trajectory length schema.

## Deferred block experiment

Content-aware partitioning requires geometry visibility/co-visibility,
7-frame divisibility, block-local ordering, and per-block captured-reference
selection. It is intentionally not exposed yet: a fake `content` mode based on
consecutive frame chunks would make later denoising results difficult to
interpret. Generated frames reused as context will also use a separate future
contract rather than being mixed into captured `traj_refs` indices.

## Validation

Synthetic tests cover clockwise/counter-clockwise projection mapping, camera
center and forward invariants, rotation round-trip, landscape-only behavior,
and exact raster rotation. The renderer backend comparison issue inherited
from V3.3 remains separate from this output transform.
