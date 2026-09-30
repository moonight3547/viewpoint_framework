# Viewpoint Framework V3.4

V3.4 is a Stage 3 experiment over the existing V3.2/V3.3 pose-generation
strategies. It does not introduce `pose_generation_v34.py` or change Stage 2
geometry safety.

## Portrait viewport

`portrait_output=auto` keeps `w2c`, `c2w`, `fx`, and `fy` unchanged. For a
landscape camera it changes the render canvas from `W x H` to `H x W` and
preserves the principal-point offset from the canvas center. The renderer then
samples the portrait view directly: horizontal coverage becomes narrower and
vertical coverage becomes wider around the same optical-axis content.

No raster rotation or camera roll is performed. The former `auto_cw90` and
`auto_ccw90` values remain compatibility aliases for `auto` so old batch
commands do not fail, but they no longer rotate the image.

References listed by `traj_refs.json` retain their original resolution and
orientation. Their cross-attention use belongs to the downstream pipeline.

## Content blocks

`block_mode=content` operates on the final safe Stage 2 target set:

1. Build geometry-visibility affinity, with an explicitly recorded pose-only
   fallback if visibility rendering is unavailable.
2. Trim at most six redundant targets so the output count is divisible by 7.
3. Produce balanced content blocks, normally targeting 28 frames per block.
4. Order views locally inside each block.
5. Select at most six original captured references per block using geometry
   target recall, with a recorded pose fallback.

For example, 160 valid targets become 154 outputs with block lengths
`[28,28,28,28,21,21]`; 168 targets become six 28-frame blocks.

Block outputs remain flat and globally numbered. The output contract adds:

- `traj_lens.json`: one 7-aligned length per block.
- `traj_refs.json`: one list of original captured-camera indices per block.
- `pano_blocks.json`: candidate IDs, offsets, references and diagnostics.
- `pano_frame_manifest.json`: `block_id`, local index and trunk index.

## Configs and CLI

- `configs/stage3_v3_4.json`: compatibility baseline; portrait and blocks off.
- `configs/stage3_v3_4_blocks.json`: full V3.3 targets with content blocks.
- `configs/stage3_v3_4_portrait.json`: full targets, content blocks and portrait
  viewport.

Important overrides:

```text
--portrait-output {off,auto,auto_cw90,auto_ccw90}
--block-mode {off,content}
--block-max-refs 6
--block-trunk-frames 7
--all-generated-frames
```

V3.3 portrait + blocks:

```bash
python -m viewpoint_framework.run_pipeline \
  --cameras train_cameras.json \
  --point_cloud pi3_init_aligned.ply \
  --gaussian_ply point_cloud_final.ply \
  --pose_config_json viewpoint_framework/configs/v3_3_pose_generation.json \
  --stage3_config_json viewpoint_framework/configs/stage3_v3_4_portrait.json \
  --output_dir outputs/v3_4
```

V3.2 remains directly selectable with
`configs/v3_2_pose_generation.json`; V3.3 remains directly selectable with
`configs/v3_3_pose_generation.json`. The shell wrapper now leaves Stage 3
config values untouched unless the corresponding environment override is
explicitly supplied.
