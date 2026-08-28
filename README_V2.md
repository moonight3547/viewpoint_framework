# Viewpoint Framework V2

V2 targets Stage-2 candidate scarcity while preserving the V1 implementation as
the default/configurable baseline. Stage 3 and free-space/room containment are not
changed in this version.

## V2 semantics

- Stage 1 keeps robust four-way camera understanding (`outside_in`, `inside_out`,
  `ambiguous`, `outlier`). The V2 thresholds are slightly relaxed to 25 degrees
  alignment, 0.45 residual ratio, and 0.15 fit weight.
- When the larger strict Outside-In/Inside-Out subset contains at least 20% of all
  captured cameras, its sight lines refine scene center exactly once. Four-way
  labels are not classified again after refinement.
- Stage 2 independently assigns every captured camera to Outside-In or Inside-Out
  from the refined center and chooses count majority.
- The V2 angular bbox uses that binary dominant subset. Azimuth uses the minimal
  circular interval; elevation uses robust 2/98 percentiles. Each side is expanded
  by 10% of its observed span. Circular azimuth expansion continues past 180
  degrees and saturates only at 360 degrees.
- Directional radius support uses all finite captured positions with equal support
  weight, independent of Stage-1 acquisition labels. If a directional estimate is
  unavailable, fallback remains the Stage-2 dominant-mode radius median.
- `view_limits.json` uses unexpanded observed angular bounds and the 40/60 radius
  percentiles of the Stage-2 binary dominant subset.

## Configuration

V1 remains available through:

```text
configs/default_scene_understanding.json
configs/default_pose_generation.json
```

V2 is enabled explicitly through:

```text
configs/v2_scene_understanding.json
configs/v2_pose_generation.json
```

The V2 pose config prints one compact line per grid point and one per candidate.
The candidate line includes azimuth/elevation in degrees, initial/final radius,
radius source, depth status, clearance, path safety, final status, and rejection
reason. The same structured information remains in `gen_cameras_meta.json`.

## Server A/B test

From the package directory or any other directory:

```bash
bash experiments/test_v2_001_stage2_candidate_expansion.sh \
  /path/train_cameras.json \
  /path/pi3_init_aligned.ply \
  /path/point_cloud_final.ply \
  /path/select_view_dir \
  /path/output_root
```

The script writes isolated `v1_baseline`, `v2_stage2`, and `logs` directories.
