# Candidate Viewpoint Pose Generation (Stage 2)

This patch adds the second stage after `scene_understanding.py`.

## Scope

Stage 2 only does:

1. freeze one default observation mode (`outside_in` / `inside_out`);
2. export endpoint `view_limits.json`;
3. generate azimuth/elevation grid candidates;
4. project an initial radius from `DirectionalRadiusField`;
5. optionally render reverse-view 3DGS depth to maximize camera backoff;
6. use point-cloud KD-tree checks as a hard geometry safety guard;
7. export every safe candidate to `gen_cameras.json`.

It intentionally does **not** select the final views, perform FPS, information gain,
hole filling, or coverage optimization. Those belong to Stage 3.

## New files

```text
viewpoint_framework/
├── geometry_safety.py
├── gs_depth_probe.py
├── pose_generation.py
├── generate_poses.py
└── configs/
    ├── default_pose_generation.json
    └── prior_only_pose_generation.json
```

No existing public API needs to be changed.

## Default algorithm

```text
SceneUnderstandingResult
        │
        ▼
count-majority default mode
        │
        ├── export view_limits.json
        ▼
azimuth × elevation grid (20°)
        │
        ▼
DirectionalRadiusField -> initial safe radius prior
        │
        ▼
reverse 3DGS depth probe (low-res, center crop, low quantile)
        │
        ├── outside-in: move away from center as far as safely possible
        │
        └── inside-out:
              try moving toward center;
              cross center only if reverse depth supports an opposite-side
              radius >= minRadius
        │
        ▼
point-cloud sampled-path + final-position safety
        │
        ▼
gen_cameras.json
```

### Important Inside-Out convention

The generation policy stays `inside_out` even after a camera crosses the center.
For a grid direction `u`:

- camera forward is always `+u`;
- position is `center + s*u`;
- `s < 0` means the camera crossed the center.

This avoids accidentally flipping a cross-center candidate back to Outside-In.

## Why 3DGS + point cloud are both used

- **3DGS depth**: dense directional free-space estimate; decides how far the camera
  can move backward.
- **point cloud**: cheap all-direction spatial guard; checks camera center and motion
  path clearance.

The render is deliberately low resolution (`max_image_dim=256` by default), because
only a robust depth bound is needed.

## Dependencies for 3DGS depth

Existing framework dependencies remain unchanged for camera-only / point-cloud-only use.
For 3DGS depth probing, additionally use the rendering environment containing:

```bash
pip install plyfile
# plus the already-available torch + gsplat matching your project CUDA environment
```

The default loader expects standard 3DGS PLY properties:

```text
x y z
opacity
scale_0 scale_1 scale_2
rot_0 rot_1 rot_2 rot_3
```

By default `scale_*` is `exp` activated and `opacity` is `sigmoid` activated.

## Run

From the parent directory containing the `viewpoint_framework` package:

```bash
python -m viewpoint_framework.generate_poses \
    --cameras /path/train_cameras.json \
    --point_cloud /path/pi3_init_aligned.ply \
    --gaussian_ply /path/point_cloud_final.ply \
    --output_dir /path/pose_generation
```

Outputs:

```text
pose_generation/
├── view_limits.json
├── gen_cameras.json
└── gen_cameras_meta.json
```

Visualize using the existing codebase:

```bash
python -m viewpoint_framework.visualize_cameras \
    --point_cloud /path/pi3_init_aligned.ply \
    --captured_cameras /path/train_cameras.json \
    --generated_cameras /path/pose_generation/gen_cameras.json \
    --output /path/pose_generation/camera_vis/index.html
```

## Fast ablations

### Disable 3DGS placement optimization

```bash
python -m viewpoint_framework.generate_poses ... \
    --no_gs_depth \
    --outside_placement prior_only \
    --inside_placement prior_only
```

### Force acquisition mode

```bash
--mode outside_in
# or
--mode inside_out
```

### Radius prior

```bash
--radius_strategy directional
--radius_strategy azimuth_only
--radius_strategy bbox_median
```

### Inside-Out crossing

```bash
--inside_placement center_crossing_depth  # default
--inside_placement no_crossing
--inside_placement prior_only
```

### Depth aggregation

```bash
--depth_strategy central_low_quantile      # default, conservative
--depth_strategy median
--depth_strategy mean                      # close to old behavior
```

## Design choice for urgent V1

This implementation does not yet add Gaussian ellipsoid collision, six-direction
cubemaps, exact mesh ray-casting, or coverage selection. The intended V1 acceptance
criteria are:

1. deterministic 20-degree candidate grid;
2. output compatible with the current camera visualizer;
3. good use of aligned 3DGS depth without expensive full-resolution rendering;
4. point-cloud hard safety prevents obvious camera/geometry intersections;
5. every major policy can later be swapped through a strategy/config option.
