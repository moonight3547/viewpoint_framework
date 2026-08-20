# Scene Understanding — Phase 1 Refactor

This patch adds the first algorithm layer on top of the existing `viewpoint_framework` camera / point-cloud / Plotly utilities.

The stage intentionally **does not generate new cameras yet**. It converts captured-camera input into a reusable `SceneProfile`:

```text
Captured Cameras
      |
      v
scene center fit
      |
      v
per-camera collection mode
      |
      v
per-mode spherical view bbox
      |
      v
directional camera-radius field
      |
      v
SceneProfile + diagnostic HTML
```

## Files

Copy the new Python files in `viewpoint_framework/` into the existing repository directory next to:

```text
geometry_util.py
cameras_util.py
points_util.py
util.py
visualize_cameras.py
```

New files:

```text
scene_types.py
scene_analysis.py
view_space.py
radius_field.py
scene_understanding.py
visualize_scene_analysis.py
analyze_scene.py
```

No existing public visualization API is changed in this phase.

Reproducible experiment configs are included under `configs/`. CLI precedence is `preset < config_json < explicit CLI args`.

---

## 1. Strategy design

Every algorithmic change that should be ablated is exposed as a strategy rather than hard-coded into the pipeline.

### Scene center

`--center_strategy`

| Strategy | Purpose |
| --- | --- |
| `legacy_check_alignment` | Strict historical two-pass `check_camera_alignment()` behavior. It intentionally keeps the first-camera/zero-target weakness for baseline comparison. |
| `legacy_augmented_normal_eq` | Historical `compute_sight_center()` formulation with `[center, lambda_i]` unknowns and normal equations. |
| `projected_ls` | Improved 3x3 point-to-line least squares using `P_i = I - d_i d_i^T`. |
| `robust_irls` | Default. Projected least squares + Huber/Tukey IRLS. |

### Per-camera collection mode

`--mode_strategy`

| Strategy | Purpose |
| --- | --- |
| `legacy_sign` | Every camera is classified only by the sign of `lambda = forward dot (center - position)`. |
| `robust` | Default. Adds center-fit weight, line residual/radius, and alignment-angle gates; can output `ambiguous` and `outlier`. |

Global aggregation is separately controlled by:

```text
--global_mode_strategy legacy_majority
--global_mode_strategy weighted
```

### View-space bbox

`--bbox_strategy`

| Strategy | Purpose |
| --- | --- |
| `legacy_view_limits` | Reads historical `view_limits.json` and reproduces old fixed `0.1 rad` angular extension. |
| `legacy_minmax` | Camera-derived ordinary min/max rectangular bbox, preserving azimuth wrap-around weakness for ablation. |
| `circular_robust` | Default. Shortest circular azimuth interval + percentile elevation/radius + adaptive angular extension. |

BBoxes are built **per camera mode**. Outside-in cameras do not contaminate the inside-out bbox and vice versa.

### Directional radius field

`--radius_strategy`

| Strategy | Purpose |
| --- | --- |
| `legacy_pointcloud_rule` | Historical point-cloud cone depth + historical radius rules/post-scale. Outside-in matches the old point-cloud branch; inside-out uses point-cloud cone as an explicit proxy because the old implementation depended on a 3DGS renderer. |
| `global_median` | Direction-independent median captured-camera radius; useful legacy camera-only baseline. |
| `nearest` | Radius from nearest captured radial direction. |
| `angular_knn` | Default. Angular-KNN weighted median / quantile radius prior with support confidence. |
| `pointcloud_cone` | Historical cone-median geometry depth exposed directly as a strategy. Not an exact ray first hit. |
| `hybrid` | Angular-KNN camera prior constrained by point-cloud cone geometry. |

The runtime `DirectionalRadiusField` is continuous: later viewpoint generation can call `field.query(direction)` at arbitrary directions. The `scene_profile.json` stores a sampled grid only for diagnostics and experiment inspection.

---

## 2. Recommended first run

Improved CPU pose-only analysis:

```bash
python analyze_scene.py \
    --cameras /path/to/train_cameras.json \
    --point_cloud /path/to/pi3_init_aligned.ply \
    --output_dir outputs/scene_analysis \
    --center_strategy robust_irls \
    --mode_strategy robust \
    --bbox_strategy circular_robust \
    --radius_strategy angular_knn \
    --no_serve
```

`--point_cloud` is optional for `angular_knn`; it is still useful for the HTML scene context.

The same experiment can be pinned to a config file:

```bash
python analyze_scene.py \
    --cameras /path/to/train_cameras.json \
    --config_json configs/default_scene_understanding.json \
    --output_dir outputs/scene_analysis \
    --no_serve
```

Outputs:

```text
outputs/scene_analysis/
├── scene_profile.json
├── camera_relations.json
├── strategy_config.json
└── index.html
```

To launch the same localhost server behavior as `visualize_cameras.py`, omit `--no_serve`.

---

## 3. Legacy ablation

Camera-only historical baseline:

```bash
python analyze_scene.py \
    --cameras /path/to/train_cameras.json \
    --preset legacy \
    --output_dir outputs/scene_analysis_legacy \
    --no_serve
```

The `legacy` preset selects:

```text
center  = legacy_check_alignment
mode    = legacy_sign + legacy_majority
bbox    = legacy_minmax
radius  = global_median
```

To reproduce the historical external bbox source:

```bash
python analyze_scene.py \
    --cameras /path/to/train_cameras.json \
    --view_limits /path/to/view_limits.json \
    --bbox_strategy legacy_view_limits \
    --center_strategy legacy_check_alignment \
    --mode_strategy legacy_sign \
    --global_mode_strategy legacy_majority \
    --radius_strategy global_median \
    --output_dir outputs/legacy_view_limits \
    --no_serve
```

To compare the old outside-in point-cloud radius logic:

```bash
python analyze_scene.py \
    --cameras /path/to/train_cameras.json \
    --point_cloud /path/to/pi3_init_aligned.ply \
    --view_limits /path/to/view_limits.json \
    --bbox_strategy legacy_view_limits \
    --radius_strategy legacy_pointcloud_rule \
    --output_dir outputs/legacy_radius \
    --no_serve
```

---

## 4. SceneProfile semantics

### Center fit

The improved estimator minimizes perpendicular distance from one common point `c` to all camera sight lines:

```text
P_i = I - d_i d_i^T
min_c sum_i || P_i (c - p_i) ||^2
```

`robust_irls` additionally estimates per-camera robust weights.

### Per-camera relation

For every captured camera:

```text
radius             = ||position - center||
lambda_center      = forward dot (center - position)
sight_residual     = distance(center, camera sight line)
alignment_deg      = acos(|lambda_center| / radius)
radial_direction   = (position - center) / radius
```

Mode sign convention:

```text
lambda_center > 0  -> outside_in
lambda_center < 0  -> inside_out
```

Robust mode may override the sign label with `ambiguous` or `outlier` when geometric support is weak.

### Spherical coordinates

Internal names are explicit:

```text
azimuth   in [-180, 180)
elevation in [-90, 90]
radius    > 0
```

Default frame:

```text
world_up              = [0, 1, 0]
azimuth_zero_reference= [0, 0, 1]
```

These are CLI-configurable. No hidden coordinate correction is performed.

---

## 5. Diagnostic HTML

The analysis HTML adds to the existing Plotly camera visualization:

- fitted scene-center marker;
- original captured trajectory;
- outside-in / inside-out / ambiguous / outlier camera groups;
- camera sight-axis projection and center residual segment;
- observed spherical bbox;
- expanded generation bbox;
- sampled directional radius estimates.

The visualizer is downstream-only: it does not modify algorithm outputs.

---

## 6. Suggested ablations

Center only:

```text
legacy_check_alignment
legacy_augmented_normal_eq
projected_ls
robust_irls
```

BBox only:

```text
legacy_minmax
circular_robust
```

Radius only:

```text
global_median
nearest
angular_knn
pointcloud_cone
hybrid
```

This makes each later experiment a strategy/config change rather than a source-code edit.
