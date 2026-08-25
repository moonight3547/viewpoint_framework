# Stage 3 — Denoising View Selection & End-to-End Integration

This patch adds Stage 3 to `viewpoint_framework` and wires Stage 1 → Stage 2 → Stage 3 into one runnable pipeline.

## 1. Output contract

`--output_dir` always contains the runtime artifacts required by the downstream denoising pipeline:

```text
output_dir/
├── view_limits.json           # Stage 2 endpoint constraint
├── gen_cameras.json           # all Stage-2 geometry-safe candidates, 18-D
├── gen_cameras_meta.json      # Stage-2 candidate provenance
├── pano_cameras.json          # final Stage-3 denoising targets, 18-D
├── pano_images/
│   ├── frame_0000.png
│   └── ...
├── traj_refs.json             # [[0,125,296,...]]
└── traj_lens.json             # [49] (actual selected count)
```

When `--debug-mode` is enabled, **all Stage-3-only debug artifacts are placed under `output_dir/debug/`**:

```text
debug/
├── stage1_scene_profile.json
├── candidate_images/          # RGB render of every Stage-2 valid candidate
└── stage3_metadata.json       # holes, selection trace, reference trace, coverage stats
```

Normal mode does not render all Stage-2 candidates; only final `pano_images` are rendered.

## 2. Stage-3 structure

```text
viewpoint_framework/stage3/
├── types.py                  # data contracts
├── selected_views.py         # --select_view_dir loader
├── visibility.py             # Gaussian / point-cloud coverage backend
├── hole_detection.py         # 3-D under-observation holes + focused views
├── selection.py              # FPS / greedy coverage / ordering
├── reference_selection.py    # reusable reference frame selection
├── render_output.py          # pano/debug render and JSON output
└── pipeline.py               # orchestration
```

Public entry points:

```text
viewpoint_framework/select_views.py   # Stage 3 standalone
viewpoint_framework/run_pipeline.py   # Stage 1→2→3 in-memory E2E
run_viewpoint_pipeline.sh              # root bash experiment runner
```

Shared Stage-2/3 Gaussian renderer:

```text
viewpoint_framework/gs_renderer.py
```

The E2E runner loads the Gaussian PLY only once. Stage 2 uses it through `GsplatDepthProbe(renderer=...)`; Stage 3 reuses it for Gaussian visibility, geometric-hole analysis, reference covisibility and final RGB rendering.

## 3. Default Stage-3 policy

Default config: `viewpoint_framework/configs/default_stage3.json`

```text
Base selection     : angular_fps
Selection seed     : generated_only
Information gain   : gaussian_visibility (tie-break only)
Quality band       : none
Near captured pose : hard duplicate filter only
Geometric holes    : gaussian_undercoverage
References         : legacy_global_fps
Ordering           : grid_order
num_panos          : 49
num_refs           : 12
```

### Coverage first, IG second

Angular FPS remains the primary selection metric. When several candidates are within `ig_tie_ratio` (default 0.97) of the best current angular-FPS distance, Gaussian information gain breaks the tie. This prevents a representation score from collapsing directional coverage.

## 4. Information gain

For sampled surface elements (Gaussian samples or point-cloud samples), define:

- `V(v)`: surface samples visible from candidate view `v`;
- `C_j(S)`: number of already selected/anchor views in set `S` that see sample `j`;
- `w_j`: sample weight (Gaussian opacity-derived for the Gaussian backend).

The implemented marginal gain is:

```text
IG(v | S) = Σ_{j ∈ V(v)} w_j / (1 + C_j(S))
```

Interpretation:

```text
C=0  -> full gain w
C=1  -> 1/2 w
C=2  -> 1/3 w
C=3  -> 1/4 w
...
```

This is a diminishing-return coverage objective: the first observation of a region matters most, while repeated observations remain useful but contribute less. The same interface supports:

```text
none
gaussian_visibility
pointcloud_visibility
```

`greedy_coverage` uses this objective directly. `angular_fps` uses it only as a near-tie break.

## 5. Geometric holes (not angular-grid holes)

Stage 2 already samples the angular grid, so Stage 3 does **not** add cameras to fill 2-D grid/Voronoi gaps.

V1 geometric-hole detection:

1. Load the preselected captured anchors from `--select_view_dir` (normally ~40 views).
2. Sample opacity-weighted Gaussians from the aligned high-quality 3DGS.
3. For each selected captured anchor, render low-resolution Gaussian depth/alpha.
4. Count how many selected captured anchors visibly observe every sampled Gaussian.
5. Mark samples with low coverage (`coverage <= max_coverage_count`) as under-observed.
6. Voxel-connect under-observed samples into 3-D clusters.
7. For each important cluster, reuse a Stage-2 safe candidate **position** in the matching scene sector.
8. Reorient the camera toward the hole centroid.
9. Adjust focal length so the cluster plus margin fits the view.
10. If one view cannot fit the region at the minimum allowed focal scale, recursively split the 3-D cluster along its PCA major axis.
11. Re-render visibility from the focused camera; only keep views that actually see a minimum fraction of the target cluster.

Reasonable hole views are marked `forced_select=True` and reserve Stage-3 pano slots before generic FPS selection.

A second experimental hole strategy is also implemented:

```text
pointcloud_gaussian_gap
```

It treats the aligned point cloud as independent geometry evidence and searches for sampled point-cloud regions that remain farther than a scale-aware threshold from opaque Gaussian support. By default a gap point must also be visible from at least one of the preselected captured anchors, which suppresses isolated feed-forward floaters. This option is closer to a literal reconstruction-geometry hole, while `gaussian_undercoverage` remains the V1 default because it is more conservative.

Because camera position is inherited from a Stage-2 safe candidate, Stage-2 geometry collision guarantees remain valid. Only orientation/intrinsics change.

## 6. Selection strategies

```text
legacy_position_fps
angular_fps                 # default
utility_angular_fps
greedy_coverage
```

Selection reference options:

```text
generated_only              # default
captured_seeded             # experimental combined captured+pano angular coverage
```

`captured_seeded` is intentionally retained as an ablation. Captured cameras are not assumed to form a perfect spherical trajectory; outside-in uses their center-relative position sector and inside-out uses forward direction. For non-radial captures, `greedy_coverage + gaussian_visibility` is generally the more representation-aware combined-coverage experiment.

## 7. Denoising quality strategy

V1 does **not** infer under-supervision simply from pose distance. `quality_strategy=none` is the default.

A very conservative duplicate filter removes a generated grid view only when it is both:

- spatially extremely close to a real captured camera; and
- has nearly the same forward direction.

Focused hole views bypass this cheap filter because the hole detector has explicit under-observation evidence.

`render_quality_band` is reserved in the strategy interface for later experiments; selecting it currently raises `NotImplementedError` rather than silently using an unvalidated proxy.

## 8. Reference selection

Reference selection is independent from pano target selection.

Inputs:

```text
train_cameras.json                  # original captured cameras / original indices
--select_view_dir/selection.json    # selected_indices in original sequence
--select_view_dir/selected_cameras.json (optional)
final pano cameras
optional visibility model
```

Output:

```text
ReferenceSelectionResult.original_indices
```

These original indices are written as:

```json
[[0,125,296,368,414,1039]]
```

Strategies:

### `legacy_global_fps`

Matches the previous panorama generator's position FPS on the reference candidate pool. Original frame 0 is forced if available. Default V1 reference strategy.

### `target_coverage_greedy`

Greedy facility-location objective over final pano targets:

```text
maximize Σ_target max_{selected_ref} support(ref, target)
```

V1 support combines forward-angle and normalized position similarity. The optimizer is intentionally separate from the metric so a projected-image/render-content support metric can replace it later.

### `artifixer_style_covisibility`

An **ArtiFixer-inspired** sparse-view scene-coverage strategy, not a claim of exact code reproduction. It greedily selects captured references whose Gaussian visible sets maximize diminishing-return scene coverage. The public ArtiFixer preparation describes half-covisibility sparse-camera split generation and 2/3/6/12-view sparse reconstructions; this option provides a directly testable covisibility-style counterpart in this framework.

No acquisition-mode filtering is applied to reference candidates.

## 9. Ordering

Selection and ordering are separate.

```text
grid_order        # V1 default; preserves Stage-2 grid progression
nearest_neighbor  # optional pose-continuity experiment
selection_order
```

Hole views retain the source Stage-2 row/column and are placed near that location in `grid_order`.

## 10. Standalone Stage 3

```bash
python -m viewpoint_framework.select_views \
    --cameras /path/train_cameras.json \
    --point_cloud /path/pi3_init_aligned.ply \
    --gaussian_ply /path/point_cloud_final.ply \
    --select_view_dir /path/select_views \
    --output_dir /path/output \
    --num_panos 49 \
    --num_refs 12
```

Debug:

```bash
... --debug-mode
```

## 11. End-to-end Stage 1→2→3

```bash
python -m viewpoint_framework.run_pipeline \
    --cameras /path/train_cameras.json \
    --point_cloud /path/pi3_init_aligned.ply \
    --gaussian_ply /path/point_cloud_final.ply \
    --select_view_dir /path/select_views \
    --output_dir /path/output \
    --num_panos 49 \
    --num_refs 12
```

Or use the root bash script:

```bash
bash run_viewpoint_pipeline.sh \
    /path/train_cameras.json \
    /path/pi3_init_aligned.ply \
    /path/point_cloud_final.ply \
    /path/select_views \
    /path/output
```

Ablation example without editing Python:

```bash
SELECTION_STRATEGY=greedy_coverage \
SELECTION_REFERENCE=captured_seeded \
INFORMATION_GAIN=gaussian_visibility \
REFERENCE_STRATEGY=artifixer_style_covisibility \
NUM_PANOS=49 NUM_REFS=12 DEBUG_MODE=1 \
bash run_viewpoint_pipeline.sh \
    train_cameras.json pi3_init_aligned.ply point_cloud_final.ply \
    select_view_dir output_exp
```

## 12. Suggested first ablations

```text
A. legacy_position_fps vs angular_fps
B. generated_only vs captured_seeded
C. IG none vs gaussian_visibility
D. holes none vs gaussian_undercoverage vs pointcloud_gaussian_gap
E. refs legacy_global_fps vs target_coverage_greedy vs artifixer_style_covisibility
F. ordering grid_order vs nearest_neighbor
```

Keep `quality_strategy=none` until a render/content-based denoising-quality proxy is validated.


## 13. Tests

After installation, from the parent directory of `viewpoint_framework`:

```bash
python -m pytest \
  viewpoint_framework/tests/test_scene_understanding.py \
  viewpoint_framework/tests/test_pose_generation_synthetic.py \
  viewpoint_framework/tests/test_stage3_core.py -q
```

The bundled synthetic suite currently covers Stage 1 center/mode logic, Stage 2 outside/inside pose semantics, Stage 3 forced-hole selection/reference selection, and the point-cloud-vs-Gaussian gap detector.
