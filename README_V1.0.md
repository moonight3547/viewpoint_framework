# Stage 1: Scene Understanding

Stage 1 为 V3.2/V3.3 提供统一的 scene center、collection mode、球面坐标系、
mode-specific bbox 和 directional radius field。

默认配置是 `configs/stage1_scene_understanding.json`，包含：

- `robust_irls` scene-center fitting；
- robust per-camera outside-in / inside-out / ambiguous / outlier classification；
- weighted global collection mode；
- dominant-mode one-shot center refinement；
- circular robust azimuth bbox 与 percentile elevation/radius bbox；
- all-camera `angular_knn` radius field。

运行：

```bash
python -m viewpoint_framework.analyze_scene \
  --cameras train_cameras.json \
  --point_cloud pi3_init_aligned.ply \
  --config_json viewpoint_framework/configs/stage1_scene_understanding.json \
  --output_dir outputs/scene_analysis \
  --no_serve
```

输出包括 `scene_profile.json`、`camera_relations.json` 和实际使用的
`strategy_config.json`。Stage 2 正常运行时直接在内存中消费同一
`SceneUnderstandingResult`。

## 当前策略边界

历史 `legacy_check_alignment`、`legacy_sign`、普通 min/max bbox、
historical view-limits import 和旧 point-cloud radius rule 已删除。保留的可选
radius field（`global_median`、`nearest`、`angular_knn`、
`pointcloud_cone`、`hybrid`）只影响 Stage 1 profile/debug；V3.2/V3.3
placement 的安全约束仍由 trajectory safe field、point-cloud KNN 和
local/global height safety 决定。
