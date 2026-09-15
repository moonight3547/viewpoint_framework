# Viewpoint Framework V3.1

V3.1 是在 V3.0 `TrajectorySafeField + candidate-local clearance` 基础上的小步修复。
它不修改 scene center、角度网格、position elevation、height guard 默认状态或
Stage3 选择算法。主要解决 inside-out crossing 朝向、inside-out 半径调整、skybox
几何污染和 gsplat near plane 隐式默认值问题。

推荐配置：`configs/v3_1_pose_generation.json`。兼容入口
`configs/v3_pose_generation.json` 也已更新到同样的 V3.1 默认值。V2 的逐 grid 日志
默认关闭，仍可通过 `console_log_candidates=true` 临时开启。

## 1. Inside-out 位置与朝向

对角度网格原始方向记为 `grid_direction = u`。V3.1 将三个方向分开：

- `grid_direction`：原始方位角/俯仰角方向，生成过程中保持不变；
- `position_direction`：`normalize(final_position - scene_center)`；
- `camera_forward`：相机真实光轴方向。

规则如下：

```text
Outside-in: camera_forward = -position_direction
Inside-out: camera_forward =  grid_direction
FPS:        fps_direction  =  position_direction
```

Inside-out crossing 后相机位于 `scene_center - r*u`，但仍保持
`camera_forward=+u`。因此 crossing candidate 满足：

```text
position_direction = -u
camera_forward      = +u
dot(camera_forward, position_direction) ~= -1
```

Stage3 的内存适配器和 JSON 文件适配器都使用最终位置方向做 angular FPS，不再把
crossing 后的 `camera_forward` 误当成空间位置方向。

## 2. Inside-out crossing 与 radius_max

`radius_max` 只限制远离 `scene_center` 的半径扩张，不禁止向中心移动：

1. Outside-in 向外 proposal 超过上限时保留安全初始位置；
2. Inside-out 同侧 inward adjustment 不受上限限制，即使 `r0 > radius_max`；
3. Inside-out crossing 允许先穿过中心；
4. 如果 proposal 的对侧绝对半径超过上限，使用 `-radius_max` 作为对侧终点；
5. 截断后的完整路径与终点仍必须通过 candidate-local geometry safety；crossing 的整段
   路径检查不受通用 `use_path_safety` 调试开关影响；
6. crossing 路径或终点不安全时，放弃 crossing 并退回已经验证安全的初始位置。

最后一种情况通常不应发生。它会通过
`inside_out_crossing_fallback_initial_count`、candidate metadata 以及
`[S2:V3.1_WARNING]` 明示，便于批量测试定位。

## 3. Skybox Gaussian 检测

`skybox_detection.py` 只使用 Gaussian mean/scale，不使用 `scene_center`。估计出的中心
统一命名为 `skybox_center`。

检测步骤：

1. 使用 Gaussian AABB 得到初始 `skybox_center`、轴向 half extent 和初始半径；
2. 从外层 Gaussian 生成 sphere-fit seeds；
3. 用 IRLS robust sphere fit 修正 `skybox_center` 和 `skybox_radius`；
4. 首先直接判断
   `abs(norm(mean-skybox_center)-skybox_radius) <= 0.005*skybox_radius`；
5. 再用固定 skybox scale 的 log-scale 一致性作保守确认；
6. 仅将同时满足径向和 scale 条件的高置信 Gaussian 划入 skybox。

误删真实场景 Gaussian 的风险高于漏掉少量 skybox。点数不足、轴向各向异性异常等
低置信情况会保留保守 mask，并输出 `[GS:SKYBOX]` warning。诊断信息包含数量、比例、
中心、半径、half extent、anisotropy、径向残差、scale MAD 和 confidence。

## 4. Renderer 数据职责

`GsplatRenderer` 在 PLY load 后保存互斥的 geometry 和 skybox tensor groups，不永久
保存一份 full tensor 再复制 geometry tensor：

- `geometry_*`：Stage2 reverse depth、Stage3 visibility/coverage/IG/hole geometry；
- `skybox_*`：最终 RGB 背景；
- 兼容字段 `means_np/scales_np/opacities_np/max_scale_np` 在 V3.1 表示 geometry-only。

`render_geometry_depth()` 明确排除 skybox。Stage2 depth probe、Gaussian visibility 和
point-cloud/GS gap detection 均只使用 geometry group。`render_rgb()` 临时组合两组
Gaussian 交给 gsplat 做一次完整、按深度排序的 rasterization，因此最终 RGB 保留
skybox，且不会用 skybox 伪造墙面或 hole coverage。

Generated camera 还会检查是否位于 `skybox_radius - margin` 内。V3.1 只记录
`CAMERA_OUTSIDE_SKYBOX`，不因此拒绝视角。

## 5. Explicit near plane

所有 gsplat rasterization 显式传入 `near_plane`。默认策略为：

```text
near_plane = 0.01 * median_captured_horizontal_radius
```

水平半径相对 Stage1 的 center/up 计算，不使用可能被窗外远景污染的 point-cloud AABB。
建议批量比较 ratio `0.005 / 0.01 / 0.02`。Near plane 只是渲染稳定性的第二层保护，
不能替代 local clearance 或最终 hard geometry check。

## 6. Metadata 与统计

每个 V3.1 candidate 的 `geometry_metadata` 记录：

```text
grid_direction
position_direction
camera_forward
fps_direction
position_azimuth_deg / position_elevation_deg
proposed_signed_radius / final_signed_radius
adjustment_type
radius_max_applied / crossing_radius_capped
crossing_failed_fallback_initial
local_clearance_threshold
renderer_near_plane
depth_probe_excludes_skybox
```

`gen_cameras_meta.json` 顶层 `skybox` 和 `renderer` 保存 scene-level renderer 诊断。
Stage2 diagnostics 另外汇总 inside-out crossing、同侧 inward、初始半径超限后成功调整、
crossing cap/fallback、geometry/skybox 数量及 detection confidence。

## 7. 运行与验证

Stage2：

```bash
python -m viewpoint_framework.generate_poses \
  --cameras train_cameras.json \
  --point_cloud aligned_points.ply \
  --gaussian_ply point_cloud_final.ply \
  --config_json viewpoint_framework/configs/v3_1_pose_generation.json \
  --output_dir output
```

E2E 使用同一配置作为 `--pose_config_json`。Standalone Stage3 会从
`gen_cameras_meta.json` 恢复相同 skybox 与 near-plane 配置，保证其 geometry/FPS
语义和 E2E 路径一致。

自动测试覆盖 inside-out crossing/no-crossing、`r0 > radius_max` inward、crossing cap
及安全回退、synthetic skybox、外层真实 geometry 保留、geometry-only depth、Stage3
geometry-only sampling 和 captured-scale near plane。

## 8. V3.1 暂不处理

Height guard 仍默认关闭；暂不加入 position elevation/view pitch 解耦、voxel occupancy、
strict inside-room、front/back directional clearance、scene-center 新算法或 Outside-in
专项 near-geometry 策略。这些应根据 V3.1 批量测试的 remaining broken cases 决定
V3.2 优先级。
