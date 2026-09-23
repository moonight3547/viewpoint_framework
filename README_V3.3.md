# Viewpoint Framework V3.3

V3.3 在 V3.2 已验证的 trajectory + geometry safety 基线上扩大可探索视角域，并把“候选提出”和“安全证明”分开：trajectory 提供可信初始位置，Gaussian depth、Local/Global Height、Center Guard、完整路径与终点检测共同决定候选是否可用。

## V3.2 基线、效果与遗留问题

基线 commit 为 `93fbefa`。既有 34-case 批量测试为 34/34 成功，耗时 675 秒；Stage2 共 2472 个 grid、2387 个有效候选（96.6%），Stage3 输出 1536 张 pano。562 个 direct-rho column 与 43 个 fallback-rho column 中，segment-ray direct 比例约 92.9%。

V3.1 人工检查的 86 个异常中，78 个属于 skybox/ceiling/floor 等 vertical placement 问题。V3.2 引入 segment-ray rho、geometry-only Local Height、Global Height 及 initial/final height clip 后，34-case 中人工标记异常降为 7 帧，且不再出现大规模 skybox-only、越过 ceiling 或贴 floor/ceiling。因此 V3.3 保留 V3.2 的 geometry/height safety 主链和旧行为入口。

V3.2 的主要限制是 proposal 仍偏保守：`radius_max` 对原地/窄轨迹/走廊/大型场景角落约束过强；position elevation 不能表达真实 eye pitch/FOV；azimuth 小缺口无法闭环；inside-out 前贴几何需要更强 diagnostics 继续观察。V3.3 不加入独立 view pitch、Gaussian-aware 最终硬拒绝、多房间/多楼层等复杂策略。

## V3.3 算法

### 1. Angular proposal

- 使用所有有限 captured cameras 的 canonical `camera.forward`。
- 每帧计算 eye pitch，以及 `0.5 * max(fov_x, fov_y)`；`lower_i/upper_i` 分别取 2/98 percentile。
- outside-in 将 viewing envelope 反号映射到 position elevation；inside-out 保持同号。
- elevation 两端各扩张 `max(10% span, 10°)`，并 clamp 到 `[-80°, 80°]`。
- azimuth 扩张 `max(10% span, 20°)`；剩余 gap 不超过 90° 时闭合为 360°。
- 360° grid 使用半开区间，避免 `-180°/+180°` 重复。

### 2. 先 rho、后 height

所有方位先完成 rho sub-stage，再开始 height sub-stage：

- observed column 使用 trajectory segment-ray minimum。
- observed range 两端外延直接复制临近边界 `rho(phi)`。
- 360° close-loop 中间 gap 在外延后的两端 rho 之间插值。
- 两侧外延交替传播临近 direct column 的 `h_cross`。每一步必须同时通过目标点 clearance 与相邻方位路径 safety；某侧一旦失败，该侧后续 column 熔断并使用 Global Height。两侧传播直到结束或相遇。
- fallback rho column 不执行不可信的 Local Height probe，直接使用 Global Height。

当前传播安全验证采用 point-cloud endpoint/path probe；接口保持独立，后续可替换为低分辨率 Gaussian depth sweep。

### 3. Local / Global Height

Local Height 的 probe origin 固定为 `P_traj(phi) = rho(phi) × h_cross(phi)`，向上/向下分别 geometry-only render。障碍距离采用 central crop Q10，并用中心 3×3 patch 的最小距离作 safeguard。`hole_ratio > 0.5 && center invalid` 标记为 hole-uncertain，不把空洞当作无限安全空间。

批量测试发现 Coverage Consensus 在部分场景会放宽到贴地面或越过吊灯的高度。当前 V3.3 安全基线暂时强制使用所有可靠 Local Height interval 的 strict common intersection；交集为空或没有可靠 interval 时回退 captured height range。Coverage Consensus 实现仍保留，等待后续版本实验改进。margin 只在 Local Height 中扣除一次。

### 4. Radius 与 crossing

- 不再使用 placement `radius_min`；中心附近由 `center_guard = max(local_clearance, renderer_near_plane)` 管理。
- nominal cap 为 `2 × max captured horizontal radius`。
- depth-supported endpoint 超过 nominal 时，只尝试 `(2*R_cap + R_probe)/3`；失败后 retry nominal。
- depth probe 判定有空洞时禁止 over-nominal extension，严格限制在 nominal cap 内。
- Emergency radius 为 `skybox_radius - max(local_clearance, near_plane)`；无 skybox 时使用 geometry radius Q99。不会减去 `norm(scene_center-skybox_center)`。
- outside-in 向外移动并始终面向 scene center。
- inside-out non-crossing 向 center 移动并背向 scene center。
- inside-out crossing 穿过 center 后在对侧向外，最终重新面向 scene center；crossing 使用 Global Height，并强制完整 path + endpoint safety。失败时保留安全初始位置并在 console/diagnostics 明示。

### 5. Skybox

`tail_strict` 首先按原始 PLY 顺序验证末尾 40962 个 Gaussian。验证使用到 AABB center 的 0.5% radial band、逐 Gaussian 三轴 scale 一致性、scale population 一致性和近全覆盖 spherical angular bins。tail 验证失败后才执行同样严格的无顺序 outer-shell 判断；任一高阈值条件不满足即认为无 skybox。检测来源、数量、radial/angular coverage 与 scale 指标输出到 console。

Skybox split 发生在任何 finite filtering 之前，因此 tail 语义不会被重排。geometry-only 数据用于 depth、height、visibility 和正式 pano；full set 仅保留给显式 debug。

## Renderer 与输出 contract

`renderer.backend` 支持：

- `auto`：先执行顶层 `import gs_render`。只有包本身不存在时才尝试 `gsplat`；一旦 import 成功，本次 pipeline invocation 永久锁定 `gs_render`，adapter 初始化或运行错误均直接失败。
- `gs_render` / `gsplat`：仅尝试指定 backend。
- backend 一旦选定，render 过程绝不切换；rasterization 异常统一为 `Renderer Rasterization Failure`。

`gs_render` adapter 直接构造 `GsRenderGaussianData`、`GsRenderCameraData` 和 `GsRenderConfigData`。RGB/alpha-only 请求调用 `GsRenderer.render`；需要 depth 时调用 `GsRenderer.render_with_distance`，再通过 `compute_plane_depth(normal, distance, camera)` 转换为统一的 camera-space planar/Z depth。两条路径都从返回 tuple 的 index 1 读取 alpha，index 2 为可选 semantics。console 会输出 backend 与 version。`GaussianSceneData` 保存 PLY/raw optimization arrays：未归一化 quaternion、log-scale 和 opacity logits；`gs_render` 原样接收 raw tensors，`gsplat` 在自身 backend 边界执行 normalize/exp/sigmoid。V3.3 配置显式使用黑色 background，`clamp_color_min=false`。

Stage3 每个最终相机只进行一次 geometry-only unified render，并输出：

- `pano_images/frame_XXXX.png`：geometry-only RGB；
- `pano_alphas/frame_XXXX.png`：geometry-only 单通道 uint8 alpha；
- 可选 `pano_depths/frame_XXXX.npy`：float32 camera-Z planar depth，0 表示 invalid，并带 `depth_meta.json`。
- `pano_frame_manifest.json`：稳定记录 output index 到 candidate/grid/row/column/camera matrix 的对应，用于 `gsplat` 和 `gs_render` 按 grid id 逐帧对齐。

新增 `--render-pano-depths`。Stage3 baseline 关闭 focused hole views，并提供 `elevation_center_out` 输出顺序：先完整输出最接近 elevation midpoint 的 grid row，再依次完整输出向上的 rows，最后依次完整输出向下的 rows；每一行严格按 circular grid column 排列，最后才使用 stable candidate id 打破平局。

`configs/stage3_v3_3.json` 当前开启 `output_all_stage2_candidates=true`：Stage3 跳过 FPS/coverage、near-duplicate filter 和 `num_panos` 降采样，直接按上述 V3.3 顺序输出所有 Stage-2 valid grid candidates。也可以通过 `--all-generated-frames` 对其他 Stage3 配置显式开启；该模式要求 `holes.strategy=none`，以保持 Stage-2 候选与输出帧一一对应。
两次完整 pipeline 对照可分别传入 `--renderer-backend gsplat` 和 `--renderer-backend gs_render`，从而不修改配置文件中的其他参数。

为排除 Stage-2 depth 导致的位姿差异，可以将同一份固定 camera JSON 分别交给两个 backend：

```powershell
python -m viewpoint_framework.renderer.render_sequence_diagnostic `
  --backend gsplat --gaussian-ply <scene.ply> --cameras <gen_cameras.json> `
  --pose-config-json viewpoint_framework/configs/v3_3_pose_generation.json `
  --output-dir <compare>/gsplat

python -m viewpoint_framework.renderer.render_sequence_diagnostic `
  --backend gs_render --gaussian-ply <scene.ply> --cameras <gen_cameras.json> `
  --pose-config-json viewpoint_framework/configs/v3_3_pose_generation.json `
  --output-dir <compare>/gs_render
```

该诊断默认只渲染 geometry-only RGB/alpha；只有显式传入 `--render-depths` 才调用 depth path。两次必须使用相同的 `--max-image-dim` 和 `--near-plane`。

输出可逐帧汇总 RGB/alpha 误差、camera 对齐，以及可选 depth valid-mask/relative error：

```powershell
python -m viewpoint_framework.renderer.compare_sequence_outputs `
  --left <compare>/gsplat --right <compare>/gs_render `
  --output-json <compare>/comparison.json
```

## 目录与兼容性

V3.3 正式代码进入 `stage1/`、`stage2/`、`renderer/`、`utils/`、`visualization/`。`stage2/pipeline.py` 是 V3.3 主实现，不新增 `pose_generation_v33.py`。本轮按要求不删除 V1/V2/V3.2 文件和旧 Python import 路径；旧模块由薄 forwarding surface 与原实现继续兼容，清理留到后续版本。

关键配置：

- `configs/v3_3_pose_generation.json`
- `configs/stage3_v3_3.json`

推荐命令：

```powershell
python -m viewpoint_framework.run_pipeline `
  --cameras <cameras.json> --point_cloud <points.ply> `
  --gaussian_ply <scene.ply> --output_dir <output> `
  --pose_config_json viewpoint_framework/configs/v3_3_pose_generation.json `
  --stage3_config_json viewpoint_framework/configs/stage3_v3_3.json
```

候选可视化新入口：

```powershell
python -m viewpoint_framework.visualization.candidate_placements `
  --point_cloud <points.ply> --metadata <gen_cameras_meta.json> `
  --stage3_metadata <debug/stage3_metadata.json> --show_local_limits
```

visualizer 支持 V3.2/V3.3 metadata，显示 scene center、Global/Local Height、缩小后的 camera frustum，以及 radius/crossing/hole diagnostics hover。frustum 默认基于 median captured rho，而不是 point-cloud extent。

## `gs_render` contract smoke test

The canonical scene loader preserves the PLY optimization representation:
unnormalized quaternions, log-scales, and opacity logits. `gs_render` receives
those raw tensors because it performs normalization/`exp`/`sigmoid` internally.
The `gsplat` adapter activates the same canonical arrays at its own boundary.
Geometry-side NumPy fields and skybox scale checks continue to use activated,
physical values.

针对首次 Stage2 depth probe 的 CUDA illegal memory access，新增真实私有 backend 隔离脚本：

```bash
python -m viewpoint_framework.renderer.gs_render_contract_smoke \
  --gaussian-ply <scene.ply> \
  --cameras <cameras.json> \
  --pose-config-json viewpoint_framework/configs/v3_3_pose_generation.json \
  --camera-index 0 --probe-max-dim 256 --device cuda \
  --output-dir <output>/gs_render_contract_smoke
```

脚本默认只在独立 Python 进程中测试 `full_render`。只有显式传入 `--cases`时才会运行 `geometry_render`、`geometry_distance`、`probe_render` 或 `probe_distance`，避免在 raw contract 确认前提前调查 distance/bucket。每次 rasterization 前记录 GaussianData/CameraData/config 的 shape、dtype、device、contiguous、finite 与数值范围，CUDA 同步成功后输出 `GS:SMOKE_PRE_RASTER_OK`；rasterization 后再次同步并记录返回 tuple。

该诊断直接使用 framework loader 与真实 `gs_render`，不调用 Reconstruction，也不使用 mock renderer。未显式给出 pose config 时仍加载随代码提供的 V3.3 配置，以保持 raw-tail 40962 Skybox split 语义。可选的 `--include-use-bucket-false-ab` 仅增加两个 distance case 的诊断 A/B，不会修改 production adapter，也不能作为最终 workaround。

## 验证状态

- 原有回归 + V3.3 专项单元测试：`81 passed, 2 skipped`。新增 raw PLY storage、`gs_render` raw boundary、`gsplat` activated boundary、camera contract、frame manifest 和 backend comparison metrics 的回归测试；真实 CUDA rasterization contract 仍由上述独立进程 smoke test 验证。
- 新增覆盖：hole 禁止 over-nominal、center guard/crossing、Coverage Consensus、360° 半开采样、raw PLY tail 40962 skybox、elevation center-out ordering与 planar-depth 数值测试。私有 renderer contract 不再由 mock 单元测试替代，改由真实环境中的独立进程 smoke test 验证。
- V3.2 的 34-case 数字作为冻结基线记录于本文；本次代码环境未执行完整数据集批量渲染。正式接受 V3.3 前仍需在真实 `gs_render` 私有环境中验证 planar depth、geometry-only split、pano RGB/alpha/depth 对齐，并重跑 34-case 与高噪声专项 case。
