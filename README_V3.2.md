# Viewpoint Framework V3.2

V3.2 是当前保留的稳定 placement/height-safety 基线。配置入口：
`configs/v3_2_pose_generation.json`。

## 生成链路

```text
captured trajectory
  -> binary dominant mode/bbox
  -> uniform azimuth/elevation grid
  -> segment-ray horizontal rho
  -> point-cloud local clearance/path safety
  -> geometry-only Gaussian up/down depth
  -> local/global height clip
  -> reverse-depth radial adjustment
  -> final geometry and height validation
```

### 水平半径

`TrajectorySafeField.query_segment_ray_min(phi)` 将采集轨迹按顺序连接成
segments，在水平 XZ 平面与目标方位射线求交，并选取最小正交点作为该方位的
`rho`。无直接交点时使用受角度与置信度限制的 trajectory interval fallback；
不再回退到全局 Gaussian/point-cloud radius heuristic。

### 垂直范围

每个可靠方位列从水平轨迹端点向上、向下执行 geometry-only Gaussian depth
probe，得到 Local Height Limits。Global Height Limits 使用可靠 local columns
的严格交集，并在样本不足时退到 captured camera height range。

- initial candidate：优先 local limits，空洞/不可用列用 global limits；
- crossing final：使用 global limits，避免在场景空洞处误信 local probe；
- height clip 通过改变 radial distance 实现，不改变该 grid point 的 elevation
  语义。

### 朝向

- outside-in：始终看向 scene center；
- inside-out：保留原始 grid direction；越过中心后自然表现为看向中心。

## 必要约束

V3.2 要求：

- `geometry.strategy=pointcloud_knn`
- `geometry.clearance_strategy=local_horizontal_radius`
- `trajectory_safe_field.rho_strategy=segment_ray_min`
- `local_height.enabled=true`
- `global_height.enabled=true`
- depth probe 持有支持 `render_geometry_depth` 的 Gaussian renderer

缺少上述任一条件会直接报错，不使用静默 fallback。

## 渲染与 skybox

V3.2 只使用 gsplat backend。Skybox radial-band 检测把 shell Gaussian 从
geometry depth、visibility 与 collision sampling 中排除，但 full RGB 仍包含
skybox。Renderer near plane 由 captured trajectory radius 比例推导。

## 输出诊断

`gen_cameras_meta.json` 为每个 candidate 保存：

- trajectory support/source/confidence；
- initial/final signed radius；
- local/global/effective height limits 与 clip 原因；
- initial/final clearance 与 path-safety；
- depth-probe 结果、status 和 reject reason。

这些字段是 V3.2 与 V3.3 placement 对照的主要 debug contract。
