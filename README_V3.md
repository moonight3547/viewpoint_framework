# Viewpoint Framework V3.0

本版采用 2026-09-15 对齐后的**小步迭代方案**，优先用于 V2/V3.0 批量对照。
最初《V3 几何安全与视角生成设计》中的独立 view pitch、unsafe proposal repair、
固定高度 crossing 和 20% repair 位移上限没有进入 V3.0。

## 流程与公式

1. 使用 V2 scene understanding 的 center、坐标系和二分类主模式。
2. 沿用 V2 的 azimuth/elevation 范围和网格。elevation 始终表示**位置仰角**。
3. 有序 captured trajectory 按相邻位移跳变切分、空间重采样，投影到水平面；
   每个 azimuth 查询一次 `TrajectorySafeField`，得到局部水平半径 `rho`。
4. 若 `u_h` 是水平单位方向、`g` 是 scene frame 的 up、`e` 是位置仰角：

   ```text
   u  = cos(e) * u_h + sin(e) * g
   r0 = rho / cos(e)
   P0 = center + r0 * u
      = center + rho * u_h + rho * tan(e) * g
   ```

   因此同一 azimuth 的各 elevation 共用水平半径，高度随 elevation 改变。
   轨迹高度仅用于关闭状态的 height guard 雏形，未叠加到 `P0`，否则会改变指定仰角。
5. `P0` 未通过局部 clearance 检查时，直接 `UNSAFE_INITIAL_PRIOR`，不渲染、不修复。
6. 初始位置安全后，可用原有 gsplat reverse depth 进行 V2 风格的径向调整：
   outside-in 沿 `+u` 向外；inside-out 沿 `-u` 朝 center，可跨过三维 center。
7. 深度建议的最终绝对三维半径超过上限，整次调整取消，保留 `P0`。
8. 默认保留 V2 的 sampled path safety，并传入本候选的 clearance。
   路径裁剪后若落到 center 附近的下限半径内，回到已检查的初始位置。
9. 所有最终位置再次检查 clearance。最终方向严格按最终位置计算：
   outside-in `forward=normalize(center-P)`；inside-out `forward=normalize(P-center)`。

### 半径上限

`adjustment_radius_max=null` 使用 V2 `bbox.generation_radius` 的最大值（通常是
主模式 captured 三维半径 95 分位数），可设置正数覆盖，单位与场景一致。
上限控制**调整**，不裁剪轨迹提供的初始半径。
初始半径已超出调整上限时，通过几何检查后仍可保留，但不再执行 depth adjustment。
未知/无效深度保留初始位置；没有向 `radius_max` 回退的隐式扩张。

### Inside-out crossing

`P(t)=center+r(t)*u` 始终沿同一条三维径向线，crossing 经过真实 center，
相对高度随 signed radius 变化。越过中心后重新朝外，因此实际位置方向和光轴
相对初始径向方向翻转，azimuth 相差 180°，位置 elevation 反号。
输出的 `azimuth_deg/elevation_deg/direction` 描述最终径向方向，使 Stage3 的方向
判断与实际位置一致；原始采样方向保存在 `geometry_metadata.grid_*`。
`grid_id/row/col` 保留原始网格身份，`final_signed_radius` 仍相对于原始采样方向带符号。
crossing 保留可选能力，不表示每个网格都一定 crossing 或最终集合覆盖所有网格方向。

## TrajectorySafeField

新模块 `trajectory_safe_field.py` 每个场景构建一次，使用全部相机位置，保持 JSON
顺序。无效 pose 切断轨迹；非零相邻步长的 robust typical step 用于检测跳变；
默认超过 typical step 的 5 倍不插值。重复位置不导致 typical step 为零。
极短或稀疏序列的跳变判断证据不足，轨迹 tube 本身不构成自由空间证明。

每个 sample 提供 `(azimuth, rho, height)`，构造 `rho ± 0.04rho` 区间。
仅同一 branch 内有重叠的区间可以合并，不连接径向空白；preferred rho 使用角距
加权中位数。同一 azimuth 存在多个 interval 时选择置信度最高者，平局按原始
branch 顺序和半径稳定打破。所有区间保留在 metadata。

默认在 10° 内查直接支持，无直接支持时只允许最近支持在 30° 内的有限外推。
超过角距上限明确拒绝；没有 V2 全局半径 fallback。

## Candidate-local clearance

```text
median_rho = median(有效 captured 水平半径，不包含重采样点)
clearance  = clip(0.05 * rho, 0.01 * median_rho, 0.10 * median_rho)
```

三个系数均可配置，V3.0 不使用点云 AABB 或 legacy `clearance_abs` 决定阈值。
候选的初始、路径、最终检查使用同一个阈值。KNN 距离仍采用 V2 的 k=3 中位数，
KDTree 每个场景只建立一次。V3 必须提供有效点云并开启 `pointcloud_knn`。

`height_guard.enabled=false` 是 V3.0 默认。实验性开启后只拒绝超出所选轨迹
branch 高度带（加局部 margin）的候选，不改变位置或视角公式。
该开关会影响大仰角和 crossing 接受率，应在 V3.x 再单独消融。

## 调用方式

在 `viewpoint_framework` 的父目录执行（数据路径替换为实际路径）：

```bash
python -m viewpoint_framework.generate_poses \
  --cameras /data/train_cameras.json \
  --point_cloud /data/pi3_init_aligned.ply \
  --gaussian_ply /data/point_cloud_final.ply \
  --scene_config_json viewpoint_framework/configs/v2_scene_understanding.json \
  --config_json viewpoint_framework/configs/v3_pose_generation.json \
  --output_dir /output/v3_stage2
```

端到端 `run_pipeline` 使用相同 scene config，pose 参数名为 `--pose_config_json`。
已有 `--grid_gap` 仍同时设置 azimuth 和 positional elevation 步长。
V3 的局部 clearance 系数在 JSON 中修改；legacy `--safety_ratio/--safety_abs`
不控制 V3 阈值。无 GS 深度时加 `--no_gs_depth`，仍会执行初始和最终几何检查。

## 兼容性与诊断

- `position_strategy=legacy` 是默认，V1/V2 配置和算法行为保留。
- Stage1、`radius_field.py`、Stage3 算法和 18D `gen_cameras.json` 协议保持兼容。
- `view_limits.json` 保留 V2 endpoint 格式，它是展示约束，不用于裁剪 V3 初始 rho。
- `gen_cameras_meta.json` 在每个 candidate 增加 `geometry_metadata`，记录局部区间、
  source、branch、原始网格角度、初始/最终位置、阈值、调整建议/上限/取消原因、最终检查。
- `[S2:V3_CANDIDATE]` 和 `[S2:V3_FUNNEL]` 输出逐候选及漏斗统计。
- 拒绝原因区分无轨迹支持、角距过大、非法 elevation、初始碰撞、可选高度越界、
  最终碰撞、非法朝向。深度无效和调整越界是**保留 prior 的原因**，并不直接拒绝。

## 验证与边界

```bash
python -m pytest -q viewpoint_framework/tests
```

合成测试检查：rho/elevation 公式、任意 up 坐标、branch/跳变/角度接缝、远景点
不放大阈值、初始不安全直接拒绝、超上限不移动、穿三维中心后朝外、最终硬检查、
默认关闭 height guard、元数据到 Stage3 的内存/文件两种输入，以及 V2 回归。

V3.0 的局部 KNN 检查描述的是到已有点几何的距离，并不证明点在房间内部或物体外部。
点云空洞、玻璃漏建、稀疏薄结构仍可能漏检；height guard 默认关闭，
因此本版也不宣称消除了天花板/地板越界。真实 GS 场景需要批量渲染与人工反馈。
批测应分别观察 initial reject、radius-limit skip、final reject、crossing 数量和实际覆盖，
再决定 V3.1 是否引入高度限制、恢复搜索或更强的几何验证。

本机测试环境缺少 Open3D：合成测试仅在缺少该依赖时以精确 NumPy KNN 替代外部
树查询，实际的阈值、路径和 placement 代码照常执行。原生 Open3D 集成测试将标记
skip；未执行真实 GS/CUDA 批量渲染，运行环境仍需原项目的 Open3D/gsplat 依赖。
