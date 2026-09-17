# Viewpoint Framework V3.2

V3.2 是在 V3.1 的 skybox 分离、candidate-local clearance 和 crossing 安全检查之上的小步几何安全迭代。本版只替换两个核心环节：用连续 trajectory segment 决定每个方位的水平半径 `rho`，以及用 geometry-only Gaussian depth 建立 local/global height safety。V2 与 V3.1 仍走原代码入口，行为不变。

推荐配置：`configs/v3_2_pose_generation.json`。批量测试 Stage3 使用 `configs/stage3_v3_2_grid_only.json`，其中 `holes.max_views_per_hole=0`，避免 hole view 混入后干扰对 Stage2 grid view 的判断。

## 1. V3.1 批量结果、效果与剩余问题

V3.1 在 34 个场景的批量检查中发现 86 个异常视角：

- 41 个仅看到 skybox；
- 34 个接近或越过天花板；
- 3 个接近地板；
- 8 个由普通 Gaussian 几何遮挡导致。

其中 78/86（90.7%）属于垂直方向问题。V3.1 已经解决了 skybox 被当作实体碰撞物、candidate clearance 被全局场景尺度污染、inside-out crossing 路径不完整检查等问题，但其初始水平半径仍来自离散角邻域的 trajectory prior，且 positional elevation 未受到局部地板/天花板限制。因此，同一 `phi` 列可能从不合适的半径起步，较大的正/负 elevation 还会把相机放到天花板上方、地板下方或局部开洞区域。

V3.2 的目标不是引入完整 occupancy 或重新估计 scene center，而是先针对这两个主要误差源建立可解释、可回退、可批量统计的约束。

## 2. 修改思路与处理流程

```text
angular grid 的唯一 phi 列
        |
        v
连续 trajectory segment 与 phi ray 求交 -> rho
        | 无直接交点
        +----> V3.1 fallback（不删除该 phi）
        |
        v
轨迹覆盖内：在 Q(phi,rho) 上下做 geometry-only probe
外延 phi：跳过 local probe，直接采信 Global Height Limits
        |
        v
可靠 local limits 的严格交集 -> Global Height Limits
        |
        v
每个 theta 生成 raw initial
        |
        v
仅沿原 grid ray 缩小 |radius| 做 height clip
        |
        v
initial point-cloud hard safety
        |
        v
V3.1 depth adjustment / inside-out crossing
        |
        v
final height clip -> path safety -> final hard safety
```

高度修正从不改变 `phi/theta`，也不通过增大绝对半径把相机向外推；它只允许沿原始 grid ray 向 scene center 方向缩小 `|signed_radius|`。无法用这种方式到达安全高度区间时，initial candidate 被拒绝，final proposal 则回退到已经验证安全的 corrected initial。

## 3. Trajectory segment-ray 水平半径

实现位于 `trajectory_safe_field.py::query_segment_ray_min()`。

对方位角 `phi`，在 coordinate frame 的水平 `x/z` 平面构造从 scene center 出发的正向射线。只对原始相机序列中被判定为连续的相邻 edge 求线段/射线交点：

- 无效 pose 会断开 trajectory；
- 超过 `max_step_multiplier * typical_step` 的跳变 edge 不参与求交；
- 一个方位有多个正向交点时选择最小正 `rho`；
- 直接交点不使用 confidence threshold；
- 交点的插值轨迹高度和 edge 两端高度同时写入 metadata，作为 local height probe 的上下 anchor。

若没有直接交点，不丢弃该 `phi`，而是使用 V3.1 interval fallback：

1. 在 `confidence >= fallback_confidence_threshold` 的 interval 中选最小 `rho`；
2. 若全部低置信，先选最高 confidence，再以较小 `rho` 和 branch id 做稳定 tie-break；
3. 记录 `LOW_CONFIDENCE_TRAJECTORY_FALLBACK`，供批量分析。

这样不会因为 trajectory 的采样密度或角度离散造成方位覆盖塌缩。

## 4. Local Height Limits

实现位于 `height_safety.py`。每个轨迹覆盖范围内且有可用 `rho` 的唯一 `phi` 只 probe 一次，所有该列的 elevation grid points 复用结果。

设水平 anchor 为：

```text
Q = scene_center + rho * horizontal_direction(phi)
```

上探相机放在该 trajectory segment 的最高 anchor，朝 scene up；下探相机放在最低 anchor，朝 `-scene_up`。默认用 64 像素最长边和中央 50% crop 调用 `render_geometry_depth()`，因此 skybox Gaussian 不参与深度、hole、地板或天花板判断。

每个有效像素按相机内外参反投影到世界坐标，并投影到 scene up 得到真实世界高度。上界取严格高于 seed 的最近 hit，下界取严格低于 seed 的最近 hit。灯具、横梁等局部障碍也会作为有效限制，而不是只拟合一个理想平面。

raw limit 使用与该 `rho` 对应的 candidate-local clearance 收缩：

```text
upper_safe = upper_raw - local_clearance
lower_safe = lower_raw + local_clearance
```

每一侧独立标记：

- `LOCAL_RELIABLE`：检测到 limit，且不是“中心无效并且 hole ratio > 0.5”；
- `LOCAL_HOLE_UNCERTAIN`：检测到 limit，但中央区域和 hole 统计表明该 probe 可能穿过空洞；
- `LOCAL_UNAVAILABLE`：没有该方向的可用 hit。

V3.2 不做相邻 `phi` 的 local height 插值。

## 5. Global Height Limits 与外延 phi

Global 下界取所有非外延、可靠 local 下界中的最大值；Global 上界取所有非外延、可靠 local 上界中的最小值。两侧独立回退：某一侧没有可靠 local limit 时，仅该侧使用 captured trajectory height 的 min/max。

若 strict intersection 得到 `height_min >= height_max`，认为全局区间冲突，整段回退到 captured height range，同时输出 warning，并在 metadata 中保留冲突标记和原 contributor 方位。

用户确认的 angular bbox 外延规则在本版显式实现：

- `bbox.observed_azimuth` 内的列才运行并采信 local probe；
- generation bbox 外延产生的列标记 `is_extension_column=true`；
- extension column 不参与 Global limits 的建立；
- extension column 的 initial 与同侧 non-crossing final 均直接使用 Global Height Limits。

普通列的 effective limits：可靠侧用 local；hole-uncertain 侧取 local/global 中更严格者；unavailable 侧用 global。Inside-out crossing 到对侧以后统一使用 Global Height Limits，不错误复用原方位的 local 天花板/地板。

## 6. Initial、final 与 crossing 安全规则

Initial 流程：

1. 用 `azimuth + rho + positional elevation` 得到 raw initial；
2. 若高度越界，只缩小正半径；
3. 无法到达高度区间则 `INITIAL_HEIGHT_LIMIT_UNREACHABLE`；
4. 对 corrected initial 运行 point-cloud hard safety；
5. height-corrected point 发生碰撞则 `INITIAL_HEIGHT_CLIP_GEOMETRY_COLLISION`，直接拒绝，不做 rescue。

Final 流程：

1. 保留 V3.1 的 outside depth backoff、inside same-side inward 与 center crossing proposal；
2. outside 向外增大半径受 `adjustment_radius_max` 限制；
3. inside crossing 先允许向内穿过中心；只有到对侧后继续增大绝对半径时才应用上限，必要时截到 `-radius_max`；
4. same-side final 使用列 effective limits，crossing final 使用 global limits；
5. final height clip 后继续执行 path safety 和 endpoint hard safety；
6. final height 无法达到或 height-corrected final 碰撞时回退 corrected initial；
7. crossing 全路径不安全或终点不安全时仍回退 corrected initial，并在 summary 中显式统计。

V3.2 的最终朝向严格区分三种运动语义：

- outside-in 向外移动后仍面向 scene center：`camera_forward = -position_direction`；
- inside-out non-crossing 向 scene center 移动，但保持背向 scene center：`camera_forward = grid_direction = position_direction`；
- inside-out crossing 穿过中心并在对侧向外移动后，仍保持原始 `grid_direction`，此时 `grid_direction = -position_direction`，因此相机面向 scene center。

也就是说，inside-out 的朝向保持原始 grid direction，而不是按 crossing 后的最终径向方向重新背离中心。V2/V3.1 的旧入口和旧行为没有改变。Crossing final 暂时仍使用 Global Height Limits；即使其最终位置理论上可以重新计算 local height，也不在 V3.2 中启用，以避免场景空洞造成错误 local limit。

## 7. 配置

V3.2 新增配置段：

```json
{
  "trajectory_safe_field": {
    "rho_strategy": "segment_ray_min",
    "fallback_strategy": "v3_1",
    "fallback_confidence_threshold": 0.2
  },
  "local_height": {
    "enabled": true,
    "strategy": "geometry_depth_up_down",
    "probe_resolution": 64,
    "central_crop_ratio": 0.5,
    "alpha_threshold": 0.05,
    "hole_ratio_threshold": 0.5,
    "center_patch_size": 3,
    "center_min_valid_ratio": 0.5,
    "use_nearest_depth": true
  },
  "global_height": {
    "enabled": true,
    "strategy": "strict_local_intersection",
    "fallback": "captured_height_range"
  }
}
```

这些开关在 V3.2 配置中必须启用。旧配置缺少它们时默认关闭，因此不会意外改变 V2/V3/V3.1。

## 8. Metadata 与诊断统计

`gen_cameras_meta.json` 新增/扩展：

- `placement.scene_center` 与 `placement.coordinate_frame`；
- 每个 `trajectory_columns` 的 rho source、直接交点数、selected edge/t、cross height、fallback confidence 和 extension 标记；
- 每个 `local_height_columns` 的 raw/safe 上下界、valid/hole/center ratios 与可靠性状态；
- `global_height` 的上下界、来源、contributor、独立 fallback 与 conflict；
- candidate 的 raw/corrected initial、有效高度来源、两次 height clip、proposal/final、回退原因和最终方向。

`diagnostics` 汇总 direct/fallback/low-confidence rho 列数、extension 列数、local 各状态计数、global fallback/conflict、initial/final clip、unreachable、碰撞回退、crossing 与 rejection reasons。

控制台逐 grid 输出仍默认关闭：`console_log_candidates=false`。

## 9. View generalization 可视化

新增 `visualize_candidate_placements.py`。颜色固定为：

- raw initial：黄色 `#A78324`；
- rejected/conflict：红色 `#913F3F`；
- final 未选择：蓝色 `#315D86`；
- Stage3 selected：绿色 `#347054`。

点云 RGB 会轻度白化，默认 opacity 为 0.42。图中包含 final camera center、完整 frustum、forward line、raw/corrected initial 到 final 的连线、height correction line、Global Height planes；使用 `--show_local_limits` 时增加每个方位的 local vertical segment。hover 显示 grid 角度、rho source、extension、clearance、高度来源、clip、adjustment 和 rejection 信息。

```bash
python -m viewpoint_framework.visualize_candidate_placements \
  --point_cloud aligned_points.ply \
  --metadata output/gen_cameras_meta.json \
  --stage3_metadata stage3_output/debug/stage3_metadata.json \
  --show_local_limits \
  --output output/candidate_placements/index.html \
  --no_serve
```

`--stage3_metadata` 可省略，此时所有有效 final candidate 都显示为蓝色。

## 10. 运行与验证

Stage2：

```bash
python -m viewpoint_framework.generate_poses \
  --cameras train_cameras.json \
  --point_cloud aligned_points.ply \
  --gaussian_ply point_cloud_final.ply \
  --config_json viewpoint_framework/configs/v3_2_pose_generation.json \
  --output_dir output
```

Stage3 批量检查使用 `configs/stage3_v3_2_grid_only.json`。本次自动化验证结果为 `68 passed, 1 skipped`；skip 项是当前基础环境没有 Open3D 时的 native Open3D 查询，测试套件仍用精确 NumPy KNN adapter 覆盖相同的 clearance、path 和 final safety 生产逻辑。

新增测试覆盖：连续 segment/ray 的最小正交点、jump edge 不桥接、低置信 fallback、extension 不参与 global、global 两侧独立 fallback、global conflict、hole-uncertain effective limits、正负 signed radius 仅向内 height clip、不可达高度，以及 outside-in / inside-out non-crossing / inside-out crossing 三种最终朝向语义。

## 11. V3.2 暂不处理

本版不修改 scene center 算法，不加入 voxel occupancy、独立 view pitch、neighbor-phi height interpolation、point-cloud/GS height fusion、skybox sanity rejection 或复杂的 height rescue。`height_guard` 雏形仍保持关闭。这些策略留待 V3.2 批量结果明确剩余 failure mode 后再进入 V3.x 小步迭代。
