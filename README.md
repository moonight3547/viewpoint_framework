# Viewpoint Framework

V3.4 的可选竖屏输出实验见 [README_V3.4.md](README_V3.4.md)。V3.2/V3.3
位姿策略仍分别通过对应的 pose config 直接调用；V3.4 当前不新增位姿版本。

当前只维护 V3.2 和 V3.3 两套 Stage 2 行为。默认运行 V3.3：使用 V3.3
azimuth × elevation 网格，以及经过验证的 V3.2 placement/height safety。原生
V3.3 placement 仍可通过配置启用，便于继续实验。

## 数据流

```text
captured cameras + aligned point cloud + Gaussian PLY
    -> Stage 1: robust scene understanding
    -> Stage 2: V3.2/V3.3 candidate generation
    -> Stage 3: output-all or selection/reference/render
```

主要入口：

- `python -m viewpoint_framework.run_pipeline`：完整 Stage 1–3。
- `python -m viewpoint_framework.generate_poses`：只运行 Stage 1–2。
- `python -m viewpoint_framework.renderer.compare_sequence_outputs`：逐帧比较两个后端。
- `run_viewpoint_pipeline.sh`：服务器批处理包装。

## 当前配置

```text
configs/
├── stage1_scene_understanding.json
├── v3_2_pose_generation.json
├── v3_3_pose_generation.json
├── stage3_v3_2_grid_only.json
├── stage3_v3_3.json
├── stage3_gaussian_coverage.json
├── stage3_pointcloud_gap.json
└── stage3_artifixer_refs.json
```

不传配置时，CLI 使用：

- Stage 1：`stage1_scene_understanding.json`
- Stage 2：`v3_3_pose_generation.json`
- Stage 3：`stage3_v3_3.json`

V3.2 使用 `pose_generation_v32.py`。V3.3 的网格与原生 placement 位于
`stage2/`；`v33_placement_strategy` 可取：

- `v3_2`：V3.3 网格 + V3.2 placement（当前默认）。
- `v3_3`：V3.3 网格 + V3.3 placement。

## 目录职责

```text
scene_analysis.py / scene_understanding.py / view_space.py / radius_field.py
    Stage 1 robust scene model

pose_generation.py
    公共数据协议、bbox/grid、V3.2/V3.3 dispatch

pose_generation_v32.py
    V3.2 placement 与 local/global height safety

stage2/
    V3.3 angular grid、placement primitives、原生 V3.3 pipeline

renderer/
    backend factory、gs_render adapter、输出比较工具

stage3/
    selection、holes、references、render output

utils/
    camera protocol、geometry、point-cloud IO、HTML/server runtime

visualization/
    camera、candidate placement、scene-analysis visualization CLI
```

## 关键渲染 contract

Canonical Gaussian scene 始终保存 PLY/raw optimization representation：

- quaternion：raw quaternion
- scale：log-scale
- opacity：logit

`gs_render` 接收 raw 值；`gsplat` adapter 在自身边界执行 quaternion
normalize、`exp(scale)` 和 `sigmoid(opacity)`。Skybox 检测若需要实际
scale/opacity，只能局部激活，不得修改 canonical storage。

相机 JSON 使用统一 18-D 格式：
`[fx, fy, cx, cy, width, height, w2c[:3, :4].flatten()]`。V3.3 默认保留输入
相机内参，不隐式缩放 focal，也不强制主点居中。

## 示例

```bash
python -m viewpoint_framework.run_pipeline \
  --cameras train_cameras.json \
  --point_cloud pi3_init_aligned.ply \
  --gaussian_ply point_cloud_final.ply \
  --output_dir outputs/v3_3
```

切换 V3.2：

```bash
python -m viewpoint_framework.run_pipeline \
  --cameras train_cameras.json \
  --point_cloud pi3_init_aligned.ply \
  --gaussian_ply point_cloud_final.ply \
  --pose_config_json viewpoint_framework/configs/v3_2_pose_generation.json \
  --stage3_config_json viewpoint_framework/configs/stage3_v3_2_grid_only.json \
  --output_dir outputs/v3_2
```

更详细的版本语义见 [README_V3.2.md](README_V3.2.md) 和
[README_V3.3.md](README_V3.3.md)。
