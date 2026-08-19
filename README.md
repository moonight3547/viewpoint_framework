# Viewpoint Framework

面向 3D 场景重建的**视角约束、视角生成与相机位姿可视化框架**。

当前版本首先完成基础数据协议与交互式可视化能力：

* 统一相机参数格式；
* 统一 `w2c / c2w` 相机表示；
* 加载 PLY 点云；
* 同时显示采集相机与生成相机；
* 支持采集相机降采样；
* 根据场景尺度与相机密度自适应调整相机可视化尺寸；
* 导出独立交互式 HTML；
* 启动 localhost HTTP Server；
* 浏览器中自由旋转、缩放和平移；
* `Ctrl-C` 结束可视化服务。

后续将在此基础上扩展：

* 场景中心估计；
* Outside-In / Inside-Out 模式判断；
* 视角空间约束；
* 球壳视角参数化；
* 新视角生成；
* 位姿过滤与覆盖度评估；
* 点云碰撞与可见性约束。

---

# 1. Repository Structure

当前基础仓库结构：

```text
viewpoint_framework/
├── geometry_util.py
├── cameras_util.py
├── points_util.py
├── util.py
├── visualize_cameras.py
├── requirements.txt
└── README.md
```

各模块职责：

```text
geometry_util.py
│
├── 向量归一化
├── 向量夹角
├── 齐次坐标转换
├── 3D 点坐标变换
├── 方向向量变换
├── Pinhole Camera 几何
├── Camera Frustum 几何
├── Bounding Box 几何
└── Camera 可视化尺度估计


cameras_util.py
│
├── Camera 数据结构
├── 18 维相机协议解析
├── w2c / c2w
├── 相机内参 K
├── camera position
├── right / down / forward
├── Camera JSON 加载
├── Camera 序列降采样
└── Camera position 提取


points_util.py
│
├── PointCloudData 数据结构
├── PLY 加载
├── NaN / Inf 清理
├── 点云 RGB 处理
├── 点云随机降采样
├── AABB center
├── AABB extent
└── AABB diagonal


util.py
│
├── HTML 输出路径管理
├── Local HTTP Server
├── localhost URL
├── 浏览器拉起
└── Ctrl-C Server Shutdown


visualize_cameras.py
│
├── 加载点云
├── 加载 Captured Cameras
├── 加载 Generated Cameras
├── Captured Camera 降采样
├── Camera Frustum 自适应缩放
├── Plotly WebGL 可视化
├── Standalone HTML 导出
└── localhost Server
```

---

# 2. Dependencies

当前依赖：

```text
numpy
open3d
plotly
```

推荐创建：

```text
requirements.txt
```

内容：

```txt
numpy
open3d
plotly
```

安装：

```bash
pip install -r requirements.txt
```

---

# 3. Camera Data Convention

## 3.1 JSON 格式

框架统一约定相机参数存储为一个 JSON 文件。

JSON 顶层是一个 `list`：

```json
[
    [... camera 0 ...],
    [... camera 1 ...],
    [... camera 2 ...]
]
```

每个 Camera 是一个长度为 `18` 的 list：

```text
[
    fx,
    fy,
    cx,
    cy,
    width,
    height,

    w2c_00,
    w2c_01,
    w2c_02,
    w2c_03,

    w2c_10,
    w2c_11,
    w2c_12,
    w2c_13,

    w2c_20,
    w2c_21,
    w2c_22,
    w2c_23
]
```

即：

```python
camera[:6] = [
    fx,
    fy,
    cx,
    cy,
    width,
    height,
]

camera[6:18] = w2c[:3, :4].flatten()
```

---

## 3.2 内参约定

前 6 个值：

```text
fx
fy
cx
cy
width
height
```

对应标准 pinhole camera intrinsic：

```text
        fx   0   cx
K =      0  fy   cy
         0   0    1
```

---

## 3.3 外参约定

JSON 中存储：

```text
w2c
```

即：

```text
World → Camera
```

JSON 只保存 `w2c` 的前三行：

```text
3 × 4
```

读取之后恢复成：

```python
w2c = np.eye(4)

w2c[:3, :4] = values[6:18].reshape(3, 4)
```

内部同时计算：

```python
c2w = np.linalg.inv(w2c)
```

---

# 4. Camera Coordinate Convention

当前框架统一采用：

```text
Camera Local Coordinate

+X : image right
+Y : image down
+Z : camera forward
```

因此对于 `c2w`：

```python
camera.position
```

表示世界坐标中的 Camera Center：

```python
camera.c2w[:3, 3]
```

相机三个轴在世界坐标中的方向分别为：

```python
camera.right
```

对应：

```python
camera.c2w[:3, 0]
```

---

```python
camera.down
```

对应：

```python
camera.c2w[:3, 1]
```

---

```python
camera.forward
```

对应：

```python
camera.c2w[:3, 2]
```

即相机看向：

```text
+Z
```

方向。

---

# 5. Important Coordinate-System Principle

基础 Camera Parser **不会隐式进行任何坐标修正**。

不会自动：

```text
flip X
flip Y
flip Z
```

也不会自动：

```text
旋转 180°
```

或者：

```text
修复 det(R) < 0
```

如果：

```python
det(R)
```

明显偏离：

```text
+1
```

框架只会给出 warning。

例如：

```text
Camera 5: det(R)=-1.000000.
A rigid rotation normally has det(R) ~= +1.
The matrix will NOT be modified.
```

这是刻意设计的。

原因是可视化模块本身需要帮助检查：

* 坐标系错误；
* 手性问题；
* 相机翻转；
* forward 方向错误；
* w2c/c2w 使用错误。

如果 Parser 自动修复，会隐藏真正的数据问题。

---

# 6. Point Cloud Convention

当前场景几何输入使用：

```text
PLY
```

格式。

例如：

```text
scene.ply
```

通过：

```python
open3d.io.read_point_cloud()
```

加载。

内部使用：

```python
PointCloudData
```

表示：

```python
PointCloudData(
    points: [N, 3],
    colors: [N, 3] or None
)
```

其中：

```text
points
```

为世界坐标中的 XYZ。

如果 PLY 包含颜色：

```text
colors ∈ [0, 1]
```

---

# 7. Point Cloud Preprocessing

加载过程中会自动清理：

```text
NaN
Inf
-Inf
```

几何点。

可视化时支持随机降采样，例如：

```text
1,000,000 points
        ↓
100,000 points
```

该降采样**仅用于浏览器可视化**：

```text
不会修改原始 PLY
不会影响后续几何算法输入
```

默认：

```text
max_points = 100000
```

---

# 8. Camera Types

当前可视化支持两组 Camera。

## Captured Cameras

表示真实数据采集过程中获得的相机位姿。

例如：

```text
train_cameras.json
```

由于采集帧可能非常多，因此支持降采样：

```python
captured_cameras[::stride]
```

默认：

```text
captured_stride = 4
```

即：

```text
Camera 0
Camera 4
Camera 8
Camera 12
...
```

用于可视化。

Camera 的原始：

```python
camera.index
```

不会改变。

---

## Generated Cameras

表示视角生成算法产生的新相机位姿。

例如：

```text
generated_cameras.json
```

Generated Cameras 默认：

```text
全部显示
```

不进行降采样。

---

# 9. Visualization

可视化采用：

```text
Plotly WebGL
```

浏览器中同时显示：

```text
Point Cloud
+
Captured Camera Centers
+
Captured Camera Frustums
+
Captured Camera Trajectory
+
Generated Camera Centers
+
Generated Camera Frustums
+
Generated Camera Trajectory
```

默认颜色：

```text
Captured Cameras
    Blue

Generated Cameras
    Orange / Red
```

---

# 10. Camera Frustum

Camera Frustum 根据真实内参计算。

对于图像平面像素：

```text
(u, v)
```

在 Camera Coordinate 中：

```text
x = (u - cx) / fx × depth

y = (v - cy) / fy × depth

z = depth
```

使用：

```text
top-left
top-right
bottom-right
bottom-left
```

四个角构成 Camera Frustum。

随后通过：

```python
c2w
```

转换到世界坐标。

因此 Camera 可视化不仅表示：

```text
Camera Position
```

也能体现：

```text
Camera Orientation
Camera FOV
Camera Aspect Ratio
```

---

# 11. Adaptive Camera Visualization Scale

大量采集 Camera 非常密集时，如果 Camera Frustum 仅根据场景大小确定，会产生：

```text
大量 Camera 相互交叉
Frustum 遮挡
难以观察轨迹
难以区分 Generated Camera
```

因此当前框架使用：

```text
场景尺度
+
相机空间密度
```

共同决定 Camera 可视化尺寸。

---

## 11.1 Scene Scale

首先计算 PLY 的 AABB：

```text
min_bound
max_bound
```

得到：

```text
scene_diagonal
```

即：

```text
AABB diagonal
```

场景尺度候选值：

```text
scene_based_depth
    =
scene_diagonal × scene_scale
```

默认：

```text
scene_scale = 0.015
```

即：

```text
Camera Frustum ≈ 场景对角线的 1.5%
```

---

## 11.2 Camera Density

对于所有当前显示的 Camera Center：

```text
Captured
+
Generated
```

计算每个 Camera 的：

```text
nearest-neighbor distance
```

然后取：

```text
median nearest-camera distance
```

得到典型相机间距：

```text
typical_spacing
```

Camera 密度限制：

```text
spacing_based_depth
    =
typical_spacing × spacing_scale
```

默认：

```text
spacing_scale = 0.35
```

---

## 11.3 Final Camera Size

最终：

```text
frustum_depth
    =
min(
    scene_based_depth,
    spacing_based_depth
)
```

并限制：

```text
min_scene_scale
≤
frustum_depth / scene_diagonal
≤
max_scene_scale
```

默认：

```text
frustum_min_scale = 0.002
frustum_max_scale = 0.02
```

因此：

```text
相机稀疏
    ↓
主要由 Scene Scale 决定


相机密集
    ↓
Camera Frustum 自动缩小
```

从而避免密集 Camera 大面积相互穿插。

---

# 12. Basic Usage

完整调用：

```bash
python visualize_cameras.py \
    --point_cloud scene.ply \
    --captured_cameras train_cameras.json \
    --generated_cameras generated_cameras.json \
    --output outputs/camera_vis/index.html
```

运行后会输出类似：

```text
==========================================================
 Visualization Server
==========================================================

 HTML:
   /path/to/outputs/camera_vis/index.html

 URL:
   http://localhost:8000/index.html

 Controls:
   Left drag     : rotate
   Right drag    : pan
   Mouse wheel   : zoom
   Click legend  : show / hide group

 Press Ctrl-C to stop.
==========================================================
```

浏览器打开：

```text
http://localhost:8000/index.html
```

---

# 13. Browser Interaction

浏览器端支持：

```text
Left Mouse Drag
    Rotate

Right Mouse Drag
    Pan

Mouse Wheel
    Zoom

Legend Click
    Show / Hide Camera Group
```

Plotly 自带 Mode Bar，可进行：

```text
Reset Camera
Zoom
Pan
Orbit
Save Screenshot
```

等操作。

---

# 14. Stop Visualization Server

Server 会一直阻塞运行。

结束：

```text
Ctrl-C
```

随后：

```text
[Server] Ctrl-C received.
[Server] Server stopped.
```

HTTP Server 关闭。

但是生成的：

```text
index.html
```

不会删除。

---

# 15. Standalone HTML

HTML 使用：

```python
include_plotlyjs=True
```

因此 Plotly JS 会直接嵌入 HTML。

输出是：

```text
Standalone HTML
```

Server 结束之后仍然可以单独打开：

```text
outputs/camera_vis/index.html
```

---

# 16. Generate HTML Without Server

如果只希望保存 HTML：

```bash
python visualize_cameras.py \
    --point_cloud scene.ply \
    --captured_cameras train_cameras.json \
    --generated_cameras generated_cameras.json \
    --output outputs/camera_vis/index.html \
    --no_serve
```

---

# 17. Captured Camera Downsampling

默认：

```text
stride = 4
```

修改为：

```bash
--captured_stride 8
```

例如：

```bash
python visualize_cameras.py \
    --point_cloud scene.ply \
    --captured_cameras train_cameras.json \
    --captured_stride 8
```

则：

```text
Camera 0
Camera 8
Camera 16
Camera 24
...
```

参与显示。

---

# 18. Point Cloud Visualization Parameters

限制显示点数量：

```bash
--max_points 200000
```

关闭降采样：

```bash
--max_points 0
```

调整 Point Size：

```bash
--point_size 1.0
```

调整透明度：

```bash
--point_opacity 0.5
```

例如：

```bash
python visualize_cameras.py \
    --point_cloud scene.ply \
    --captured_cameras train_cameras.json \
    --max_points 200000 \
    --point_size 1.0 \
    --point_opacity 0.6
```

---

# 19. Camera Visualization Parameters

默认：

```text
frustum_scale         = 0.015
frustum_spacing_scale = 0.35
frustum_min_scale     = 0.002
frustum_max_scale     = 0.02
```

如果仍然感觉 Camera 太大：

```bash
--frustum_scale 0.01 \
--frustum_spacing_scale 0.25
```

如果 Camera 太小：

```bash
--frustum_scale 0.02 \
--frustum_spacing_scale 0.5
```

一般建议优先调整：

```text
frustum_spacing_scale
```

因为它直接控制密集轨迹中的 Camera 相对间距。

---

# 20. HTTP Server Parameters

默认：

```text
host = 127.0.0.1
port = 8000
```

修改端口：

```bash
--port 8080
```

如果希望操作系统自动分配空闲端口：

```bash
--port 0
```

---

# 21. Remote GPU Server

如果程序运行在远程服务器：

```text
GPU Server
```

而浏览器位于本地电脑，需要注意：

服务器：

```text
127.0.0.1
```

并不是本机：

```text
127.0.0.1
```

推荐使用 SSH Port Forward。

例如服务器：

```text
10.50.121.201
```

本地执行：

```bash
ssh -L 8000:127.0.0.1:8000 user@10.50.121.201
```

服务器运行：

```bash
python visualize_cameras.py \
    --point_cloud scene.ply \
    --captured_cameras train_cameras.json \
    --generated_cameras generated_cameras.json \
    --port 8000
```

本地浏览器访问：

```text
http://localhost:8000/index.html
```

数据链路：

```text
Local Browser
     │
     │ localhost:8000
     ▼
SSH Tunnel
     │
     ▼
Remote Server
127.0.0.1:8000
     │
     ▼
index.html
```

---

# 22. Framework Data Flow

当前完整数据流：

```text
                 Camera JSON
                     │
                     │ 18-D
                     ▼
             cameras_util.py
                     │
                     │
             ┌───────┴────────┐
             │                │
             ▼                ▼
            w2c              c2w
                              │
                   ┌──────────┼───────────┐
                   │          │           │
                   ▼          ▼           ▼
               position    forward    right/down


                   PLY Point Cloud
                         │
                         ▼
                   points_util.py
                         │
               ┌─────────┼───────────┐
               │         │           │
               ▼         ▼           ▼
             points    colors      AABB
                                     │
                             ┌───────┼───────┐
                             ▼       ▼       ▼
                          center   extent  diagonal


 Camera Geometry                        Scene Geometry
       │                                      │
       └────────────────┬─────────────────────┘
                        ▼
                geometry_util.py
                        │
              ┌─────────┼─────────┐
              │         │         │
              ▼         ▼         ▼
           Frustum    Scale     Vector
           Geometry   Estimate  Geometry
              │
              └─────────┬─────────┘
                        ▼
              visualize_cameras.py
                        │
               Plotly WebGL Figure
                        │
                        ▼
                    index.html
                        │
               ┌────────┴────────┐
               │                 │
               ▼                 ▼
         standalone file      util.py
                                  │
                                  ▼
                           Local HTTP Server
                                  │
                                  ▼
                     http://localhost:8000
```

---

# 23. Core Architecture Principle

框架设计遵循四层结构。

## Layer 1 — Geometry

```text
geometry_util.py
```

只处理纯数学几何。

要求：

```text
不依赖 Camera
不依赖 PointCloud
不依赖 Open3D
不依赖 Plotly
```

例如：

```python
normalize_vector()
angle_between_vectors()
transform_points()
build_camera_frustum_world()
```

---

## Layer 2 — Data Representation

```text
cameras_util.py
points_util.py
```

负责：

```text
外部数据
    ↓
统一内部表示
```

Camera：

```text
JSON
 ↓
Camera
```

Point Cloud：

```text
PLY
 ↓
PointCloudData
```

上层算法不应该直接处理：

```text
18-D array
```

而应该使用：

```python
camera.position
camera.forward
camera.c2w
```

---

## Layer 3 — Algorithm

未来加入：

```text
scene_center.py
view_constraints.py
view_generator.py
```

算法层只依赖：

```text
Camera
PointCloudData
geometry_util
```

不应该依赖：

```text
Plotly
HTTP Server
HTML
```

---

## Layer 4 — Visualization

```text
visualize_cameras.py
```

只负责：

```text
Geometry / Data
       ↓
Visual Representation
```

不会反向影响算法。

---

# 24. Planned Framework

后续建议仓库逐渐扩展为：

```text
viewpoint_framework/
│
├── geometry_util.py
├── cameras_util.py
├── points_util.py
├── util.py
│
├── scene_center.py
├── view_constraints.py
├── view_generator.py
│
├── visualize_cameras.py
│
├── main.py
│
├── requirements.txt
└── README.md
```

之后如果功能继续增长，可以进一步拆成：

```text
viewpoint_framework/
│
├── utils/
│   ├── geometry_util.py
│   ├── cameras_util.py
│   ├── points_util.py
│   └── util.py
│
├── constraints/
│   ├── base.py
│   ├── spherical_constraint.py
│   ├── distance_constraint.py
│   ├── angle_constraint.py
│   ├── visibility_constraint.py
│   └── collision_constraint.py
│
├── generators/
│   ├── base.py
│   ├── spherical_generator.py
│   └── interpolation_generator.py
│
├── visualization/
│   └── visualize_cameras.py
│
└── main.py
```

---

# 25. Planned Viewpoint Pipeline

目标框架整体通路：

```text
Captured Camera Poses
        +
     Point Cloud
        │
        ▼
┌────────────────────┐
│ Scene Understanding│
└─────────┬──────────┘
          │
          ├── Scene Center
          ├── Camera Distribution
          ├── Scene Scale
          └── Capture Mode
                  │
                  ├── Outside-In
                  └── Inside-Out
          │
          ▼
┌────────────────────┐
│ View Constraints   │
└─────────┬──────────┘
          │
          ├── Position Range
          ├── Radius Range
          ├── Azimuth Range
          ├── Elevation Range
          ├── Orientation
          ├── Point-cloud Distance
          ├── Visibility
          └── Collision
          │
          ▼
┌────────────────────┐
│ Candidate Generator│
└─────────┬──────────┘
          │
          ▼
    Candidate Cameras
          │
          ▼
┌────────────────────┐
│ Candidate Filtering│
└─────────┬──────────┘
          │
          ├── Constraint Check
          ├── Existing-view Distance
          ├── Angular Difference
          └── Coverage Evaluation
          │
          ▼
   Generated Cameras
          │
          ▼
  generated_cameras.json
          │
          ▼
┌────────────────────────┐
│ visualize_cameras.py   │
└────────────────────────┘
```

---

# 26. Current Scope

当前仓库已经完成：

```text
[✓] Camera 数据协议
[✓] Camera JSON Parser
[✓] w2c / c2w
[✓] Camera Coordinate Convention
[✓] Point Cloud Loader
[✓] Point Cloud Visualization Sampling
[✓] Camera Frustum Geometry
[✓] Adaptive Camera Visualization Size
[✓] Captured / Generated Camera Visualization
[✓] Interactive HTML
[✓] Local HTTP Server
[✓] Ctrl-C Shutdown
```

待实现：

```text
[ ] Scene Center Estimation
[ ] Outside-In / Inside-Out Detection
[ ] Spherical Camera Parameterization
[ ] View Range Estimation
[ ] Candidate Camera Generation
[ ] Camera Constraint System
[ ] Point-cloud Collision Constraint
[ ] Visibility Constraint
[ ] View Coverage Evaluation
[ ] Final Novel-view Selection
```

---

# 27. Basic Development Rule

后续开发时建议遵循：

```text
Data parsing
    → cameras_util / points_util

Pure mathematics
    → geometry_util

General infrastructure
    → util

Viewpoint algorithms
    → algorithm modules

Rendering / debugging
    → visualization modules
```

避免出现：

```text
view_generator.py
    import plotly
```

或者：

```text
geometry_util.py
    import Camera
```

等反向依赖。

最终目标是保证：

```text
核心视角生成算法
```

可以完全脱离：

```text
HTML / Plotly / HTTP Server
```

单独运行。

---

# 28. Example

最典型调试流程：

```bash
python visualize_cameras.py \
    --point_cloud data/scene.ply \
    --captured_cameras data/train_cameras.json \
    --generated_cameras outputs/generated_cameras.json \
    --captured_stride 4 \
    --max_points 100000 \
    --output outputs/visualization/index.html \
    --port 8000
```

浏览器：

```text
http://localhost:8000/index.html
```

检查：

```text
1. Captured Camera 轨迹是否正确

2. Camera forward 是否指向正确方向

3. Camera 是否出现翻转

4. Generated Camera 是否位于合理空间

5. Generated Camera 是否与 Captured Camera 分布互补

6. Camera Frustum 是否与场景尺度匹配
```

这套可视化作为后续视角约束和生成算法的主要 Debug 工具。
