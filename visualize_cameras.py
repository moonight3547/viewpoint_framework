#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive point-cloud + camera-pose visualization.

Visualization backend:
    Plotly WebGL

Inputs
------
1. PLY point cloud.

2. Captured camera JSON:
       unified 18-D camera format.

3. Generated camera JSON:
       unified 18-D camera format.

Output
------
Standalone interactive HTML.

Optional
--------
Start a localhost HTTP server and block until Ctrl-C.
"""

import argparse
from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go

from viewpoint_framework.cameras_util import (
    Camera,
    camera_positions,
    downsample_cameras,
    load_cameras_json,
)

from viewpoint_framework.geometry_util import (
    build_camera_frustum_world,
    estimate_camera_visualization_depth,
)

from viewpoint_framework.points_util import (
    PointCloudData,
    load_ply_point_cloud,
)

from viewpoint_framework.util import (
    prepare_html_output_path,
    serve_html,
)


# ==============================================================================
# Visualization defaults
# ==============================================================================

CAPTURED_CAMERA_COLOR = "#2196F3"
GENERATED_CAMERA_COLOR = "#FF5722"

CAMERA_CENTER_SIZE = 3
CAMERA_FRUSTUM_LINE_WIDTH = 1.5
CAMERA_TRAJECTORY_LINE_WIDTH = 3


# ==============================================================================
# Plotly: point cloud
# ==============================================================================

def build_point_cloud_trace(
    point_cloud: PointCloudData,
    point_size: float = 1.5,
    point_opacity: float = 0.75,
) -> go.Scatter3d:
    """
    Build point-cloud Plotly trace.
    """

    points = point_cloud.points
    colors = point_cloud.colors

    marker = {
        "size": point_size,
        "opacity": point_opacity,
    }

    if colors is not None:

        rgb = np.clip(
            colors * 255.0,
            0,
            255,
        ).astype(
            np.uint8
        )

        marker["color"] = [
            f"rgb({r},{g},{b})"
            for r, g, b in rgb
        ]

    else:

        marker["color"] = (
            "rgb(150,150,150)"
        )

    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],

        mode="markers",

        marker=marker,

        name="Point Cloud",

        hoverinfo="skip",

        legendgroup="pointcloud",

        showlegend=True,
    )


# ==============================================================================
# Plotly: camera frustums
# ==============================================================================

def build_camera_frustum_trace(
    cameras: Sequence[Camera],
    depth: float,
    color: str,
    legend_group: str,
) -> go.Scatter3d:
    """
    Merge all camera frustum edges into one Plotly trace.

    This is much more efficient than creating one Plotly trace
    for every individual camera.
    """

    xs = []
    ys = []
    zs = []

    # --------------------------------------------------------------------------
    # Frustum topology
    #
    # 0: camera center
    # 1: top-left
    # 2: top-right
    # 3: bottom-right
    # 4: bottom-left
    # --------------------------------------------------------------------------

    edges = [
        # Camera center -> image plane
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),

        # Image plane rectangle
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 1),
    ]

    for camera in cameras:

        vertices = build_camera_frustum_world(
            c2w=camera.c2w,

            fx=camera.fx,
            fy=camera.fy,
            cx=camera.cx,
            cy=camera.cy,

            width=camera.width,
            height=camera.height,

            depth=depth,
        )

        for i, j in edges:

            xs.extend(
                [
                    vertices[i, 0],
                    vertices[j, 0],
                    None,
                ]
            )

            ys.extend(
                [
                    vertices[i, 1],
                    vertices[j, 1],
                    None,
                ]
            )

            zs.extend(
                [
                    vertices[i, 2],
                    vertices[j, 2],
                    None,
                ]
            )

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,

        mode="lines",

        line={
            "color": color,
            "width": CAMERA_FRUSTUM_LINE_WIDTH,
        },

        hoverinfo="skip",

        legendgroup=legend_group,

        showlegend=False,
    )


def build_camera_center_trace(
    cameras: Sequence[Camera],
    color: str,
    name: str,
    legend_group: str,
) -> go.Scatter3d:
    """
    Build camera-center point trace.

    Hovering over a camera displays:
        - original camera index
        - position
        - forward direction
    """

    positions = camera_positions(
        cameras
    )

    hover_text = []

    for camera in cameras:

        p = camera.position
        f = camera.forward

        hover_text.append(
            (
                f"<b>{name}</b><br>"
                f"index: {camera.index}<br><br>"

                f"position:<br>"
                f"x = {p[0]:.6f}<br>"
                f"y = {p[1]:.6f}<br>"
                f"z = {p[2]:.6f}<br><br>"

                f"forward:<br>"
                f"x = {f[0]:.6f}<br>"
                f"y = {f[1]:.6f}<br>"
                f"z = {f[2]:.6f}"
            )
        )

    return go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],

        mode="markers",

        marker={
            "size": CAMERA_CENTER_SIZE,
            "color": color,
        },

        name=name,

        text=hover_text,

        hovertemplate=(
            "%{text}<extra></extra>"
        ),

        legendgroup=legend_group,

        showlegend=True,
    )


def build_camera_trajectory_trace(
    cameras: Sequence[Camera],
    color: str,
    legend_group: str,
) -> Optional[go.Scatter3d]:
    """
    Build camera-center trajectory.
    """

    if len(cameras) <= 1:

        return None

    positions = camera_positions(
        cameras
    )

    return go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],

        mode="lines",

        line={
            "color": color,
            "width": CAMERA_TRAJECTORY_LINE_WIDTH,
        },

        opacity=0.55,

        hoverinfo="skip",

        legendgroup=legend_group,

        showlegend=False,
    )


def add_camera_group(
    fig: go.Figure,
    cameras: Sequence[Camera],
    depth: float,
    color: str,
    name: str,
    legend_group: str,
) -> None:
    """
    Add one logical camera group:

        frustums
        + trajectory
        + camera centers

    All traces share the same legend group.
    """

    if len(cameras) == 0:
        return

    # --------------------------------------------------------------------------
    # Frustums
    # --------------------------------------------------------------------------

    fig.add_trace(
        build_camera_frustum_trace(
            cameras=cameras,
            depth=depth,
            color=color,
            legend_group=legend_group,
        )
    )

    # --------------------------------------------------------------------------
    # Trajectory
    # --------------------------------------------------------------------------

    trajectory = (
        build_camera_trajectory_trace(
            cameras=cameras,
            color=color,
            legend_group=legend_group,
        )
    )

    if trajectory is not None:

        fig.add_trace(
            trajectory
        )

    # --------------------------------------------------------------------------
    # Camera centers + legend
    # --------------------------------------------------------------------------

    fig.add_trace(
        build_camera_center_trace(
            cameras=cameras,
            color=color,
            name=(
                f"{name} ({len(cameras)})"
            ),
            legend_group=legend_group,
        )
    )


# ==============================================================================
# Adaptive visualization scale
# ==============================================================================

def collect_visualized_camera_positions(
    captured_cameras: Sequence[Camera],
    generated_cameras: Sequence[Camera],
) -> np.ndarray:
    """
    Collect camera centers used in the current visualization.

    Returns:
        [N,3]
    """

    groups = []

    if len(captured_cameras) > 0:

        groups.append(
            camera_positions(
                captured_cameras
            )
        )

    if len(generated_cameras) > 0:

        groups.append(
            camera_positions(
                generated_cameras
            )
        )

    if len(groups) == 0:

        return np.empty(
            (0, 3),
            dtype=np.float64,
        )

    return np.concatenate(
        groups,
        axis=0,
    )


# ==============================================================================
# Main visualization API
# ==============================================================================

def visualize_cameras(
    point_cloud_path: str,
    captured_camera_json: Optional[str],
    generated_camera_json: Optional[str],
    output_path: str,

    captured_stride: int = 4,

    max_points: int = 100_000,

    # Adaptive camera visualization size
    frustum_scale: float = 0.015,
    frustum_spacing_scale: float = 0.35,
    frustum_min_scale: float = 0.002,
    frustum_max_scale: float = 0.02,

    point_size: float = 1.5,
    point_opacity: float = 0.75,
) -> str:
    """
    Generate standalone interactive HTML.

    Args
    ----
    point_cloud_path:
        Input PLY point cloud.

    captured_camera_json:
        Captured camera pose JSON.

        May be None.

    generated_camera_json:
        Generated camera pose JSON.

        May be None.

    output_path:
        Output HTML path or output directory.

    captured_stride:
        Captured-camera downsampling stride.

        Default:
            4

    max_points:
        Maximum number of displayed point-cloud points.

        <=0:
            keep all points.

    frustum_scale:
        Preferred camera-frustum depth relative
        to scene diagonal.

    frustum_spacing_scale:
        Maximum camera-frustum depth relative
        to median nearest-camera spacing.

    frustum_min_scale:
        Minimum camera-frustum depth relative
        to scene diagonal.

    frustum_max_scale:
        Maximum camera-frustum depth relative
        to scene diagonal.

    Returns
    -------
    str:
        Absolute generated HTML path.
    """

    if captured_stride <= 0:

        raise ValueError(
            "captured_stride must be > 0, "
            f"got {captured_stride}"
        )

    # ==========================================================================
    # 1. Load point cloud
    # ==========================================================================

    print(
        f"[Visualization] Loading point cloud:\n"
        f"  {point_cloud_path}"
    )

    point_cloud = load_ply_point_cloud(
        point_cloud_path,
        max_points=max_points,
    )

    print(
        "[Visualization] Point cloud:"
        f"\n  display points = {point_cloud.num_points}"
        f"\n  center         = {point_cloud.center}"
        f"\n  extent         = {point_cloud.extent}"
        f"\n  diagonal       = {point_cloud.diagonal:.6f}"
    )

    # ==========================================================================
    # 2. Load cameras
    # ==========================================================================

    captured_cameras = []
    generated_cameras = []

    # --------------------------------------------------------------------------
    # Captured
    # --------------------------------------------------------------------------

    if captured_camera_json:

        captured_all = load_cameras_json(
            captured_camera_json
        )

        captured_cameras = downsample_cameras(
            captured_all,
            stride=captured_stride,
        )

        print(
            "[Visualization] Captured cameras:"
            f"\n  input   = {len(captured_all)}"
            f"\n  display = {len(captured_cameras)}"
            f"\n  stride  = {captured_stride}"
        )

    # --------------------------------------------------------------------------
    # Generated
    # --------------------------------------------------------------------------

    if generated_camera_json:

        generated_cameras = load_cameras_json(
            generated_camera_json
        )

        print(
            "[Visualization] Generated cameras:"
            f"\n  input   = {len(generated_cameras)}"
            f"\n  display = {len(generated_cameras)}"
        )

    # ==========================================================================
    # 3. Adaptive camera visualization size
    # ==========================================================================

    vis_camera_positions = (
        collect_visualized_camera_positions(
            captured_cameras=captured_cameras,
            generated_cameras=generated_cameras,
        )
    )

    frustum_depth = (
        estimate_camera_visualization_depth(
            scene_diagonal=point_cloud.diagonal,

            camera_positions=vis_camera_positions,

            scene_scale=frustum_scale,

            spacing_scale=frustum_spacing_scale,

            min_scene_scale=frustum_min_scale,

            max_scene_scale=frustum_max_scale,
        )
    )

    print(
        "[Visualization] Camera visualization scale:"
        f"\n  scene diagonal = {point_cloud.diagonal:.6f}"
        f"\n  frustum depth  = {frustum_depth:.6f}"
    )

    # ==========================================================================
    # 4. Plotly figure
    # ==========================================================================

    fig = go.Figure()

    # --------------------------------------------------------------------------
    # Point cloud
    # --------------------------------------------------------------------------

    fig.add_trace(
        build_point_cloud_trace(
            point_cloud=point_cloud,
            point_size=point_size,
            point_opacity=point_opacity,
        )
    )

    # --------------------------------------------------------------------------
    # Captured cameras
    # --------------------------------------------------------------------------

    add_camera_group(
        fig=fig,

        cameras=captured_cameras,

        depth=frustum_depth,

        color=CAPTURED_CAMERA_COLOR,

        name="Captured Cameras",

        legend_group="captured",
    )

    # --------------------------------------------------------------------------
    # Generated cameras
    # --------------------------------------------------------------------------

    add_camera_group(
        fig=fig,

        cameras=generated_cameras,

        depth=frustum_depth,

        color=GENERATED_CAMERA_COLOR,

        name="Generated Cameras",

        legend_group="generated",
    )

    # ==========================================================================
    # 5. Layout
    # ==========================================================================

    fig.update_layout(

        title={
            "text": (
                "Camera Pose Visualization"
                "<br>"
                "<sup>"
                f"Captured: {len(captured_cameras)} | "
                f"Generated: {len(generated_cameras)}"
                "</sup>"
            ),

            "x": 0.02,
        },

        margin={
            "l": 0,
            "r": 0,
            "t": 70,
            "b": 0,
        },

        scene={

            "xaxis": {
                "title": "World X",
                "showbackground": True,
                "backgroundcolor": "rgb(245,245,245)",
                "gridcolor": "rgb(210,210,210)",
            },

            "yaxis": {
                "title": "World Y",
                "showbackground": True,
                "backgroundcolor": "rgb(245,245,245)",
                "gridcolor": "rgb(210,210,210)",
            },

            "zaxis": {
                "title": "World Z",
                "showbackground": True,
                "backgroundcolor": "rgb(245,245,245)",
                "gridcolor": "rgb(210,210,210)",
            },

            # One world-space unit should have equal visual scale
            # along X / Y / Z.
            "aspectmode": "data",

            "camera": {

                "eye": {
                    "x": 1.5,
                    "y": -1.5,
                    "z": 1.2,
                },
            },
        },

        legend={
            "x": 0.01,
            "y": 0.99,

            # Clicking Captured / Generated toggles all traces
            # belonging to the same legend group.
            "groupclick": "togglegroup",
        },

        hoverlabel={
            "bgcolor": "white",
            "font_size": 12,
        },

        height=900,
    )

    # ==========================================================================
    # 6. Save standalone HTML
    # ==========================================================================

    output_path = prepare_html_output_path(
        output_path
    )

    fig.write_html(
        str(output_path),

        # Self-contained:
        # Plotly JS is embedded into the HTML.
        include_plotlyjs=True,

        full_html=True,

        config={
            "displaylogo": False,

            # Mouse-wheel zoom
            "scrollZoom": True,

            # Keep Plotly mode bar
            "displayModeBar": True,

            # Resize with browser
            "responsive": True,
        },
    )

    print()
    print(
        "=========================================================="
    )
    print(
        "[Visualization] HTML generated"
    )
    print(
        f"  {output_path}"
    )
    print(
        "=========================================================="
    )

    return str(
        output_path
    )


# ==============================================================================
# CLI
# ==============================================================================

def build_argparser() -> argparse.ArgumentParser:
    """
    Build command-line interface.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Interactive point-cloud and "
            "camera-pose visualization."
        )
    )

    # --------------------------------------------------------------------------
    # Input / output
    # --------------------------------------------------------------------------

    parser.add_argument(
        "--point_cloud",
        type=str,
        required=True,
        help="Input PLY point cloud.",
    )

    parser.add_argument(
        "--captured_cameras",
        type=str,
        default=None,
        help=(
            "Captured camera JSON in unified "
            "18-D format."
        ),
    )

    parser.add_argument(
        "--generated_cameras",
        type=str,
        default=None,
        help=(
            "Generated camera JSON in unified "
            "18-D format."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="camera_visualization/index.html",
        help=(
            "Output HTML file or directory."
        ),
    )

    # --------------------------------------------------------------------------
    # Camera sequence
    # --------------------------------------------------------------------------

    parser.add_argument(
        "--captured_stride",
        type=int,
        default=4,
        help=(
            "Captured-camera downsampling stride. "
            "Default: 4."
        ),
    )

    # --------------------------------------------------------------------------
    # Adaptive camera visualization size
    # --------------------------------------------------------------------------

    parser.add_argument(
        "--frustum_scale",
        type=float,
        default=0.015,
        help=(
            "Preferred camera-frustum depth relative "
            "to scene diagonal. Default: 0.015."
        ),
    )

    parser.add_argument(
        "--frustum_spacing_scale",
        type=float,
        default=0.35,
        help=(
            "Maximum camera-frustum depth relative "
            "to typical camera spacing. "
            "Default: 0.35."
        ),
    )

    parser.add_argument(
        "--frustum_min_scale",
        type=float,
        default=0.002,
        help=(
            "Minimum camera-frustum depth relative "
            "to scene diagonal. Default: 0.002."
        ),
    )

    parser.add_argument(
        "--frustum_max_scale",
        type=float,
        default=0.02,
        help=(
            "Maximum camera-frustum depth relative "
            "to scene diagonal. Default: 0.02."
        ),
    )

    # --------------------------------------------------------------------------
    # Point cloud
    # --------------------------------------------------------------------------

    parser.add_argument(
        "--max_points",
        type=int,
        default=100000,
        help=(
            "Maximum displayed point-cloud points. "
            "Use <=0 to disable downsampling."
        ),
    )

    parser.add_argument(
        "--point_size",
        type=float,
        default=1.5,
        help=(
            "Plotly point marker size. "
            "Default: 1.5."
        ),
    )

    parser.add_argument(
        "--point_opacity",
        type=float,
        default=0.75,
        help=(
            "Point-cloud opacity. "
            "Default: 0.75."
        ),
    )

    # --------------------------------------------------------------------------
    # HTTP server
    # --------------------------------------------------------------------------

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help=(
            "HTTP server bind host. "
            "Default: 127.0.0.1."
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help=(
            "HTTP server port. "
            "Use 0 to let OS select a free port. "
            "Default: 8000."
        ),
    )

    parser.add_argument(
        "--open_browser",
        action="store_true",
        help=(
            "Automatically open local system browser. "
            "Usually disable on remote GPU servers."
        ),
    )

    parser.add_argument(
        "--no_serve",
        action="store_true",
        help=(
            "Only generate HTML and do not start "
            "the HTTP server."
        ),
    )

    return parser


# ==============================================================================
# Entry
# ==============================================================================

def main() -> None:

    parser = build_argparser()

    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # Generate visualization
    # --------------------------------------------------------------------------

    html_path = visualize_cameras(
        point_cloud_path=args.point_cloud,

        captured_camera_json=(
            args.captured_cameras
        ),

        generated_camera_json=(
            args.generated_cameras
        ),

        output_path=args.output,

        captured_stride=args.captured_stride,

        max_points=args.max_points,

        frustum_scale=args.frustum_scale,

        frustum_spacing_scale=(
            args.frustum_spacing_scale
        ),

        frustum_min_scale=(
            args.frustum_min_scale
        ),

        frustum_max_scale=(
            args.frustum_max_scale
        ),

        point_size=args.point_size,

        point_opacity=args.point_opacity,
    )

    # --------------------------------------------------------------------------
    # HTTP server
    # --------------------------------------------------------------------------

    if not args.no_serve:

        serve_html(
            html_path=html_path,

            host=args.host,

            port=args.port,

            open_browser=args.open_browser,
        )


if __name__ == "__main__":
    main()