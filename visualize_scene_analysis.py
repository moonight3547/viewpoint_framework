#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Interactive diagnostic visualization for scene-understanding results.

This module reuses the existing viewpoint-framework Plotly camera/point-cloud
builders rather than replacing ``visualize_cameras.py``.  It adds analysis
specific overlays:
    - fitted scene center
    - per-mode camera groups
    - sight-line projection and residuals
    - observed/generation spherical bbox wireframes
    - sampled directional radius field
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import plotly.graph_objects as go

from viewpoint_framework.cameras_util import Camera, camera_positions
from viewpoint_framework.geometry_util import estimate_camera_visualization_depth
from viewpoint_framework.points_util import load_ply_point_cloud
from viewpoint_framework.scene_types import CameraMode, CameraSceneRelation
from viewpoint_framework.scene_understanding import SceneUnderstandingResult
from viewpoint_framework.util import prepare_html_output_path
from viewpoint_framework.view_space import (
    azimuth_elevation_to_direction,
    sample_circular_interval,
)
from viewpoint_framework.visualize_cameras import (
    build_camera_frustum_trace,
    build_point_cloud_trace,
)


MODE_COLORS = {
    CameraMode.OUTSIDE_IN: "#2196F3",
    CameraMode.INSIDE_OUT: "#FF9800",
    CameraMode.AMBIGUOUS: "#9C27B0",
    CameraMode.OUTLIER: "#F44336",
}

CENTER_COLOR = "#00C853"
BBOX_OBSERVED_COLOR = "#616161"
BBOX_GENERATION_COLOR = "#00ACC1"
RADIUS_OUTSIDE_COLOR = "#1565C0"
RADIUS_INSIDE_COLOR = "#EF6C00"


def _camera_map(cameras: Sequence[Camera]) -> Dict[int, Camera]:
    return {int(camera.index): camera for camera in cameras}


def _relation_map(
    relations: Sequence[CameraSceneRelation],
) -> Dict[int, CameraSceneRelation]:
    return {int(relation.camera_index): relation for relation in relations}


def _cameras_for_mode(
    cameras: Sequence[Camera],
    relation_by_index: Dict[int, CameraSceneRelation],
    mode: CameraMode,
) -> List[Camera]:
    return [
        camera
        for camera in cameras
        if camera.index in relation_by_index
        and relation_by_index[camera.index].mode == mode
    ]


def _build_analysis_camera_center_trace(
    cameras: Sequence[Camera],
    relation_by_index: Dict[int, CameraSceneRelation],
    mode: CameraMode,
) -> Optional[go.Scatter3d]:
    if len(cameras) == 0:
        return None

    positions = camera_positions(cameras)
    hover = []

    for camera in cameras:
        relation = relation_by_index[camera.index]
        hover.append(
            (
                f"<b>{mode.value}</b><br>"
                f"camera: {camera.index}<br>"
                f"radius: {relation.radius:.6f}<br>"
                f"lambda: {relation.lambda_center:.6f}<br>"
                f"alignment: {relation.alignment_deg:.3f} deg<br>"
                f"sight residual: {relation.sight_residual:.6f}<br>"
                f"residual/radius: {relation.residual_ratio:.6f}<br>"
                f"fit weight: {relation.robust_weight:.4f}<br>"
                f"confidence: {relation.confidence:.4f}<br>"
                f"azimuth: {relation.azimuth_deg}<br>"
                f"elevation: {relation.elevation_deg}"
            )
        )

    return go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="markers",
        marker={"size": 4, "color": MODE_COLORS[mode]},
        name=f"{mode.value} ({len(cameras)})",
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        legendgroup=f"mode_{mode.value}",
        showlegend=True,
    )


def _build_original_trajectory_trace(
    cameras: Sequence[Camera],
) -> Optional[go.Scatter3d]:
    if len(cameras) < 2:
        return None
    positions = camera_positions(cameras)
    return go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="lines",
        line={"color": "rgba(90,90,90,0.45)", "width": 2},
        name="Captured trajectory",
        hoverinfo="skip",
        legendgroup="trajectory",
        showlegend=True,
    )


def _build_scene_center_trace(
    center: np.ndarray,
    result: SceneUnderstandingResult,
) -> go.Scatter3d:
    fit = result.profile.center_fit
    text = (
        "<b>Fitted scene center</b><br>"
        f"x={center[0]:.6f}<br>"
        f"y={center[1]:.6f}<br>"
        f"z={center[2]:.6f}<br>"
        f"strategy={fit.strategy}<br>"
        f"median residual={fit.median_residual:.6f}<br>"
        f"condition={fit.condition_number:.4g}<br>"
        f"converged={fit.converged}"
    )
    return go.Scatter3d(
        x=[center[0]],
        y=[center[1]],
        z=[center[2]],
        mode="markers",
        marker={"size": 9, "color": CENTER_COLOR, "symbol": "diamond"},
        name="Scene center",
        text=[text],
        hovertemplate="%{text}<extra></extra>",
        legendgroup="scene_center",
        showlegend=True,
    )


def _build_sight_projection_traces(
    relations: Sequence[CameraSceneRelation],
    center: np.ndarray,
    stride: int,
) -> List[go.Scatter3d]:
    """Visualize camera-axis projection to center and perpendicular residual."""

    stride = max(int(stride), 1)
    relation_subset = list(relations)[::stride]

    traces: List[go.Scatter3d] = []

    for mode in CameraMode:
        group = [r for r in relation_subset if r.mode == mode]
        if len(group) == 0:
            continue

        axis_x, axis_y, axis_z = [], [], []
        residual_x, residual_y, residual_z = [], [], []

        for relation in group:
            p = relation.position
            q = p + relation.lambda_center * relation.forward

            axis_x.extend([p[0], q[0], None])
            axis_y.extend([p[1], q[1], None])
            axis_z.extend([p[2], q[2], None])

            residual_x.extend([q[0], center[0], None])
            residual_y.extend([q[1], center[1], None])
            residual_z.extend([q[2], center[2], None])

        traces.append(
            go.Scatter3d(
                x=axis_x,
                y=axis_y,
                z=axis_z,
                mode="lines",
                line={"color": MODE_COLORS[mode], "width": 2},
                opacity=0.45,
                name=f"{mode.value} sight-axis projection",
                hoverinfo="skip",
                legendgroup=f"sight_{mode.value}",
                showlegend=True,
            )
        )

        traces.append(
            go.Scatter3d(
                x=residual_x,
                y=residual_y,
                z=residual_z,
                mode="lines",
                line={"color": MODE_COLORS[mode], "width": 1, "dash": "dot"},
                opacity=0.45,
                name=f"{mode.value} center residual",
                hoverinfo="skip",
                legendgroup=f"residual_{mode.value}",
                showlegend=True,
            )
        )

    return traces


def _append_segment(xs, ys, zs, a: np.ndarray, b: np.ndarray) -> None:
    xs.extend([a[0], b[0], None])
    ys.extend([a[1], b[1], None])
    zs.extend([a[2], b[2], None])


def _bbox_wireframe_trace(
    center: np.ndarray,
    frame,
    bbox,
    generation: bool,
    mode: CameraMode,
) -> go.Scatter3d:
    """Build a spherical-box wireframe at inner/outer radius."""

    if generation:
        az_interval = bbox.generation_azimuth
        el_min, el_max = bbox.generation_elevation_deg
        r_min, r_max = bbox.generation_radius
        color = BBOX_GENERATION_COLOR
        label = "generation bbox"
    else:
        az_interval = bbox.observed_azimuth
        el_min, el_max = bbox.observed_elevation_deg
        r_min, r_max = bbox.observed_radius
        color = BBOX_OBSERVED_COLOR
        label = "observed bbox"

    # Sample enough points to make curved spherical edges readable.
    az_step = max(az_interval.span_deg / 48.0, 1.0) if az_interval.span_deg > 0 else 1.0
    azimuths = sample_circular_interval(az_interval, az_step, include_end=True)
    elevations = np.linspace(el_min, el_max, num=24)

    xs, ys, zs = [], [], []

    def world_point(radius: float, az: float, el: float) -> np.ndarray:
        direction = azimuth_elevation_to_direction(az, el, frame)
        return center + radius * direction

    for radius in sorted(set([float(r_min), float(r_max)])):
        # Constant elevation arcs.
        for elevation in (el_min, el_max):
            points = [world_point(radius, az, elevation) for az in azimuths]
            for a, b in zip(points[:-1], points[1:]):
                _append_segment(xs, ys, zs, a, b)

        # Constant azimuth arcs at the two azimuth boundaries.
        for azimuth in (az_interval.start_deg, az_interval.end_deg):
            points = [world_point(radius, azimuth, el) for el in elevations]
            for a, b in zip(points[:-1], points[1:]):
                _append_segment(xs, ys, zs, a, b)

    # Radial connectors at four angular corners.
    if abs(r_max - r_min) > 1e-9:
        for azimuth in (az_interval.start_deg, az_interval.end_deg):
            for elevation in (el_min, el_max):
                a = world_point(r_min, azimuth, elevation)
                b = world_point(r_max, azimuth, elevation)
                _append_segment(xs, ys, zs, a, b)

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line={"color": color, "width": 3 if generation else 2},
        opacity=0.75 if generation else 0.45,
        name=f"{mode.value} {label}",
        hoverinfo="skip",
        legendgroup=f"bbox_{mode.value}_{'gen' if generation else 'obs'}",
        showlegend=True,
    )


def _build_radius_field_trace(
    samples,
    mode: CameraMode,
) -> Optional[go.Scatter3d]:
    valid = [sample for sample in samples if sample.estimate.valid and sample.position is not None]
    if len(valid) == 0:
        return None

    positions = np.stack([sample.position for sample in valid], axis=0)
    confidence = np.asarray([sample.estimate.confidence for sample in valid])

    hover = []
    for sample in valid:
        estimate = sample.estimate
        hover.append(
            (
                f"<b>{mode.value} radius field</b><br>"
                f"azimuth={sample.azimuth_deg:.2f} deg<br>"
                f"elevation={sample.elevation_deg:.2f} deg<br>"
                f"radius={estimate.nominal:.6f}<br>"
                f"range=[{estimate.low:.6f}, {estimate.high:.6f}]<br>"
                f"confidence={estimate.confidence:.4f}<br>"
                f"strategy={estimate.strategy}<br>"
                f"source={estimate.source}"
            )
        )

    base_color = (
        RADIUS_OUTSIDE_COLOR
        if mode == CameraMode.OUTSIDE_IN
        else RADIUS_INSIDE_COLOR
    )

    # Confidence is encoded primarily as marker opacity/size while the mode keeps
    # a stable semantic color.
    marker_sizes = 3.0 + 5.0 * confidence
    marker_opacity = float(np.clip(np.mean(confidence), 0.2, 0.9))

    return go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="markers",
        marker={
            "size": marker_sizes,
            "color": base_color,
            "opacity": marker_opacity,
        },
        name=f"{mode.value} radius samples ({len(valid)})",
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        legendgroup=f"radius_{mode.value}",
        showlegend=True,
    )


def _estimate_scene_diagonal_without_pointcloud(
    cameras: Sequence[Camera],
    center: np.ndarray,
) -> float:
    positions = camera_positions(cameras)
    points = np.concatenate([positions, center[None, :]], axis=0)
    extent = np.max(points, axis=0) - np.min(points, axis=0)
    diagonal = float(np.linalg.norm(extent))
    return diagonal if diagonal > 1e-8 else 1.0


def visualize_scene_analysis(
    result: SceneUnderstandingResult,
    cameras: Sequence[Camera],
    output_path: str,
    point_cloud_path: Optional[str] = None,
    max_points: int = 100_000,
    point_size: float = 1.5,
    point_opacity: float = 0.65,
    camera_stride: int = 4,
    sight_line_stride: int = 4,
    frustum_scale: float = 0.015,
    frustum_spacing_scale: float = 0.35,
    frustum_min_scale: float = 0.002,
    frustum_max_scale: float = 0.02,
) -> str:
    """Generate standalone scene-analysis HTML diagnostics."""

    if camera_stride <= 0:
        raise ValueError("camera_stride must be >0")

    profile = result.profile
    center = np.asarray(profile.center_fit.center, dtype=np.float64)
    relation_by_index = _relation_map(profile.camera_relations)

    fig = go.Figure()

    # ------------------------------------------------------------------
    # Point cloud + scene scale.
    # ------------------------------------------------------------------
    if point_cloud_path:
        point_cloud = load_ply_point_cloud(
            point_cloud_path,
            max_points=max_points,
        )
        fig.add_trace(
            build_point_cloud_trace(
                point_cloud=point_cloud,
                point_size=point_size,
                point_opacity=point_opacity,
            )
        )
        scene_diagonal = point_cloud.diagonal
    else:
        scene_diagonal = _estimate_scene_diagonal_without_pointcloud(
            cameras,
            center,
        )

    display_cameras = list(cameras)[::camera_stride]
    display_positions = camera_positions(display_cameras)

    frustum_depth = estimate_camera_visualization_depth(
        scene_diagonal=scene_diagonal,
        camera_positions=display_positions,
        scene_scale=frustum_scale,
        spacing_scale=frustum_spacing_scale,
        min_scene_scale=frustum_min_scale,
        max_scene_scale=frustum_max_scale,
    )

    trajectory = _build_original_trajectory_trace(display_cameras)
    if trajectory is not None:
        fig.add_trace(trajectory)

    # ------------------------------------------------------------------
    # Per-mode camera frustums + diagnostic centers.
    # ------------------------------------------------------------------
    for mode in CameraMode:
        group = _cameras_for_mode(
            display_cameras,
            relation_by_index,
            mode,
        )
        if len(group) == 0:
            continue

        fig.add_trace(
            build_camera_frustum_trace(
                cameras=group,
                depth=frustum_depth,
                color=MODE_COLORS[mode],
                legend_group=f"mode_{mode.value}",
            )
        )
        centers = _build_analysis_camera_center_trace(
            group,
            relation_by_index,
            mode,
        )
        if centers is not None:
            fig.add_trace(centers)

    # ------------------------------------------------------------------
    # Fitted center + sight-line diagnostics.
    # ------------------------------------------------------------------
    fig.add_trace(_build_scene_center_trace(center, result))

    for trace in _build_sight_projection_traces(
        profile.camera_relations,
        center,
        stride=sight_line_stride,
    ):
        fig.add_trace(trace)

    # ------------------------------------------------------------------
    # BBoxes + directional radius samples.
    # ------------------------------------------------------------------
    for key, bbox in profile.view_bboxes.items():
        mode = CameraMode(key)
        fig.add_trace(
            _bbox_wireframe_trace(
                center=center,
                frame=profile.coordinate_frame,
                bbox=bbox,
                generation=False,
                mode=mode,
            )
        )
        fig.add_trace(
            _bbox_wireframe_trace(
                center=center,
                frame=profile.coordinate_frame,
                bbox=bbox,
                generation=True,
                mode=mode,
            )
        )

        radius_trace = _build_radius_field_trace(
            profile.radius_field_samples.get(key, []),
            mode,
        )
        if radius_trace is not None:
            fig.add_trace(radius_trace)

    mode_summary = profile.mode_summary
    fit = profile.center_fit

    fig.update_layout(
        title={
            "text": (
                "Scene Input Understanding"
                "<br><sup>"
                f"center={fit.strategy} | mode={mode_summary.dominant_mode.value} "
                f"({mode_summary.dominant_confidence:.3f}) | "
                f"outside={mode_summary.outside_in_count}, "
                f"inside={mode_summary.inside_out_count}, "
                f"ambiguous={mode_summary.ambiguous_count}, "
                f"outlier={mode_summary.outlier_count}"
                "</sup>"
            ),
            "x": 0.02,
        },
        margin={"l": 0, "r": 0, "t": 80, "b": 0},
        scene={
            "xaxis": {"title": "World X"},
            "yaxis": {"title": "World Y"},
            "zaxis": {"title": "World Z"},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.5, "y": -1.5, "z": 1.2}},
        },
        legend={"x": 0.01, "y": 0.99, "groupclick": "togglegroup"},
        hoverlabel={"bgcolor": "white", "font_size": 12},
        height=900,
    )

    html_path = prepare_html_output_path(output_path)
    fig.write_html(
        str(html_path),
        include_plotlyjs=True,
        full_html=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "displayModeBar": True,
            "responsive": True,
        },
    )

    return str(html_path)
