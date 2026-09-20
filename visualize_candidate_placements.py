#!/usr/bin/env python3
"""Interactive V3.2/V3.3 placement/generalization diagnostics.

The plot separates raw/corrected initial positions, rejected conflicts, final
grid candidates, and Stage-3 selected grid candidates.  It consumes the normal
``gen_cameras_meta.json`` contract, so visualization does not rerun placement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import plotly.graph_objects as go

from viewpoint_framework.cameras_util import Camera
from viewpoint_framework.geometry_util import build_camera_frustum_world
from viewpoint_framework.points_util import load_ply_point_cloud
from viewpoint_framework.util import prepare_html_output_path, serve_html


INITIAL_COLOR = "#A78324"
CONFLICT_COLOR = "#913F3F"
FINAL_COLOR = "#315D86"
SELECTED_COLOR = "#347054"


def _camera_from_dict(value) -> Optional[Camera]:
    if not isinstance(value, dict) or value.get("c2w") is None:
        return None
    return Camera(
        index=int(value.get("index", -1)), fx=float(value["fx"]),
        fy=float(value["fy"]), cx=float(value["cx"]), cy=float(value["cy"]),
        width=int(value["width"]), height=int(value["height"]),
        w2c=np.asarray(value["w2c"], dtype=np.float64),
        c2w=np.asarray(value["c2w"], dtype=np.float64),
    )


def _point(value):
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float64)
    return result if result.shape == (3,) and np.isfinite(result).all() else None


def _selected_grid_ids(path: Optional[str]) -> dict[int, int]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("selected_candidates", []) if isinstance(payload, dict) else []
    return {int(row["grid_id"]): index for index, row in enumerate(rows)
            if row.get("grid_id") is not None}


def _hover(candidate):
    geometry = candidate.get("geometry_metadata", {})
    keys = (
        "grid_azimuth_deg", "grid_elevation_deg", "rho", "rho_source",
        "is_extension_column", "local_clearance_threshold",
        "initial_height_limit_source", "final_height_limit_source",
        "initial_height_clip_applied", "final_height_clip_applied",
        "adjustment_type", "adjustment_skip_reason", "final_signed_radius",
        "column_kind", "h_cross", "height_source", "center_guard",
        "raw_signed_radius", "nominal_radius_cap", "emergency_radius_cap",
        "radius_extension_attempted", "radius_extension_blocked_by_hole",
        "radius_choice", "radius_branch", "probe_hole_detected",
        "final_output_index",
    )
    lines = [f"<b>grid {candidate.get('grid_id')}</b>",
             f"status: {candidate.get('status')}",
             f"reject: {candidate.get('reject_reason')}"]
    lines.extend(f"{key}: {geometry.get(key)}" for key in keys)
    return "<br>".join(lines)


def _points_trace(rows, position_key, color, name, *, symbol="circle", size=5):
    packed = [(p, c) for c in rows if (p := _point(c.get("geometry_metadata", {}).get(position_key))) is not None]
    if not packed:
        return None
    points = np.asarray([item[0] for item in packed])
    return go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
        marker={"size": size, "color": color, "symbol": symbol}, name=name,
        text=[_hover(item[1]) for item in packed],
        hovertemplate="%{text}<extra></extra>", legendgroup=name,
    )


def _frame_label_trace(rows):
    packed = [(p, c) for c in rows
              if (p := _point(c.get("geometry_metadata", {}).get("final_position"))) is not None]
    if not packed:
        return None
    points = np.asarray([item[0] for item in packed])
    labels = [f"frame_{int(item[1]['geometry_metadata']['final_output_index']):04d}"
              for item in packed]
    return go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="text",
        text=labels, textposition="top center", name="Final Selected View labels",
        hoverinfo="skip", showlegend=False,
    )


def _segments(rows, first_key, second_key, color, name, *, width=2, dash=None):
    xs, ys, zs = [], [], []
    for candidate in rows:
        geometry = candidate.get("geometry_metadata", {})
        first, second = _point(geometry.get(first_key)), _point(geometry.get(second_key))
        if first is None or second is None or np.linalg.norm(first - second) <= 1e-10:
            continue
        xs.extend((first[0], second[0], None))
        ys.extend((first[1], second[1], None))
        zs.extend((first[2], second[2], None))
    if not xs:
        return None
    line = {"color": color, "width": width}
    if dash:
        line["dash"] = dash
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines", line=line, opacity=0.65,
        name=name, hoverinfo="skip", legendgroup=name,
    )


def _frustum_trace(rows, color, name, depth):
    xs, ys, zs = [], [], []
    edges = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1))
    for row in rows:
        camera = _camera_from_dict(row.get("camera"))
        if camera is None:
            continue
        vertices = build_camera_frustum_world(
            camera.c2w, camera.fx, camera.fy, camera.cx, camera.cy,
            camera.width, camera.height, depth,
        )
        for i, j in edges:
            xs.extend((vertices[i, 0], vertices[j, 0], None))
            ys.extend((vertices[i, 1], vertices[j, 1], None))
            zs.extend((vertices[i, 2], vertices[j, 2], None))
        start, end = camera.position, camera.position + depth * 1.35 * camera.forward
        xs.extend((start[0], end[0], None))
        ys.extend((start[1], end[1], None))
        zs.extend((start[2], end[2], None))
    if not xs:
        return None
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line={"color": color, "width": 2}, opacity=0.8,
        name=f"{name} frustum/forward", hoverinfo="skip", legendgroup=name,
        showlegend=False,
    )


def _plane(center, up, height, extent, color, name):
    up = up / np.linalg.norm(up)
    trial = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    axis_a = np.cross(up, trial)
    axis_a /= np.linalg.norm(axis_a)
    axis_b = np.cross(up, axis_a)
    values = np.linspace(-extent, extent, 2)
    aa, bb = np.meshgrid(values, values)
    points = center + height * up + aa[..., None] * axis_a + bb[..., None] * axis_b
    return go.Surface(
        x=points[..., 0], y=points[..., 1], z=points[..., 2],
        surfacecolor=np.zeros((2, 2)), colorscale=[[0, color], [1, color]],
        showscale=False, opacity=0.16, name=name, hoverinfo="skip",
        legendgroup="global-height", showlegend=True,
    )


def _local_limit_segments(columns: Iterable[dict], center, up, frame_x, frame_z):
    xs, ys, zs = [], [], []
    for column in columns:
        if column.get("is_extension_column"):
            continue
        lower = column.get("lower_safe")
        upper = column.get("upper_safe")
        if lower is None and isinstance(column.get("lower"), dict):
            lower = column["lower"].get("safe_limit")
        if upper is None and isinstance(column.get("upper"), dict):
            upper = column["upper"].get("safe_limit")
        if lower is None or upper is None:
            continue
        angle = np.radians(float(column["azimuth_deg"]))
        base = center + float(column["rho"]) * (np.sin(angle) * frame_x + np.cos(angle) * frame_z)
        low, high = base + float(lower) * up, base + float(upper) * up
        xs.extend((low[0], high[0], None)); ys.extend((low[1], high[1], None)); zs.extend((low[2], high[2], None))
    if not xs:
        return None
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line={"color": "#6B5B95", "width": 4}, name="Local height limits",
        hoverinfo="skip", legendgroup="local-height",
    )


def visualize_candidate_placements(point_cloud_path, metadata_path, output_path,
                                   stage3_metadata_path=None, max_points=100000,
                                   point_size=1.4, point_opacity=0.42,
                                   show_local_limits=False, camera_scale=None):
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    version = str(metadata.get("config", {}).get("version", ""))
    if not (version.startswith("3.2") or version.startswith("3.3")):
        raise ValueError("Candidate placement visualization requires V3.2/V3.3 metadata.")
    candidates = metadata.get("candidates", [])
    selected_ids = _selected_grid_ids(stage3_metadata_path)
    selected = [row for row in candidates if int(row.get("grid_id", -1)) in selected_ids and row.get("camera")]
    unselected = [row for row in candidates if row.get("camera") and int(row.get("grid_id", -1)) not in selected_ids]
    conflicts = [row for row in candidates if not row.get("camera")]
    for row in selected:
        row.setdefault("geometry_metadata", {})["final_output_index"] = selected_ids[int(row["grid_id"])]

    cloud = load_ply_point_cloud(point_cloud_path, max_points=max_points)
    colors = (np.clip(cloud.colors, 0.0, 1.0) * 0.55 + 0.45 if cloud.colors is not None else None)
    marker = {"size": point_size, "opacity": point_opacity,
              "color": ([f"rgb({r},{g},{b})" for r, g, b in (colors * 255).astype(np.uint8)]
                        if colors is not None else "rgb(185,185,185)")}
    figure = go.Figure([go.Scatter3d(
        x=cloud.points[:, 0], y=cloud.points[:, 1], z=cloud.points[:, 2],
        mode="markers", marker=marker, name="Scene geometry", hoverinfo="skip",
    )])

    initial = _points_trace(candidates, "raw_initial_position", INITIAL_COLOR, "Raw initial")
    conflict = _points_trace(conflicts, "corrected_initial_position", CONFLICT_COLOR, "Rejected/conflict", symbol="x", size=6)
    final = _points_trace(unselected, "final_position", FINAL_COLOR, "Final unselected")
    chosen = _points_trace(selected, "final_position", SELECTED_COLOR, "Stage-3 selected", size=7)
    for trace in (initial, conflict, final, chosen, _frame_label_trace(selected),
                  _segments(candidates, "corrected_initial_position", "final_position", FINAL_COLOR, "Initial to final"),
                  _segments(candidates, "raw_initial_position", "corrected_initial_position", INITIAL_COLOR, "Height correction", width=4)):
        if trace is not None:
            figure.add_trace(trace)

    placement = metadata.get("placement", {})
    center = _point(placement.get("scene_center"))
    frame = placement.get("coordinate_frame", {})
    up, frame_x, frame_z = _point(frame.get("up_axis")), _point(frame.get("x_axis")), _point(frame.get("z_axis"))
    global_height = metadata.get("global_height", {})
    if center is not None and up is not None:
        figure.add_trace(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]], mode="markers",
            marker={"size": 9, "color": "#D62728", "symbol": "diamond"},
            name="Scene center", hovertemplate="scene_center<extra></extra>"))
        extent = max(cloud.diagonal * 0.55, 1e-3)
        for key, color, name in (("height_min", "#7A68A6", "Global height min"),
                                 ("height_max", "#A6687A", "Global height max")):
            if global_height.get(key) is not None:
                figure.add_trace(_plane(center, up, float(global_height[key]), extent, color, name))
        if show_local_limits and frame_x is not None and frame_z is not None:
            trace = _local_limit_segments(metadata.get("local_height_columns", []), center, up, frame_x, frame_z)
            if trace is not None:
                figure.add_trace(trace)

    median_rho = placement.get("median_captured_horizontal_radius")
    depth = (max(float(camera_scale), 1e-4) if camera_scale is not None else
             max(float(median_rho) * 0.04, 1e-4) if median_rho is not None
             else max(cloud.diagonal * 0.015, 1e-4))
    for rows, color, name in ((unselected, FINAL_COLOR, "Final unselected"),
                              (selected, SELECTED_COLOR, "Stage-3 selected")):
        trace = _frustum_trace(rows, color, name, depth)
        if trace is not None:
            figure.add_trace(trace)
    figure.update_layout(
        title=f"V{version} Candidate Placements — valid {len(unselected) + len(selected)}, rejected {len(conflicts)}, selected {len(selected)}",
        margin={"l": 0, "r": 0, "t": 55, "b": 0}, height=900,
        scene={"aspectmode": "data", "xaxis_title": "World X", "yaxis_title": "World Y", "zaxis_title": "World Z"},
        legend={"groupclick": "togglegroup"}, hoverlabel={"bgcolor": "white", "font_size": 11},
    )
    output = prepare_html_output_path(output_path)
    figure.write_html(str(output), include_plotlyjs=True, full_html=True,
                      config={"displaylogo": False, "scrollZoom": True, "responsive": True})
    print(f"[V{version} visualization] {output}")
    return str(output)


def build_argparser():
    parser = argparse.ArgumentParser(description="Visualize V3.2/V3.3 candidate placement and height safety.")
    parser.add_argument("--point_cloud", required=True)
    parser.add_argument("--metadata", required=True, help="Stage-2 gen_cameras_meta.json")
    parser.add_argument("--stage3_metadata", default=None, help="Optional Stage-3 debug/stage3_metadata.json")
    parser.add_argument("--output", default="candidate_placements/index.html")
    parser.add_argument("--max_points", type=int, default=100000)
    parser.add_argument("--point_size", type=float, default=1.4)
    parser.add_argument("--point_opacity", type=float, default=0.42)
    parser.add_argument("--show_local_limits", action="store_true")
    parser.add_argument("--camera-scale", type=float, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open_browser", action="store_true")
    parser.add_argument("--no_serve", action="store_true")
    return parser


def main():
    args = build_argparser().parse_args()
    html = visualize_candidate_placements(
        args.point_cloud, args.metadata, args.output,
        stage3_metadata_path=args.stage3_metadata, max_points=args.max_points,
        point_size=args.point_size, point_opacity=args.point_opacity,
        show_local_limits=args.show_local_limits, camera_scale=args.camera_scale,
    )
    if not args.no_serve:
        serve_html(html, host=args.host, port=args.port, open_browser=args.open_browser)


if __name__ == "__main__":
    main()
