from types import SimpleNamespace

import numpy as np

from viewpoint_framework.stage3 import render_output
from viewpoint_framework.stage3.pipeline import Stage3Config
from viewpoint_framework.stage3.render_output import build_frame_manifest, render_camera_sequence
from viewpoint_framework.stage3.types import (
    CandidateOrigin, ReferenceSelectionResult, SelectionCandidate, Stage3Result,
)
from viewpoint_framework.utils.camera_output_transform import transform_camera_for_output
from viewpoint_framework.utils.cameras import Camera


def make_camera(width=640, height=480):
    angle = np.radians(23.0)
    rotation = np.array([[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0],
                         [-np.sin(angle), 0.0, np.cos(angle)]])
    c2w = np.eye(4)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = [1.2, -0.4, 2.7]
    return Camera(
        index=7, fx=510.0, fy=505.0, cx=(width - 1) / 2 + 1.25,
        cy=(height - 1) / 2 - 0.75, width=width, height=height,
        w2c=np.linalg.inv(c2w), c2w=c2w,
    )


def test_portrait_swaps_viewport_without_changing_extrinsics_or_focal():
    camera = make_camera()
    output, transform = transform_camera_for_output(camera, "auto")
    assert transform == "portrait_viewport"
    assert (output.width, output.height) == (camera.height, camera.width)
    assert (output.fx, output.fy) == (camera.fx, camera.fy)
    np.testing.assert_array_equal(output.w2c, camera.w2c)
    np.testing.assert_array_equal(output.c2w, camera.c2w)
    np.testing.assert_array_equal(output.position, camera.position)
    np.testing.assert_array_equal(output.forward, camera.forward)
    assert output.cx - (output.width - 1) / 2 == camera.cx - (camera.width - 1) / 2
    assert output.cy - (output.height - 1) / 2 == camera.cy - (camera.height - 1) / 2


def test_portrait_input_is_unchanged_and_old_flags_are_compatibility_aliases():
    portrait = make_camera(480, 640)
    unchanged, transform = transform_camera_for_output(portrait, "auto")
    assert transform == "none"
    np.testing.assert_array_equal(unchanged.w2c, portrait.w2c)
    for alias in ("auto_cw90", "auto_ccw90"):
        source = make_camera()
        output, transform = transform_camera_for_output(source, alias)
        assert transform == "portrait_viewport"
        np.testing.assert_array_equal(output.w2c, source.w2c)


def test_stage3_config_defaults_off_and_parses_blocks():
    assert Stage3Config.from_dict({}).output_transform.portrait_output == "off"
    cfg = Stage3Config.from_dict({
        "output_transform": {"portrait_output": "auto"},
        "blocks": {"mode": "content", "references": {"max_refs": 6}},
    })
    assert cfg.output_transform.portrait_output == "auto"
    assert cfg.blocks.mode == "content"


def test_manifest_records_viewport_mapping_without_changing_legacy_schema():
    camera = make_camera()
    candidate = SelectionCandidate(
        candidate_id=4, camera=camera, origin=CandidateOrigin.GRID,
        grid_id=12, row=1, col=2, azimuth_deg=10.0, elevation_deg=-5.0,
        observation_direction=np.array([0.0, 0.0, 1.0]),
    )
    result = Stage3Result(
        selected_candidates=[candidate], selected_cameras=[camera],
        reference_result=ReferenceSelectionResult("test", [], []),
        all_candidates=[candidate], holes=[], hole_views=[],
    )
    legacy_row = build_frame_manifest(result)[0]
    assert "source_camera" not in legacy_row
    assert "output_transform" not in legacy_row
    output_camera, transform = transform_camera_for_output(camera, "auto")
    row = build_frame_manifest(result, [output_camera], [transform])[0]
    assert row["output_transform"]["applied_transform"] == "portrait_viewport"
    np.testing.assert_array_equal(row["source_camera"]["w2c"], row["camera"]["w2c"])


def test_render_sequence_renders_the_portrait_camera_directly(monkeypatch):
    source = make_camera(4, 3)
    called = {}

    class FakeRenderer:
        def render_geometry(self, camera, **_kwargs):
            called["camera"] = camera
            return SimpleNamespace(
                rgb=np.zeros((camera.height, camera.width, 3), dtype=np.float32),
                alpha=np.zeros((camera.height, camera.width), dtype=np.float32),
                depth=np.zeros((camera.height, camera.width), dtype=np.float32),
            )

    written = {}
    monkeypatch.setattr(render_output, "_write_rgb_png",
                        lambda _path, value: written.__setitem__("rgb", value))
    monkeypatch.setattr(render_output, "_write_alpha_png",
                        lambda _path, value: written.__setitem__("alpha", value))
    monkeypatch.setattr(render_output.np, "save",
                        lambda _path, value: written.__setitem__("depth", value))
    from pathlib import Path
    render_camera_sequence(
        FakeRenderer(), [source], Path("."), alpha_dir=Path("."), depth_dir=Path("."),
        portrait_output="auto",
    )
    assert (called["camera"].width, called["camera"].height) == (3, 4)
    np.testing.assert_array_equal(called["camera"].w2c, source.w2c)
    assert written["rgb"].shape == (4, 3, 3)
    assert written["alpha"].shape == (4, 3)
    assert written["depth"].shape == (4, 3)
