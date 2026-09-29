import numpy as np
from types import SimpleNamespace

from viewpoint_framework.stage3.pipeline import Stage3Config
from viewpoint_framework.stage3 import render_output
from viewpoint_framework.stage3.render_output import (
    build_frame_manifest,
    render_camera_sequence,
)
from viewpoint_framework.stage3.types import (
    CandidateOrigin,
    ReferenceSelectionResult,
    SelectionCandidate,
    Stage3Result,
)
from viewpoint_framework.utils.camera_output_transform import (
    rotate_camera_image_plane,
    rotate_raster_for_output,
    transform_camera_for_output,
)
from viewpoint_framework.utils.cameras import Camera


def make_camera(width=640, height=480):
    angle = np.radians(23.0)
    rotation = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    c2w = np.eye(4)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = [1.2, -0.4, 2.7]
    return Camera(
        index=7, fx=510.0, fy=505.0, cx=311.25, cy=238.75,
        width=width, height=height, w2c=np.linalg.inv(c2w), c2w=c2w,
    )


def project(camera, world_point):
    point = camera.w2c @ np.r_[world_point, 1.0]
    return np.array([
        camera.fx * point[0] / point[2] + camera.cx,
        camera.fy * point[1] / point[2] + camera.cy,
    ])


def test_cw_and_ccw_projection_mapping_and_world_pose_invariants():
    camera = make_camera()
    point = camera.position + 4.0 * camera.forward + 0.3 * camera.right
    old_pixel = project(camera, point)

    cw = rotate_camera_image_plane(camera, "cw90")
    ccw = rotate_camera_image_plane(camera, "ccw90")

    np.testing.assert_allclose(
        project(cw, point), [camera.height - 1 - old_pixel[1], old_pixel[0]],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        project(ccw, point), [old_pixel[1], camera.width - 1 - old_pixel[0]],
        atol=1e-10,
    )
    for transformed in (cw, ccw):
        np.testing.assert_allclose(transformed.position, camera.position, atol=1e-12)
        np.testing.assert_allclose(transformed.forward, camera.forward, atol=1e-12)
        np.testing.assert_allclose(transformed.w2c @ transformed.c2w, np.eye(4), atol=1e-12)
        assert np.linalg.det(transformed.rotation_w2c) > 0.999999
        assert (transformed.width, transformed.height) == (480, 640)


def test_cw_then_ccw_roundtrip_restores_camera():
    camera = make_camera()
    restored = rotate_camera_image_plane(
        rotate_camera_image_plane(camera, "cw90"), "ccw90")
    for name in ("fx", "fy", "cx", "cy", "width", "height"):
        np.testing.assert_allclose(getattr(restored, name), getattr(camera, name))
    np.testing.assert_allclose(restored.w2c, camera.w2c, atol=1e-12)
    np.testing.assert_allclose(restored.c2w, camera.c2w, atol=1e-12)


def test_portrait_auto_only_rotates_landscape_and_rasters_match():
    landscape = make_camera(4, 3)
    portrait = make_camera(3, 4)
    output, rotation = transform_camera_for_output(landscape, "auto_cw90")
    assert rotation == "cw90"
    assert (output.width, output.height) == (3, 4)
    unchanged, rotation = transform_camera_for_output(portrait, "auto_cw90")
    assert rotation == "none"
    np.testing.assert_allclose(unchanged.w2c, portrait.w2c)

    raster = np.arange(3 * 4 * 2).reshape(3, 4, 2)
    np.testing.assert_array_equal(
        rotate_raster_for_output(raster, "cw90"), np.rot90(raster, k=-1))
    np.testing.assert_array_equal(
        rotate_raster_for_output(raster, "ccw90"), np.rot90(raster, k=1))


def test_stage3_config_defaults_off_and_accepts_v34_mode():
    assert Stage3Config.from_dict({}).output_transform.portrait_output == "off"
    cfg = Stage3Config.from_dict({
        "output_transform": {"portrait_output": "auto_ccw90"}
    })
    assert cfg.output_transform.portrait_output == "auto_ccw90"


def test_manifest_off_schema_is_unchanged_and_portrait_records_mapping():
    camera = make_camera()
    candidate = SelectionCandidate(
        candidate_id=4,
        camera=camera,
        origin=CandidateOrigin.GRID,
        grid_id=12,
        row=1,
        col=2,
        azimuth_deg=10.0,
        elevation_deg=-5.0,
        observation_direction=np.array([0.0, 0.0, 1.0]),
    )
    result = Stage3Result(
        selected_candidates=[candidate],
        selected_cameras=[camera],
        reference_result=ReferenceSelectionResult("test", [], []),
        all_candidates=[candidate],
        holes=[],
        hole_views=[],
    )

    legacy_row = build_frame_manifest(result)[0]
    assert "source_camera" not in legacy_row
    assert "output_transform" not in legacy_row

    output_camera, rotation = transform_camera_for_output(camera, "auto_cw90")
    portrait_row = build_frame_manifest(
        result, [output_camera], [rotation]
    )[0]
    assert portrait_row["output_transform"]["applied_rotation"] == "cw90"
    assert portrait_row["source_camera"]["width"] == 640
    assert portrait_row["camera"]["width"] == 480


def test_render_sequence_rotates_rgb_alpha_and_depth_together(monkeypatch):
    camera = make_camera(4, 3)
    rgb = np.arange(3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3) / 100.0
    alpha = np.arange(3 * 4, dtype=np.float32).reshape(3, 4) / 20.0
    depth = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)

    class FakeRenderer:
        def render_geometry(self, *_args, **_kwargs):
            return SimpleNamespace(rgb=rgb, alpha=alpha, depth=depth)

    written = {}
    monkeypatch.setattr(
        render_output, "_write_rgb_png",
        lambda _path, array: written.__setitem__("rgb", np.asarray(array)),
    )
    monkeypatch.setattr(
        render_output, "_write_alpha_png",
        lambda _path, array: written.__setitem__("alpha", np.asarray(array)),
    )
    monkeypatch.setattr(
        render_output.np, "save",
        lambda _path, array: written.__setitem__("depth", np.asarray(array)),
    )
    from pathlib import Path
    existing_dir = Path(".")
    render_camera_sequence(
        FakeRenderer(), [camera], existing_dir,
        alpha_dir=existing_dir, depth_dir=existing_dir,
        portrait_output="auto_cw90",
    )

    np.testing.assert_array_equal(written["rgb"], np.rot90(rgb, k=-1))
    np.testing.assert_array_equal(written["alpha"], np.rot90(alpha, k=-1))
    np.testing.assert_array_equal(written["depth"], np.rot90(depth, k=-1))
