"""V3.0 contract tests, including explicit differences from V2."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import viewpoint_framework.geometry_safety as safety_module

from viewpoint_framework.geometry_safety import (
    GeometrySafetyConfig, PointCloudSafety, local_clearance_threshold,
)
from viewpoint_framework.pose_generation import (
    PoseGenerationConfig, generate_candidate_poses, save_pose_generation_result,
)
from viewpoint_framework.pose_generation_v3 import initial_position_from_rho
from viewpoint_framework.scene_types import CameraMode, SphericalFrame
from viewpoint_framework.scene_understanding import (
    SceneUnderstandingResult, SceneUnderstandingConfig, understand_scene,
)
from viewpoint_framework.stage3.pipeline import candidates_from_files, candidates_from_pose_result
from viewpoint_framework.trajectory_safe_field import TrajectorySafeField, TrajectorySafeFieldConfig
from viewpoint_framework.tests.test_pose_generation_synthetic import (
    _DepthProbe, _Field, _camera_at, _profile,
)


ROOT = Path(__file__).resolve().parents[1]
FRAME = SphericalFrame(np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.]))
FAR_POINTS = np.array([[20., 20., 20.], [20., 20., 20.01], [20., 20.01, 20.]])
NATIVE_OPEN3D = safety_module.o3d


@pytest.fixture(autouse=True)
def exact_knn_if_open3d_unavailable(monkeypatch):
    """Exercise real safety code with exact small-array KNN when CI lacks Open3D.

    Only the external tree primitive is replaced: distances, thresholds, path
    sampling and final validation are all evaluated by the production code.
    """
    if NATIVE_OPEN3D is not None:
        return

    class ExactTree:
        def __init__(self, cloud):
            self.points = np.asarray(cloud.points)

        def search_knn_vector_3d(self, position, k):
            dist2 = np.sum((self.points - position) ** 2, axis=1)
            indices = np.argsort(dist2)[:k]
            return len(indices), indices.tolist(), dist2[indices].tolist()

    monkeypatch.setattr(safety_module, "o3d", SimpleNamespace(
        geometry=SimpleNamespace(PointCloud=SimpleNamespace, KDTreeFlann=ExactTree),
        utility=SimpleNamespace(Vector3dVector=np.asarray),
    ))


@pytest.mark.skipif(NATIVE_OPEN3D is None, reason="Open3D not installed; numerical KNN adapter tested separately")
def test_native_open3d_local_query():
    safety = PointCloudSafety(np.array([[0., 0., 0.], [.01, 0., 0.], [-.01, 0., 0.]]))
    assert safety.is_position_safe(np.array([0., 1., 0.]), .05)[0]
    assert not safety.is_position_safe(np.zeros(3), .05)[0]


def config_v3():
    cfg = PoseGenerationConfig.from_dict(json.loads((ROOT / "configs/v3_pose_generation.json").read_text()))
    cfg.mode_strategy = "count_majority"
    cfg.bbox_strategy = "scene_profile"
    cfg.view_limits_strategy = "generation_bbox"
    cfg.console_log_candidates = False
    return cfg


def run_case(mode=CameraMode.OUTSIDE_IN, elevation=0., depth=None, points=None, cfg=None):
    profile = _profile(mode)
    profile.center_fit.center = np.array([4., 7., -2.])
    profile.view_bboxes[mode.value].generation_elevation_deg = (elevation, elevation)
    angle = np.radians(elevation)
    position = profile.center_fit.center + np.array([0., 1.5 * np.tan(angle), 1.5])
    radial = position - profile.center_fit.center
    camera = _camera_at(0, position, -radial if mode == CameraMode.OUTSIDE_IN else radial)
    scene = SceneUnderstandingResult(profile=profile, radius_fields={mode.value: _Field(1.5)})
    return generate_candidate_poses(
        [camera], scene, FAR_POINTS if points is None else points,
        depth_probe=depth, config=cfg or config_v3(),
    )


def trajectory(positions, cfg=None):
    cams = [_camera_at(i, p, [0., 0., 1.]) for i, p in enumerate(positions)]
    return TrajectorySafeField(cams, np.zeros(3), FRAME, cfg)


@pytest.mark.parametrize("elevation", [-30., 0., 30., 60.])
def test_horizontal_rho_and_positional_elevation(elevation):
    result = run_case(elevation=elevation)
    camera = result.valid_cameras[0]
    local = camera.position - result.view_limits["target"]
    assert np.isclose(np.hypot(local[0], local[2]), 1.5)
    assert np.isclose(local[1], 1.5 * np.tan(np.radians(elevation)))
    assert np.isclose(np.dot(camera.forward, local / np.linalg.norm(local)), -1.)
    assert result.candidates[0].geometry_metadata["height_guard_enabled"] is False


def test_position_formula_uses_scene_up_not_world_y():
    center, up, horizontal = np.array([2., 3., 4.]), np.array([1., 0., 0.]), np.array([0., 1., 0.])
    point, radius, direction = initial_position_from_rho(center, horizontal, up, 2., 45.)
    np.testing.assert_allclose(point, [4., 5., 4.])
    np.testing.assert_allclose(point - center, radius * direction)


@pytest.mark.parametrize("angle", [90., -90., np.nan])
def test_position_formula_rejects_singular_elevation(angle):
    with pytest.raises(ValueError):
        initial_position_from_rho(np.zeros(3), FRAME.z_axis, FRAME.y_axis, 1., angle)


def test_v3_outlier_leaves_clearance_and_prior_unchanged():
    a = run_case(points=FAR_POINTS)
    b = run_case(points=np.vstack((FAR_POINTS, [[10000., 0., 0.]])))
    assert a.candidates[0].geometry_metadata["local_clearance_threshold"] == pytest.approx(0.075)
    assert b.candidates[0].geometry_metadata["local_clearance_threshold"] == pytest.approx(0.075)
    np.testing.assert_allclose(a.valid_cameras[0].position, b.valid_cameras[0].position)
    # Legacy AABB scaling remains observable when V3 is not selected.
    s1 = PointCloudSafety(FAR_POINTS)
    s2 = PointCloudSafety(np.vstack((FAR_POINTS, [[10000., 0., 0.]])))
    assert s2.clearance > s1.clearance * 100


def test_clearance_bounds_and_explicit_queries():
    cfg = GeometrySafetyConfig(clearance_strategy="local_horizontal_radius")
    assert local_clearance_threshold(.001, 2., cfg) == pytest.approx(.02)
    assert local_clearance_threshold(100., 2., cfg) == pytest.approx(.2)
    safety = PointCloudSafety(np.zeros((3, 3)), cfg)
    with pytest.raises(ValueError, match="required_clearance"):
        safety.is_position_safe([1., 0., 0.])
    assert safety.is_position_safe([.1, 0., 0.], .05)[0]
    assert not safety.is_position_safe([.1, 0., 0.], .2)[0]


def test_unsafe_initial_is_rejected_before_depth():
    class MustNotRender:
        def probe(self, camera):
            raise AssertionError("Unsafe initial must never invoke depth/repair.")
    initial = np.array([4., 7., -.5])
    result = run_case(points=np.tile(initial, (3, 1)), depth=MustNotRender())
    assert not result.valid_cameras
    candidate = result.candidates[0]
    assert candidate.reject_reason == "UNSAFE_INITIAL_PRIOR"
    assert candidate.depth_probe is None
    assert candidate.geometry_metadata["adjustment_attempted"] is False


@pytest.mark.parametrize("mode", [CameraMode.OUTSIDE_IN, CameraMode.INSIDE_OUT])
def test_depth_exceeding_upper_radius_keeps_exact_prior(mode):
    result = run_case(mode=mode, depth=_DepthProbe(100.))
    c = result.candidates[0]
    assert c.final_signed_radius == pytest.approx(c.initial_radius)
    assert c.geometry_metadata["adjustment_radius_exceeded"]
    assert c.geometry_metadata["adjustment_skip_reason"] == "ADJUSTMENT_EXCEEDS_RADIUS_MAX"
    assert c.geometry_metadata["final_geometry_safe"]
    np.testing.assert_allclose(c.geometry_metadata["initial_position"], c.camera.position)


def test_explicit_radius_cap_allows_or_skips_identical_proposal():
    cfg = config_v3()
    cfg.adjustment_radius_max = 2.
    low = run_case(depth=_DepthProbe(1.), cfg=cfg)
    cfg.adjustment_radius_max = 3.
    high = run_case(depth=_DepthProbe(1.), cfg=cfg)
    assert low.candidates[0].final_signed_radius == pytest.approx(1.5)
    assert high.candidates[0].final_signed_radius == pytest.approx(2.425)


def test_prior_is_not_clipped_to_global_upper_bound():
    cfg = config_v3()
    cfg.adjustment_radius_max = 1.
    c = run_case(cfg=cfg, depth=_DepthProbe(5.)).candidates[0]
    assert c.initial_radius == pytest.approx(1.5)
    assert c.final_signed_radius == pytest.approx(1.5)
    assert c.depth_probe is None
    assert c.geometry_metadata["initial_radius_exceeds_adjustment_max"]


def test_inside_crossing_passes_actual_3d_center_and_points_outward():
    result = run_case(mode=CameraMode.INSIDE_OUT, elevation=30., depth=_DepthProbe(3.5))
    c = result.candidates[0]
    center = np.asarray(result.view_limits["target"])
    start = np.asarray(c.geometry_metadata["initial_position"]) - center
    end = c.camera.position - center
    assert c.crossed_center
    assert start[1] > 0 and end[1] < 0  # no constant-height crossing
    np.testing.assert_allclose(np.cross(start, end), np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(c.camera.forward, end / np.linalg.norm(end))
    assert np.dot(c.camera.forward, center - c.camera.position) < 0
    assert c.elevation_deg == pytest.approx(-30.)
    assert c.geometry_metadata["grid_elevation_deg"] == 30.


def test_final_collision_cannot_bypass_check_when_path_disabled():
    cfg = config_v3()
    cfg.use_path_safety = False
    # r0=1.5, clearance=.075, probe=1 -> r_final=2.425.
    obstruction = np.tile([4., 7., .425], (3, 1))
    result = run_case(points=obstruction, depth=_DepthProbe(1.), cfg=cfg)
    assert not result.valid_cameras
    c = result.candidates[0]
    assert c.geometry_metadata["proposal_safe"]
    assert c.reject_reason == "FINAL_POSITION_TOO_CLOSE_TO_GEOMETRY"


def test_path_clipping_still_uses_candidate_local_clearance():
    # An intermediate obstacle clips V2-style motion despite a clear endpoint.
    obstacle = np.tile([4., 7., -.1], (3, 1))
    c = run_case(points=obstacle, depth=_DepthProbe(1.)).candidates[0]
    assert c.camera is not None
    assert c.path_safe_fraction < 1.
    assert c.final_clearance >= c.geometry_metadata["local_clearance_threshold"]


@pytest.mark.parametrize("depth", [np.nan, np.inf, -1., 0.])
def test_invalid_depth_keeps_safe_prior(depth):
    c = run_case(depth=_DepthProbe(depth)).candidates[0]
    assert c.camera is not None
    assert c.final_signed_radius == c.initial_radius
    assert not c.depth_probe.valid


def test_height_guard_is_opt_in_and_rejects_instead_of_repositioning():
    # Crossing at el=30 flips the candidate height relative to the captured band.
    cfg = config_v3()
    default = run_case(mode=CameraMode.INSIDE_OUT, elevation=30., depth=_DepthProbe(3.5), cfg=cfg)
    assert default.valid_cameras
    cfg.height_guard.enabled = True
    guarded = run_case(mode=CameraMode.INSIDE_OUT, elevation=30., depth=_DepthProbe(3.5), cfg=cfg)
    assert not guarded.valid_cameras
    assert guarded.candidates[0].reject_reason == "HEIGHT_OUT_OF_TRAJECTORY_RANGE"


def test_disconnected_branches_do_not_average_radii():
    positions = [[0., 0., 1.], [.01, 0., 1.], [.02, 0., 1.],
                 [0., 1., 3.], [.01, 1., 3.], [.02, 1., 3.]]
    field = trajectory(positions)
    estimate = field.query(0.)
    assert field.branch_count == 2
    assert field.jump_rejected_count == 1
    assert len(estimate.intervals) == 2
    assert all(not (i.rho_min < 2. < i.rho_max) for i in estimate.intervals)
    assert {i.height_preferred for i in estimate.intervals} == {0., 1.}


def test_long_jump_does_not_generate_interpolated_corridor():
    field = trajectory([[0, 0, 1], [.01, 0, 1], [.02, 0, 1],
                        [100, 0, 1], [.03, 0, 1], [.04, 0, 1], [.05, 0, 1]])
    assert field.jump_rejected_count == 2
    assert not np.any((field.samples[:, 1] > 2.) & (field.samples[:, 1] < 90.))


def test_empty_and_duplicate_trajectory_and_wraparound():
    assert trajectory([]).query(0.).source == "NO_TRAJECTORY_SUPPORT"
    repeated = trajectory([[0., 0., 1.]] * 3)
    assert repeated.typical_step == 0
    assert repeated.query(0.).selected_interval.rho_preferred == pytest.approx(1.)
    wrapped = trajectory([[.01, 0, -1], [-.01, 0, -1]])
    assert wrapped.query(-180.).selected_interval is not None
    assert wrapped.query(180.).selected_interval is not None
    assert repeated.query(20.).source == "trajectory_nearest_limited"
    assert repeated.query(60.).selected_interval is None


def test_nonfinite_camera_breaks_sequence():
    cameras = [_camera_at(i, p, [0., 0., 1.]) for i, p in enumerate([[0., 0., 1.], [0., 0., 2.], [0., 0., 3.]])]
    cameras[1].c2w[0, 3] = np.nan
    field = TrajectorySafeField(cameras, np.zeros(3), FRAME)
    assert field.invalid_pose_count == 1
    assert field.branch_count == 2
    assert len(field.query(0.).intervals) == 2


def test_supported_segment_resamples_in_space_before_horizontal_projection():
    field = trajectory([[-.1, 1., 1.], [.1, 1., 1.]])
    # The spatial midpoint lies closer to the center than the endpoints.
    assert len(field.samples) == 3
    midpoint = field.samples[np.argmin(np.abs(field.samples[:, 0]))]
    assert midpoint[1] == pytest.approx(1.)
    assert field.samples[:, 1].max() > midpoint[1]
    assert field.query(0.).selected_interval.height_preferred == pytest.approx(1.)


@pytest.mark.parametrize("mode", [CameraMode.OUTSIDE_IN, CameraMode.INSIDE_OUT])
def test_v2_scene_and_unmodified_v3_json_configs_end_to_end(mode):
    cameras = []
    for i, az in enumerate(np.linspace(-60., 60., 13)):
        angle = np.radians(az)
        position = np.array([2 * np.sin(angle), .2 * np.sin(2 * angle), 2 * np.cos(angle)])
        cameras.append(_camera_at(i, position, -position if mode == CameraMode.OUTSIDE_IN else position))
    scene_config = SceneUnderstandingConfig.from_dict(json.loads(
        (ROOT / "configs/v2_scene_understanding.json").read_text()))
    scene = understand_scene(cameras=cameras, config=scene_config, point_cloud_points=FAR_POINTS)
    cfg = PoseGenerationConfig.from_dict(json.loads((ROOT / "configs/v3_pose_generation.json").read_text()))
    result = generate_candidate_poses(cameras, scene, FAR_POINTS, config=cfg)
    assert result.mode.mode == mode
    assert len(result.valid_cameras) > 0
    assert all(c.geometry_metadata["final_geometry_safe"] for c in result.candidates if c.camera)
    for camera in result.valid_cameras:
        radial = camera.position - scene.profile.center_fit.center
        polarity = np.dot(camera.forward, radial / np.linalg.norm(radial))
        assert polarity == pytest.approx(-1. if mode == CameraMode.OUTSIDE_IN else 1.)


def test_v3_requires_actual_geometry_and_local_strategy():
    with pytest.raises(ValueError, match="point-cloud"):
        cfg = config_v3()
        cfg.geometry.strategy = "none"
        run_case(cfg=cfg)


def test_v3_metadata_and_stage3_adapters(tmp_path):
    result = run_case(mode=CameraMode.INSIDE_OUT, elevation=30., depth=_DepthProbe(3.5))
    paths = save_pose_generation_result(result, str(tmp_path))
    from_file, mode = candidates_from_files(paths["gen_cameras"], paths["gen_cameras_meta"])
    from_memory = candidates_from_pose_result(result)
    assert mode == CameraMode.INSIDE_OUT
    assert len(from_file) == len(from_memory) == 1
    np.testing.assert_allclose(from_file[0].camera.forward, from_memory[0].camera.forward)
    np.testing.assert_allclose(from_file[0].observation_direction, from_file[0].camera.forward)
    meta = json.loads(Path(paths["gen_cameras_meta"]).read_text())
    assert meta["candidates"][0]["geometry_metadata"]["final_geometry_safe"]
    assert meta["diagnostics"]["initial_geometry_unsafe_count"] == 0
    assert len(json.loads(Path(paths["gen_cameras"]).read_text())[0]) == 18


def test_v2_json_config_still_clamps_far_depth_to_global_max():
    cfg = PoseGenerationConfig.from_dict(json.loads((ROOT / "configs/v2_pose_generation.json").read_text()))
    assert cfg.position_strategy == "legacy"
    assert cfg.geometry.clearance_strategy == "pointcloud_aabb"
    assert not cfg.reject_unsafe_initial_prior
    cfg.console_log_candidates = False
    result = run_case(cfg=cfg, depth=_DepthProbe(100.))
    c = result.candidates[0]
    assert c.final_signed_radius == pytest.approx(max(result.bbox.generation_radius))
    assert c.geometry_metadata == {}
