import numpy as np

from viewpoint_framework.cameras_util import Camera
from viewpoint_framework.gs_depth_probe import DepthProbeResult
from viewpoint_framework.pose_generation import (
    PoseGenerationConfig,
    build_view_limits,
    build_v2_dominant_bbox,
    camera_to_18d,
    classify_stage2_binary,
    generate_candidate_poses,
)
from viewpoint_framework.scene_types import (
    CameraMode,
    CameraSceneRelation,
    CenterFitResult,
    CircularInterval,
    GlobalCollectionMode,
    ModeSummary,
    SceneProfile,
    SphericalFrame,
    ViewBBox,
)
from viewpoint_framework.scene_understanding import SceneUnderstandingResult


class _Estimate:
    def __init__(self, radius):
        self.nominal = radius
        self.low = radius
        self.high = radius
        self.confidence = 1.0
        self.valid = True
        self.strategy = "fake"


class _Field:
    def __init__(self, radius):
        self.radius = radius

    def query(self, direction):
        return _Estimate(self.radius)


class _DepthProbe:
    def __init__(self, depth):
        self.depth = depth

    def probe(self, camera):
        return DepthProbeResult(
            valid=True,
            depth=self.depth,
            confidence=1.0,
            valid_pixels=100,
            valid_ratio=1.0,
            depth_q10=self.depth,
            depth_median=self.depth,
            depth_mean=self.depth,
        )


def _camera():
    return Camera(
        index=0,
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        width=100,
        height=100,
        w2c=np.eye(4),
        c2w=np.eye(4),
    )


def _profile(mode):
    center_fit = CenterFitResult(
        center=np.zeros(3),
        residuals=np.zeros(1),
        robust_weights=np.ones(1),
        inlier_mask=np.ones(1, dtype=bool),
        strategy="test",
        solver="test",
        converged=True,
        iterations=1,
        condition_number=1.0,
        singular_values=np.ones(3),
        median_residual=0.0,
        mad_residual=0.0,
    )
    if mode == CameraMode.OUTSIDE_IN:
        mode_summary = ModeSummary(
            dominant_mode=GlobalCollectionMode.OUTSIDE_IN,
            dominant_confidence=1.0,
            outside_in_count=10,
            inside_out_count=0,
            ambiguous_count=0,
            outlier_count=0,
            outside_in_weight=10.0,
            inside_out_weight=0.0,
            outside_in_ratio=1.0,
            inside_out_ratio=0.0,
            strategy="test",
        )
        forward = np.array([0.0, 0.0, -1.0])
    else:
        mode_summary = ModeSummary(
            dominant_mode=GlobalCollectionMode.INSIDE_OUT,
            dominant_confidence=1.0,
            outside_in_count=0,
            inside_out_count=10,
            ambiguous_count=0,
            outlier_count=0,
            outside_in_weight=0.0,
            inside_out_weight=10.0,
            outside_in_ratio=0.0,
            inside_out_ratio=1.0,
            strategy="test",
        )
        forward = np.array([0.0, 0.0, 1.0])

    frame = SphericalFrame(
        x_axis=np.array([1.0, 0.0, 0.0]),
        y_axis=np.array([0.0, 1.0, 0.0]),
        z_axis=np.array([0.0, 0.0, 1.0]),
    )
    azimuth = CircularInterval(0.0, 0.0, 0.0, False)
    bbox = ViewBBox(
        mode=mode,
        strategy="test",
        camera_indices=[0],
        observed_azimuth=azimuth,
        observed_elevation_deg=(0.0, 0.0),
        observed_radius=(1.0, 3.0),
        generation_azimuth=azimuth,
        generation_elevation_deg=(0.0, 0.0),
        generation_radius=(1.0, 3.0),
        angular_extension_deg=0.0,
        radius_extension=0.0,
        support_camera_count=1,
    )
    relation = CameraSceneRelation(
        camera_index=0,
        position=np.array([0.0, 0.0, 1.5]),
        forward=forward,
        radius=1.5,
        radial_direction=np.array([0.0, 0.0, 1.0]),
        lambda_center=1.5 if mode == CameraMode.OUTSIDE_IN else -1.5,
        sight_residual=0.0,
        residual_ratio=0.0,
        alignment_deg=0.0,
        robust_weight=1.0,
        mode=mode,
        confidence=1.0,
        azimuth_deg=0.0,
        elevation_deg=0.0,
    )
    return SceneProfile(
        center_fit=center_fit,
        mode_summary=mode_summary,
        coordinate_frame=frame,
        camera_relations=[relation],
        view_bboxes={mode.value: bbox},
        radius_field_samples={},
        strategy_config={},
    )


def test_outside_in_depth_backoff_reaches_radius_max():
    profile = _profile(CameraMode.OUTSIDE_IN)
    scene_result = SceneUnderstandingResult(
        profile=profile,
        radius_fields={CameraMode.OUTSIDE_IN.value: _Field(1.5)},
    )
    result = generate_candidate_poses(
        captured_cameras=[_camera()],
        scene_result=scene_result,
        depth_probe=_DepthProbe(10.0),
        config=PoseGenerationConfig(),
    )
    camera = result.valid_cameras[0]
    assert np.allclose(camera.position, [0.0, 0.0, 3.0])
    assert np.allclose(camera.forward, [0.0, 0.0, -1.0])
    assert np.linalg.det(camera.c2w[:3, :3]) > 0.999
    assert len(camera_to_18d(camera)) == 18


def test_inside_out_cross_center_keeps_inside_out_forward():
    profile = _profile(CameraMode.INSIDE_OUT)
    scene_result = SceneUnderstandingResult(
        profile=profile,
        radius_fields={CameraMode.INSIDE_OUT.value: _Field(1.5)},
    )
    result = generate_candidate_poses(
        captured_cameras=[_camera()],
        scene_result=scene_result,
        depth_probe=_DepthProbe(4.0),
        config=PoseGenerationConfig(),
    )
    camera = result.valid_cameras[0]
    assert camera.position[2] < -1.0
    assert np.allclose(camera.forward, [0.0, 0.0, 1.0])
    assert result.candidates[0].crossed_center


def _camera_at(index, position, forward):
    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray(forward, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(up, forward))) > 0.95:
        up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = -true_up
    c2w[:3, 2] = forward
    c2w[:3, 3] = position
    return Camera(
        index=index,
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        width=100,
        height=100,
        w2c=np.linalg.inv(c2w),
        c2w=c2w,
    )


def test_v2_binary_bbox_expands_past_180_degrees_without_minimum_extension():
    profile = _profile(CameraMode.OUTSIDE_IN)
    cameras = []
    for i, angle_deg in enumerate((-110.0, -30.0, 30.0, 110.0)):
        angle = np.radians(angle_deg)
        position = np.array([np.sin(angle), 0.0, np.cos(angle)]) * float(i + 1)
        cameras.append(_camera_at(i, position, -position))
    relations = classify_stage2_binary(cameras, np.zeros(3))
    config = PoseGenerationConfig(
        mode_strategy="binary_count_majority",
        bbox_strategy="binary_dominant",
        azimuth_extension_ratio=0.1,
        elevation_extension_ratio=0.1,
        view_limits_strategy="v2_observed_dominant_radius",
        view_limits_radius_percentiles=(40.0, 60.0),
    )
    bbox, support = build_v2_dominant_bbox(
        relations, CameraMode.OUTSIDE_IN, profile, config
    )

    assert len(support) == len(cameras)
    assert bbox.observed_azimuth.span_deg > 180.0
    assert bbox.generation_azimuth.span_deg > bbox.observed_azimuth.span_deg
    assert bbox.generation_azimuth.span_deg <= 360.0
    assert bbox.generation_elevation_deg == (0.0, 0.0)
    limits = build_view_limits(profile, bbox, config, dominant_relations=support)
    assert np.isclose(limits["minRadius"], 2.2)
    assert np.isclose(limits["maxRadius"], 2.8)


def test_v2_pose_generation_uses_binary_dominant_bbox_end_to_end():
    profile = _profile(CameraMode.OUTSIDE_IN)
    cameras = []
    for i, angle_deg in enumerate((-60.0, -20.0, 20.0, 60.0)):
        angle = np.radians(angle_deg)
        position = np.array([np.sin(angle), 0.0, np.cos(angle)]) * float(i + 1)
        cameras.append(_camera_at(i, position, -position))
    scene_result = SceneUnderstandingResult(
        profile=profile,
        radius_fields={
            CameraMode.OUTSIDE_IN.value: _Field(2.5),
            CameraMode.INSIDE_OUT.value: _Field(2.5),
        },
    )
    config = PoseGenerationConfig(
        mode_strategy="binary_count_majority",
        bbox_strategy="binary_dominant",
        azimuth_step_deg=20.0,
        elevation_step_deg=20.0,
        outside_placement_strategy="prior_only",
        view_limits_strategy="v2_observed_dominant_radius",
    )
    config.geometry.strategy = "none"
    result = generate_candidate_poses(
        captured_cameras=cameras,
        scene_result=scene_result,
        point_cloud_points=None,
        config=config,
    )

    assert result.mode.mode == CameraMode.OUTSIDE_IN
    assert result.bbox.support_camera_count == len(cameras)
    assert result.diagnostics["angular_grid_count"] == len(result.candidates)
    assert len(result.valid_cameras) == len(result.candidates)
    assert np.isclose(result.view_limits["minRadius"], 2.2)
    assert np.isclose(result.view_limits["maxRadius"], 2.8)
