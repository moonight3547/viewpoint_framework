import numpy as np

from viewpoint_framework.cameras_util import Camera
from viewpoint_framework.gs_depth_probe import DepthProbeResult
from viewpoint_framework.pose_generation import (
    PoseGenerationConfig,
    camera_to_18d,
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
