#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Synthetic geometry tests for scene-understanding strategies."""

import numpy as np

from viewpoint_framework.cameras_util import Camera
from viewpoint_framework.scene_types import GlobalCollectionMode
from viewpoint_framework.radius_field import DirectionalRadiusField, RadiusFieldConfig
from viewpoint_framework.scene_types import CameraMode, CameraSceneRelation
from viewpoint_framework.scene_understanding import (
    SceneUnderstandingConfig,
    understand_scene,
)
from viewpoint_framework.view_space import minimal_circular_interval


def _camera(index: int, position: np.ndarray, forward: np.ndarray) -> Camera:
    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray(forward, dtype=np.float64)
    forward = forward / np.linalg.norm(forward)

    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(forward, up))) > 0.95:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    true_up = true_up / np.linalg.norm(true_up)

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = -true_up
    c2w[:3, 2] = forward
    c2w[:3, 3] = position
    w2c = np.linalg.inv(c2w)

    return Camera(
        index=index,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
        w2c=w2c,
        c2w=c2w,
    )


def _ring(center, inside_out=False, count=16):
    cameras = []
    for index, angle_deg in enumerate(np.linspace(-150.0, 150.0, count)):
        angle = np.radians(angle_deg)
        direction = np.array(
            [np.sin(angle), 0.1 * np.sin(2.0 * angle), np.cos(angle)],
            dtype=np.float64,
        )
        direction /= np.linalg.norm(direction)
        radius = 2.0 + 0.2 * np.cos(angle)
        position = center + radius * direction
        forward = direction if inside_out else -direction
        cameras.append(_camera(index, position, forward))
    return cameras


def test_robust_center_rejects_one_bad_camera():
    center = np.array([1.2, -0.7, 2.5], dtype=np.float64)
    cameras = _ring(center, inside_out=False, count=20)
    cameras.append(
        _camera(
            20,
            center + np.array([4.0, 4.0, 4.0]),
            np.array([1.0, 0.0, 0.0]),
        )
    )

    result = understand_scene(cameras, SceneUnderstandingConfig.default())

    assert np.linalg.norm(result.profile.center_fit.center - center) < 1e-3
    assert result.profile.mode_summary.outside_in_count >= 19
    assert result.profile.mode_summary.outlier_count >= 1
    assert result.profile.mode_summary.dominant_mode == GlobalCollectionMode.OUTSIDE_IN


def test_inside_out_classification():
    center = np.array([0.3, 0.2, -0.4], dtype=np.float64)
    cameras = _ring(center, inside_out=True, count=12)

    result = understand_scene(cameras, SceneUnderstandingConfig.default())

    assert np.linalg.norm(result.profile.center_fit.center - center) < 1e-6
    assert result.profile.mode_summary.inside_out_count == len(cameras)
    assert result.profile.mode_summary.dominant_mode == GlobalCollectionMode.INSIDE_OUT


def test_v2_center_refinement_runs_once_and_preserves_four_way_labels():
    center = np.array([0.5, -0.2, 1.0], dtype=np.float64)
    cameras = _ring(center, inside_out=False, count=12)
    config = SceneUnderstandingConfig.default()
    config.center_refinement.strategy = "dominant_mode_once"
    config.center_refinement.min_support_ratio = 0.2
    config.radius.support_strategy = "all_cameras"

    result = understand_scene(cameras, config)
    metadata = result.profile.metadata["center_refinement"]

    assert metadata["applied"]
    assert metadata["dominant_mode"] == "outside_in"
    assert metadata["support_count"] == len(cameras)
    assert result.profile.mode_summary.outside_in_count == len(cameras)
    assert np.linalg.norm(result.profile.center_fit.center - center) < 1e-6
    assert "outside_in" in result.radius_fields
    assert "inside_out" in result.radius_fields


def test_v2_radius_field_uses_all_camera_positions_regardless_of_mode():
    relations = []
    for index, (mode, radius) in enumerate(
        ((CameraMode.OUTSIDE_IN, 1.0), (CameraMode.AMBIGUOUS, 2.0), (CameraMode.INSIDE_OUT, 3.0))
    ):
        relations.append(
            CameraSceneRelation(
                camera_index=index,
                position=np.array([0.0, 0.0, radius]),
                forward=np.array([0.0, 0.0, -1.0]),
                radius=radius,
                radial_direction=np.array([0.0, 0.0, 1.0]),
                lambda_center=radius,
                sight_residual=0.0,
                residual_ratio=0.0,
                alignment_deg=0.0,
                robust_weight=1.0,
                mode=mode,
                confidence=1.0,
            )
        )
    field = DirectionalRadiusField(
        center=np.zeros(3),
        mode=CameraMode.OUTSIDE_IN,
        relations=relations,
        config=RadiusFieldConfig(
            support_strategy="all_cameras",
            min_neighbors=1,
        ),
    )

    assert field.num_support_cameras == 3


def test_circular_bbox_wraparound():
    interval = minimal_circular_interval([175.0, 179.0, -178.0, -174.0])
    assert interval.wraps
    assert interval.span_deg == 11.0


def test_strict_legacy_first_camera_target_behavior():
    """Legacy check_camera_alignment can accept origin even if true center differs."""

    true_center = np.array([1.0, 0.0, 0.0])

    # First camera deliberately looks exactly toward world origin, reproducing
    # the historical early-accept condition.
    first_position = np.array([0.0, 0.0, 2.0])
    first_forward = -first_position / np.linalg.norm(first_position)
    cameras = [_camera(0, first_position, first_forward)]
    cameras.extend(_ring(true_center, inside_out=False, count=8))

    config = SceneUnderstandingConfig.legacy_camera_only()
    result = understand_scene(cameras, config)

    assert np.linalg.norm(result.profile.center_fit.center) < 1e-8
    assert result.profile.center_fit.strategy == "legacy_check_alignment"
