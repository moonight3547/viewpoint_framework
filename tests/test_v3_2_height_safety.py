"""Focused V3.2 trajectory-radius and vertical-safety contracts."""
from types import SimpleNamespace

import numpy as np
import pytest

from viewpoint_framework.height_safety import (
    LOCAL_HOLE_UNCERTAIN,
    LOCAL_RELIABLE,
    EffectiveHeightLimits,
    GlobalHeightConfig,
    HeightSideProbe,
    LocalHeightProbeResult,
    build_global_height_limits,
    clip_radius_to_height_limit,
    resolve_effective_height,
    unavailable_local_height,
)
from viewpoint_framework.scene_types import SphericalFrame
from viewpoint_framework.tests.test_pose_generation_synthetic import _camera_at
from viewpoint_framework.trajectory_safe_field import (
    TrajectorySafeField,
    TrajectorySafeFieldConfig,
)


FRAME = SphericalFrame(
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
)


def _field(positions, **overrides):
    config = TrajectorySafeFieldConfig(
        rho_strategy="segment_ray_min", **overrides,
    )
    cameras = [_camera_at(i, p, [0.0, 0.0, 1.0]) for i, p in enumerate(positions)]
    return TrajectorySafeField(cameras, np.zeros(3), FRAME, config)


def _side(limit, status):
    return HeightSideProbe(20, 0.8, 0.2, 9, 1.0, True, limit, limit, status)


def _local(azimuth, lower, upper, lower_status=LOCAL_RELIABLE,
           upper_status=LOCAL_RELIABLE, extension=False):
    return LocalHeightProbeResult(
        azimuth_deg=azimuth, rho=2.0,
        trajectory_height_min=-0.2, trajectory_height_max=0.2,
        lower_raw=lower, upper_raw=upper,
        lower_safe=lower, upper_safe=upper,
        down_probe=_side(lower, lower_status), up_probe=_side(upper, upper_status),
        lower_status=lower_status, upper_status=upper_status,
        is_extension_column=extension,
    )


def test_segment_ray_uses_minimum_positive_continuous_intersection():
    # Both continuous edges cross phi=0; the nearer hit must win.
    field = _field([
        [-1.0, 0.0, 1.0], [1.0, 2.0, 1.0],
        [1.0, 2.0, 3.0], [-1.0, 4.0, 3.0],
    ], max_step_multiplier=100.0)
    hit = field.query_segment_ray_min(0.0)
    assert hit.direct_intersection_count == 2
    assert hit.rho == pytest.approx(1.0)
    assert hit.trajectory_cross_height == pytest.approx(1.0)
    assert hit.selected_segment_start_index == 0


def test_segment_ray_never_bridges_rejected_trajectory_jump():
    field = _field([
        [-0.1, 0.0, 1.0], [0.1, 0.0, 1.0],
        [0.2, 0.0, 1.0], [0.3, 0.0, 1.0], [0.4, 0.0, 1.0],
        [20.0, 0.0, 20.0], [-20.0, 0.0, 20.0],
    ])
    hit = field.query_segment_ray_min(0.0)
    assert field.jump_rejected_count >= 1
    assert hit.rho == pytest.approx(1.0)


def test_no_direct_hit_uses_low_confidence_v31_fallback_without_phi_collapse():
    field = _field([[0.0, 0.0, 2.0]], fallback_confidence_threshold=0.9)
    hit = field.query_segment_ray_min(5.0)
    assert hit.direct_intersection_count == 0
    assert hit.rho == pytest.approx(2.0)
    assert hit.fallback_low_confidence
    assert hit.rho_source == "trajectory_fallback_low_confidence"


def test_global_interval_is_strict_and_excludes_extension_columns():
    locals_ = [
        _local(-20.0, -1.0, 2.0),
        _local(20.0, -0.5, 1.5),
        _local(40.0, 10.0, 10.1, extension=True),
    ]
    limits = build_global_height_limits(
        locals_, [-2.0, 0.0, 3.0], GlobalHeightConfig(enabled=True),
    )
    assert limits.height_min == pytest.approx(-0.5)
    assert limits.height_max == pytest.approx(1.5)
    assert limits.lower_contributor_azimuth == pytest.approx(20.0)
    assert limits.upper_contributor_azimuth == pytest.approx(20.0)


def test_global_sides_fallback_independently_to_captured_height_range():
    local = _local(0.0, None, 1.2, lower_status="LOCAL_UNAVAILABLE")
    limits = build_global_height_limits([local], [-2.0, 3.0])
    assert limits.height_min == pytest.approx(-2.0)
    assert limits.height_max == pytest.approx(1.2)
    assert limits.fallback_to_captured_lower
    assert not limits.fallback_to_captured_upper


def test_global_conflict_falls_back_to_complete_captured_range():
    with pytest.warns(RuntimeWarning, match="GLOBAL_HEIGHT_INTERVAL_CONFLICT"):
        limits = build_global_height_limits(
            [_local(0.0, 2.0, 1.0)], [-3.0, 4.0],
        )
    assert limits.conflict
    assert (limits.height_min, limits.height_max) == pytest.approx((-3.0, 4.0))


def test_hole_uncertain_takes_stricter_local_global_limit():
    local = _local(
        0.0, -1.0, 2.0,
        lower_status=LOCAL_HOLE_UNCERTAIN,
        upper_status=LOCAL_HOLE_UNCERTAIN,
    )
    global_limits = SimpleNamespace(height_min=-0.5, height_max=1.5)
    effective = resolve_effective_height(local, global_limits)
    assert (effective.height_min, effective.height_max) == pytest.approx((-0.5, 1.5))


def test_extension_column_directly_uses_global_limits():
    local = unavailable_local_height(30.0, 2.0, -0.1, 0.1, extension=True)
    global_limits = SimpleNamespace(height_min=-1.0, height_max=2.0)
    effective = resolve_effective_height(local, global_limits)
    assert effective == EffectiveHeightLimits(-1.0, 2.0, "global_extension", "global_extension")


@pytest.mark.parametrize(
    "radius,direction,limits,expected",
    [
        (4.0, [0.0, 0.5, np.sqrt(0.75)], (-1.0, 1.0), 2.0),
        (-4.0, [0.0, 0.5, np.sqrt(0.75)], (-1.0, 1.0), -2.0),
    ],
)
def test_height_correction_only_shrinks_absolute_signed_radius(
        radius, direction, limits, expected):
    clipped = clip_radius_to_height_limit(
        radius, direction, [0.0, 1.0, 0.0], *limits,
    )
    assert clipped.reachable and clipped.clipped
    assert clipped.radius == pytest.approx(expected)
    assert abs(clipped.radius) <= abs(radius)


def test_height_limit_that_requires_outward_motion_is_unreachable():
    clipped = clip_radius_to_height_limit(
        1.0, [0.0, 0.5, np.sqrt(0.75)], [0.0, 1.0, 0.0], 2.0, 3.0,
    )
    assert not clipped.reachable
    assert clipped.radius == pytest.approx(1.0)
