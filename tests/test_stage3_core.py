import numpy as np

from viewpoint_framework.cameras_util import Camera
from viewpoint_framework.stage3.selected_views import load_selected_view_set
from viewpoint_framework.stage3.selection import (
    SelectionConfig,
    angular_distance_deg,
    select_views,
)
from viewpoint_framework.stage3.reference_selection import (
    ReferenceSelectionConfig,
    legacy_global_fps,
    target_coverage_greedy,
)
from viewpoint_framework.stage3.types import CandidateOrigin, SelectionCandidate, SelectedViewSet
from viewpoint_framework.stage3.visibility import NullVisibilityModel
from viewpoint_framework.scene_types import CameraMode


def make_camera(index, position, forward):
    position = np.asarray(position, dtype=float)
    forward = np.asarray(forward, dtype=float)
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = -true_up
    c2w[:3, 2] = forward
    c2w[:3, 3] = position
    return Camera(index, 500, 500, 320, 240, 640, 480, np.linalg.inv(c2w), c2w)


def test_angular_distance():
    assert abs(angular_distance_deg([1, 0, 0], [0, 1, 0]) - 90.0) < 1e-8


def test_hole_views_are_forced_into_selection():
    grid = []
    for i, deg in enumerate([0, 45, 90, 135, 180]):
        rad = np.radians(deg)
        direction = np.array([np.sin(rad), 0, np.cos(rad)])
        cam = make_camera(i, 2 * direction, -direction)
        grid.append(
            SelectionCandidate(
                candidate_id=i,
                camera=cam,
                origin=CandidateOrigin.GRID,
                grid_id=i,
                row=0,
                col=i,
                azimuth_deg=deg,
                elevation_deg=0,
                observation_direction=direction,
            )
        )
    hole_cam = make_camera(99, [0, 0, 2], [0.2, 0, -1])
    hole = SelectionCandidate(
        candidate_id=99,
        camera=hole_cam,
        origin=CandidateOrigin.HOLE,
        grid_id=None,
        row=0,
        col=0,
        azimuth_deg=0,
        elevation_deg=0,
        observation_direction=hole_cam.forward,
        forced_select=True,
        hole_id=0,
    )
    selected, _ = select_views(
        grid + [hole],
        num_panos=3,
        captured_seed_cameras=[],
        scene_center=np.zeros(3),
        scene_scale=4.0,
        mode=CameraMode.OUTSIDE_IN,
        config=SelectionConfig(information_gain_strategy="none", near_duplicate_filter=False),
        visibility_model=NullVisibilityModel(),
    )
    assert 99 in [x.candidate_id for x in selected]
    assert len(selected) == 3


def test_legacy_reference_fps_keeps_original_indices():
    cameras = [make_camera(i, [i, 0, 0], [0, 0, 1]) for i in range(8)]
    selected = SelectedViewSet(original_indices=[0, 2, 4, 6], cameras=[cameras[i] for i in [0, 2, 4, 6]])
    result = legacy_global_fps(selected, cameras, num_refs=3)
    assert len(result.original_indices) == 3
    assert set(result.original_indices).issubset({0, 2, 4, 6})
    assert 0 in result.original_indices


def test_target_coverage_greedy_size():
    refs = [make_camera(i, [i - 2, 0, 0], [0, 0, 1]) for i in range(5)]
    targets = [make_camera(100 + i, [x, 0, 1], [0, 0, 1]) for i, x in enumerate([-2, 0, 2])]
    selected = SelectedViewSet(original_indices=list(range(5)), cameras=refs)
    result = target_coverage_greedy(
        selected,
        refs,
        targets,
        num_refs=2,
        scene_scale=5.0,
        config=ReferenceSelectionConfig(strategy="target_coverage_greedy"),
    )
    assert len(result.original_indices) == 2


def test_pointcloud_gaussian_gap_finds_missing_support_cluster():
    from types import SimpleNamespace
    from viewpoint_framework.stage3.hole_detection import (
        HoleDetectionConfig,
        detect_pointcloud_gaussian_gaps,
    )
    from viewpoint_framework.stage3.visibility import PointCloudVisibilityModel, VisibilityConfig

    # Two compact point-cloud patches; 3DGS only supports the first one.
    patch_a = np.array([[0.01 * i, 0.0, 1.0] for i in range(6)], dtype=float)
    patch_b = np.array([[1.0 + 0.01 * i, 0.0, 1.0] for i in range(6)], dtype=float)
    points = np.concatenate([patch_a, patch_b], axis=0)
    model = PointCloudVisibilityModel(
        points,
        scene_scale=2.0,
        config=VisibilityConfig(strategy="pointcloud_visibility", max_samples=100),
    )
    fake_renderer = SimpleNamespace(
        means_np=patch_a.copy(),
        max_scale_np=np.full(len(patch_a), 0.005, dtype=float),
        opacities_np=np.ones(len(patch_a), dtype=float),
    )
    cfg = HoleDetectionConfig(
        strategy="pointcloud_gaussian_gap",
        gap_require_anchor_visibility=False,
        gap_distance_ratio=0.05,
        gap_gaussian_scale_multiplier=1.0,
        voxel_size_ratio=0.10,
        min_cluster_samples=3,
        min_severity_fraction=0.0,
        max_holes=4,
    )
    holes, _ = detect_pointcloud_gaussian_gaps(
        model,
        anchor_cameras=[],
        renderer=fake_renderer,
        scene_scale=2.0,
        config=cfg,
    )
    assert holes
    assert holes[0].centroid[0] > 0.8
