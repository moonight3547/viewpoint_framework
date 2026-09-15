import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from viewpoint_framework.gs_depth_probe import DepthProbeConfig, GsplatDepthProbe
from viewpoint_framework.gs_renderer import (
    GaussianRenderResult,
    GsplatRenderer,
    RendererNearPlaneConfig,
    resolve_renderer_near_plane,
)
from viewpoint_framework.skybox_detection import (
    SkyboxDetectionConfig,
    detect_skybox_gaussians,
)
from viewpoint_framework.stage3.visibility import GaussianVisibilityModel, VisibilityConfig
from viewpoint_framework.tests.test_pose_generation_synthetic import _camera_at


def sphere_points(count, radius, center=(0., 0., 0.)):
    i = np.arange(count, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * (i + 0.5) / count
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    points = np.stack((np.cos(phi * i) * r, y, np.sin(phi * i) * r), axis=1)
    return np.asarray(center) + radius * points


def test_synthetic_skybox_uses_its_own_center_and_half_percent_band():
    rng = np.random.default_rng(4)
    skybox_center = np.array([3.0, -2.0, 5.0])
    sky = sphere_points(2048, 10.0, skybox_center)
    scene = skybox_center + rng.normal(size=(500, 3))
    means = np.concatenate((scene, sky), axis=0)
    scales = np.concatenate((
        rng.uniform(0.02, 0.25, size=(len(scene), 3)),
        np.full((len(sky), 3), 0.4),
    ))
    cfg = SkyboxDetectionConfig(enabled=True, radial_band_ratio=0.005)
    result = detect_skybox_gaussians(means, scales, cfg)
    np.testing.assert_allclose(result.skybox_center, skybox_center, atol=0.03)
    assert result.skybox_radius == pytest.approx(10.0, abs=0.03)
    assert np.mean(result.skybox_mask[len(scene):]) > 0.98
    assert np.mean(result.skybox_mask[:len(scene)]) < 0.01
    assert result.diagnostics["radial_band_threshold"] == pytest.approx(
        0.005 * result.skybox_radius
    )


def test_outer_scene_geometry_is_not_removed_without_scale_match():
    rng = np.random.default_rng(8)
    sky = sphere_points(1024, 10.0)
    directions = sphere_points(120, 1.0)
    scene_outer = directions * rng.uniform(8.0, 9.0, size=(len(directions), 1))
    means = np.concatenate((scene_outer, sky))
    scales = np.concatenate((
        rng.uniform(0.03, 0.3, size=(len(scene_outer), 3)),
        np.full((len(sky), 3), 0.5),
    ))
    result = detect_skybox_gaussians(means, scales, SkyboxDetectionConfig(enabled=True))
    assert not np.any(result.skybox_mask[:len(scene_outer)])
    assert np.mean(result.skybox_mask[len(scene_outer):]) > 0.98


def test_depth_probe_calls_geometry_only_renderer():
    class FakeRenderer:
        called = False

        def render_geometry_depth(self, camera, max_image_dim=None):
            self.called = True
            return GaussianRenderResult(
                rgb=None,
                alpha=np.zeros((8, 8), dtype=np.float32),
                depth=np.zeros((8, 8), dtype=np.float32),
                width=8,
                height=8,
            )

    renderer = FakeRenderer()
    probe = GsplatDepthProbe(
        renderer=renderer,
        config=DepthProbeConfig(min_valid_pixels=1, min_valid_ratio=0.0),
    )
    result = probe.probe(_camera_at(0, [0, 0, 0], [0, 0, 1]))
    assert renderer.called
    assert not result.valid


def test_gaussian_visibility_samples_geometry_group_only():
    renderer = SimpleNamespace(
        geometry_means_np=np.array([[0., 0., 1.], [1., 0., 1.]]),
        geometry_opacities_np=np.ones(2),
        geometry_max_scale_np=np.full(2, 0.01),
        skybox_means_np=sphere_points(32, 10.0),
    )
    model = GaussianVisibilityModel(
        renderer, scene_scale=2.0,
        config=VisibilityConfig(max_samples=100),
    )
    assert len(model.sample_points) == 2
    assert np.max(np.linalg.norm(model.sample_points, axis=1)) < 2.0


def test_near_plane_uses_captured_horizontal_radius_and_is_explicit():
    cameras = [
        _camera_at(0, [0., 100., 2.], [0., 0., 1.]),
        _camera_at(1, [0., -100., 4.], [0., 0., 1.]),
    ]
    near = resolve_renderer_near_plane(
        cameras, np.zeros(3), np.array([0., 1., 0.]),
        RendererNearPlaneConfig(ratio=0.01),
    )
    assert near == pytest.approx(0.03)
    assert "near_plane=float(self.config.near_plane)" in inspect.getsource(GsplatRenderer.render)
