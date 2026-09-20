import numpy as np
import sys
import types
from contextlib import nullcontext
import pytest

from viewpoint_framework.scene_types import CircularInterval
from viewpoint_framework.skybox_detection import SkyboxDetectionConfig, detect_skybox_gaussians
from viewpoint_framework.stage2.height import LocalHeightV33, RobustSide, coverage_consensus
from viewpoint_framework.stage2.radius import classify_inside_signed_radius, resolve_radius_targets
from viewpoint_framework.stage3.selection import SelectionConfig, order_selected_views
from viewpoint_framework.view_space import sample_circular_interval
from viewpoint_framework.height_safety import LOCAL_RELIABLE
from viewpoint_framework.renderer.factory import (
    RendererRasterizationFailure, create_renderer,
)
import viewpoint_framework.renderer.factory as renderer_factory
import viewpoint_framework.renderer.gs_render_backend as gs_render_backend
from viewpoint_framework.renderer.gs_render_backend import compute_plane_depth


def _side(limit):
    return RobustSide(LOCAL_RELIABLE, limit, limit, 1., 1., 1., 1., 0., True)


def _local(index, lo, hi):
    return LocalHeightV33(float(index), 1., 0., [0., 0., 0.],
                          _side(lo), _side(hi), "test")


def test_radius_hole_disables_over_nominal_extension():
    open_space = resolve_radius_targets(9., 6., hole_detected=False)
    hole = resolve_radius_targets(9., 6., hole_detected=True)
    assert open_space.extension == 7.0
    assert hole.extension is None and hole.nominal == 6.0


def test_inside_center_guard_is_not_crossing():
    assert classify_inside_signed_radius(-2., 1.)[0] == "crossing"
    assert classify_inside_signed_radius(.2, 1.) == ("center_guard", 1.)


def test_consensus_prefers_band_containing_captured_median():
    rows = [_local(i, -2., 2.) for i in range(9)] + [_local(9, 10., 12.)]
    result = coverage_consensus(rows, [-1., 0., 1.], support_ratio=.85, min_reliable=5)
    assert result.height_min == -2. and result.height_max == 2.


def test_full_circle_sampling_is_half_open():
    values = sample_circular_interval(
        CircularInterval(-180., 180., 360., False), 20.,
        half_open_full_circle=True)
    assert len(values) == 18
    assert len(np.unique(np.round(values, 8))) == 18


def test_tail_40962_strict_sphere_detection_uses_raw_order():
    count = 40962
    i = np.arange(count, dtype=np.float64)
    y = 1.0 - 2.0*(i+.5)/count
    r = np.sqrt(np.maximum(0., 1.-y*y))
    theta = np.pi*(3.-np.sqrt(5.))*i
    shell = 10.*np.stack((r*np.cos(theta), y, r*np.sin(theta)), axis=1)
    interior = np.array([[0., 0., 0.], [1., 1., 1.]])
    means = np.concatenate((interior, shell), axis=0)
    scales = np.full_like(means, .1)
    cfg = SkyboxDetectionConfig(enabled=True, strategy="tail_strict")
    result = detect_skybox_gaussians(means, scales, cfg)
    assert result.diagnostics["detection_source"] == "tail"
    assert not np.any(result.skybox_mask[:2])
    assert np.all(result.skybox_mask[2:])


def test_elevation_center_out_ordering():
    class Candidate:
        def __init__(self, ident, elevation, azimuth):
            self.candidate_id = ident
            self.elevation_deg = elevation
            self.azimuth_deg = azimuth
    values = [Candidate(0, -20., 0.), Candidate(1, 0., 20.), Candidate(2, 20., 10.)]
    ordered = order_selected_views(
        values, SelectionConfig(ordering_strategy="elevation_center_out"))
    assert [x.candidate_id for x in ordered] == [1, 0, 2]


def test_renderer_factory_explicit_backend_and_failure_boundary(monkeypatch):
    class BrokenRenderer:
        def render_geometry_depth(self, camera, max_image_dim=None):
            raise RuntimeError("rasterizer exploded")
    module = types.ModuleType("gs_render")
    module.__version__ = "test"
    monkeypatch.setitem(sys.modules, "gs_render", module)
    broken = BrokenRenderer()
    broken.device = "cpu"
    monkeypatch.setattr(renderer_factory, "_make_gs_render",
                        lambda *args, **kwargs: broken)
    renderer = create_renderer("unused.ply", backend="gs_render")
    assert renderer.backend_name == "gs_render"
    with pytest.raises(RendererRasterizationFailure, match="Renderer Rasterization Failure"):
        renderer.render_geometry_depth(None)


def test_renderer_factory_supplies_default_config(monkeypatch):
    module = types.ModuleType("gs_render")
    module.__version__ = "test"
    monkeypatch.setitem(sys.modules, "gs_render", module)
    captured = {}
    fake = types.SimpleNamespace(device="cpu")

    def make_renderer(module_arg, gaussian_ply, config, device):
        captured["config"] = config
        return fake

    monkeypatch.setattr(renderer_factory, "_make_gs_render", make_renderer)
    create_renderer("unused.ply", backend="gs_render")
    assert captured["config"].scale_activation == "exp"
    assert captured["config"].opacity_activation == "sigmoid"


def test_auto_locks_to_imported_gs_render_on_initialization_error(monkeypatch):
    module = types.ModuleType("gs_render")
    monkeypatch.setitem(sys.modules, "gs_render", module)
    monkeypatch.setattr(renderer_factory, "_make_gs_render",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("init failed")))
    gsplat_called = []
    monkeypatch.setattr(renderer_factory, "_make_gsplat",
                        lambda *args, **kwargs: gsplat_called.append(True))
    with pytest.raises(RuntimeError, match="init failed"):
        create_renderer("unused.ply", backend="auto")
    assert not gsplat_called


def test_auto_falls_back_only_when_top_level_gs_render_is_missing(monkeypatch):
    original_import = renderer_factory.importlib.import_module
    def missing_top_level(name):
        if name == "gs_render":
            raise ModuleNotFoundError("no gs_render", name="gs_render")
        return original_import(name)
    fallback = types.SimpleNamespace(device="cpu")
    module = types.SimpleNamespace(__version__="test")
    monkeypatch.setattr(renderer_factory.importlib, "import_module", missing_top_level)
    monkeypatch.setattr(renderer_factory, "_make_gsplat",
                        lambda *args, **kwargs: (fallback, module))
    result = create_renderer("unused.ply", backend="auto")
    assert result.backend_name == "gsplat"


def test_auto_does_not_fallback_for_gs_render_dependency_error(monkeypatch):
    def missing_dependency(name):
        raise ModuleNotFoundError("missing private dependency", name="private_dep")
    gsplat_called = []
    monkeypatch.setattr(renderer_factory.importlib, "import_module", missing_dependency)
    monkeypatch.setattr(renderer_factory, "_make_gsplat",
                        lambda *args, **kwargs: gsplat_called.append(True))
    with pytest.raises(ModuleNotFoundError, match="private dependency"):
        create_renderer("unused.ply", backend="auto")
    assert not gsplat_called


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def _mock_gs_render_renderer():
    calls = []

    class Api:
        @staticmethod
        def render(gs, cam, config):
            calls.append(("render", gs, config))
            return (
                _FakeTensor(np.full((3, 2, 3), .25)),
                _FakeTensor(np.full((1, 2, 3), .5)),
                None, None, None, object(),
            )

        @staticmethod
        def render_with_distance(gs, cam, config):
            calls.append(("render_with_distance", gs, config))
            return (
                _FakeTensor(np.full((3, 2, 3), .25)),
                _FakeTensor(np.full((1, 2, 3), .5)),
                None, None,
                _FakeTensor(np.ones((3, 2, 3))),
                _FakeTensor(np.ones((1, 2, 3))),
                object(),
            )

    module = types.SimpleNamespace(
        GsRenderer=Api,
        GsRenderConfigData=lambda **kwargs: types.SimpleNamespace(**kwargs),
    )
    renderer = object.__new__(gs_render_backend.GsRenderRenderer)
    renderer.torch = types.SimpleNamespace(no_grad=nullcontext)
    renderer.module = module
    renderer.config = types.SimpleNamespace(background=(0., 0., 0.))
    renderer.scene_data = types.SimpleNamespace(sh_degree=3)
    renderer._geometry_data = "geometry"
    renderer._full_data = "full"
    renderer._tensor = lambda value: _FakeTensor(value)
    renderer._camera_data = lambda camera, width, height: types.SimpleNamespace(
        width=width, height=height, intrinsic=_FakeTensor([1., 1., 0., 0.]))
    camera = types.SimpleNamespace(width=3, height=2)
    return renderer, camera, calls


def test_gs_render_rgb_uses_render_and_alpha_slot_one():
    renderer, camera, calls = _mock_gs_render_renderer()
    result = renderer.render_geometry_rgb_alpha(camera)
    assert [call[0] for call in calls] == ["render"]
    assert calls[0][2].render_normal is False
    assert calls[0][2].render_depth is False
    assert result.rgb.shape == (2, 3, 3)
    assert np.all(result.alpha == .5)
    assert result.depth is None


def test_gs_render_depth_uses_distance_api_and_raw_outputs(monkeypatch):
    renderer, camera, calls = _mock_gs_render_renderer()
    captured = {}

    def fake_plane_depth(normal, distance, cam_data, torch_module):
        captured.update(normal=normal, distance=distance, camera=cam_data)
        return _FakeTensor(np.full((2, 3), 2.))

    monkeypatch.setattr(gs_render_backend, "compute_plane_depth", fake_plane_depth)
    result = renderer.render_geometry_depth(camera)
    assert [call[0] for call in calls] == ["render_with_distance"]
    assert calls[0][2].render_normal is True
    assert calls[0][2].render_depth is False
    assert captured["normal"] is not None and captured["distance"] is not None
    assert result.rgb is None
    assert np.all(result.alpha == .5)
    assert np.all(result.depth == 2.)


def test_gs_render_combined_rgb_alpha_depth_is_one_distance_pass(monkeypatch):
    renderer, camera, calls = _mock_gs_render_renderer()
    monkeypatch.setattr(
        gs_render_backend, "compute_plane_depth",
        lambda *args: _FakeTensor(np.full((2, 3), 2.)))
    result = renderer.render_geometry(camera, need_rgb=True, need_depth=True)
    assert [call[0] for call in calls] == ["render_with_distance"]
    assert result.rgb.shape == (2, 3, 3)
    assert np.all(result.alpha == .5)
    assert np.all(result.depth == 2.)


def test_compute_plane_depth_returns_camera_z_depth():
    torch = pytest.importorskip("torch")
    normal = torch.zeros((3, 2, 3), dtype=torch.float32)
    normal[2] = -1.0
    distance = torch.full((1, 2, 3), 2.0, dtype=torch.float32)
    cam = types.SimpleNamespace(
        intrinsic=torch.tensor([2.0, 2.0, 1.0, 0.5]), height=2, width=3)
    depth = compute_plane_depth(normal, distance, cam, torch)
    assert depth.shape == (2, 3)
    assert torch.allclose(depth, torch.full((2, 3), 2.0), atol=1e-6)
