from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

from viewpoint_framework.renderer.scene_loader import load_gaussian_scene_data
from viewpoint_framework.renderer.compare_sequence_outputs import (
    _array_metrics,
    _depth_metrics,
)
from viewpoint_framework.renderer.types import GaussianSceneData
from viewpoint_framework.cameras_util import Camera
from viewpoint_framework.skybox_detection import SkyboxDetectionConfig


def _scene_data():
    return GaussianSceneData(
        means=np.array([[0., 0., 1.], [1., 0., 2.]], dtype=np.float32),
        quats=np.array([[2., 0., 0., 0.], [0., 3., 0., 0.]], dtype=np.float32),
        scales=np.array([[-10., -9., -8.], [-3., -2., -1.]], dtype=np.float32),
        opacities=np.array([-2., 2.], dtype=np.float32),
        features_dc=np.zeros((2, 1, 3), dtype=np.float32),
        features_sh=np.empty((2, 0, 3), dtype=np.float32),
        sh_degree=0,
        geometry_mask=np.array([True, False]),
        skybox_mask=np.array([False, True]),
        skybox_center=np.zeros(3, dtype=np.float64),
        skybox_radius=2.,
        metadata={"source_path": "synthetic.ply", "skybox": {}},
    )


def _config():
    return SimpleNamespace(
        scale_activation="exp",
        opacity_activation="sigmoid",
        max_sh_degree=None,
        background=(0., 0., 0.),
        skybox=SkyboxDetectionConfig(enabled=False),
    )


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def to(self, _device):
        return self

    def numpy(self):
        return self.value

    @property
    def dtype(self):
        return self.value.dtype


class _FakeTorch:
    cuda = SimpleNamespace(is_available=lambda: False)
    float32 = np.dtype(np.float32)

    @staticmethod
    def from_numpy(value):
        return _FakeTensor(value)


def test_scene_loader_preserves_raw_ply_parameters(monkeypatch):
    dtype = [(name, "f4") for name in (
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    )]
    vertex = np.zeros(1, dtype=dtype)
    vertex["z"] = 1.
    vertex["opacity"] = -2.
    vertex["scale_0"], vertex["scale_1"], vertex["scale_2"] = -10., -9., -8.
    vertex["rot_0"] = 2.
    class FakePly:
        def __contains__(self, name):
            return name == "vertex"

        def __getitem__(self, name):
            assert name == "vertex"
            return SimpleNamespace(data=vertex)

    plyfile = SimpleNamespace(
        PlyData=SimpleNamespace(read=lambda _path: FakePly()))
    monkeypatch.setitem(sys.modules, "plyfile", plyfile)

    data = load_gaussian_scene_data(Path(__file__), _config())

    np.testing.assert_array_equal(data.scales[0], [-10., -9., -8.])
    np.testing.assert_array_equal(data.quats[0], [2., 0., 0., 0.])
    assert data.opacities[0] == -2.
    assert data.metadata["representation"] == {
        "quats": "raw", "scales": "log", "opacities": "logit"}


def test_backend_comparison_metrics_cover_rgb_and_depth_validity():
    rgb = _array_metrics(np.zeros((2, 2, 3)), np.full((2, 2, 3), .5))
    assert rgb["mae"] == .5 and not rgb["exact"]
    depth = _depth_metrics(
        np.array([[1., 2.], [0., 0.]]),
        np.array([[1., 0.], [3., 0.]]),
    )
    assert depth["valid_iou"] == 1. / 3.
    assert depth["overlap_pixels"] == 1
    assert depth["mean_relative_abs_error"] == 0.


def test_gs_render_receives_raw_tensors_but_geometry_fields_are_activated(monkeypatch):
    from viewpoint_framework.renderer import gs_render_backend

    scene = _scene_data()
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    monkeypatch.setattr(gs_render_backend, "load_gaussian_scene_data",
                        lambda *_args, **_kwargs: scene)

    class GaussianData:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    module = SimpleNamespace(GsRenderGaussianData=GaussianData)
    renderer = gs_render_backend.GsRenderRenderer(
        "unused.ply", module, _config(), device="cpu")

    np.testing.assert_array_equal(renderer._full_data.scales.numpy(), scene.scales)
    np.testing.assert_array_equal(renderer._full_data.rotations.numpy(), scene.quats)
    np.testing.assert_array_equal(
        renderer._full_data.opacitys.numpy()[:, 0], scene.opacities)
    np.testing.assert_allclose(
        renderer.geometry_scales_np, np.exp(scene.scales[:1]), rtol=1e-6)
    np.testing.assert_allclose(
        renderer.geometry_opacities_np, 1. / (1. + np.exp(-scene.opacities[:1])),
        rtol=1e-6)
    assert renderer._full_data.scales.dtype == _FakeTorch.float32


def test_gs_render_camera_data_matches_framework_w2c_and_intrinsics(monkeypatch):
    from viewpoint_framework.renderer import gs_render_backend

    scene = _scene_data()
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    monkeypatch.setattr(gs_render_backend, "load_gaussian_scene_data",
                        lambda *_args, **_kwargs: scene)

    class Data:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    module = SimpleNamespace(
        GsRenderGaussianData=Data,
        GsRenderCameraData=Data,
    )
    renderer = gs_render_backend.GsRenderRenderer(
        "unused.ply", module, _config(), device="cpu")
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 3] = [1., 2., 3.]
    camera = Camera(
        4, 400., 420., 200., 150., 400, 300,
        np.linalg.inv(c2w), c2w)

    data = renderer._camera_data(camera, 200, 150)

    np.testing.assert_allclose(data.w2c_r.numpy(), camera.w2c[:3, :3])
    np.testing.assert_allclose(data.w2c_t.numpy(), camera.w2c[:3, 3])
    np.testing.assert_allclose(data.intrinsic.numpy(), [200., 210., 100., 75.])


def test_gsplat_activates_only_at_backend_boundary(monkeypatch):
    import viewpoint_framework.gs_renderer as gs_renderer

    scene = _scene_data()
    monkeypatch.setattr(gs_renderer, "load_gaussian_scene_data",
                        lambda *_args, **_kwargs: scene)
    renderer = gs_renderer.GsplatRenderer.__new__(gs_renderer.GsplatRenderer)
    renderer.config = _config()
    renderer.torch = _FakeTorch
    renderer.device = "cpu"

    renderer._load_gaussians("unused.ply")

    np.testing.assert_array_equal(renderer.scene_data.scales, scene.scales)
    np.testing.assert_array_equal(renderer.scene_data.quats, scene.quats)
    np.testing.assert_allclose(renderer.scales.numpy(), np.exp(scene.scales[:1]), rtol=1e-6)
    np.testing.assert_allclose(renderer.quats.numpy(), [[1., 0., 0., 0.]], rtol=1e-6)
    np.testing.assert_allclose(
        renderer.opacities.numpy(), 1. / (1. + np.exp(-scene.opacities[:1])),
        rtol=1e-6)
