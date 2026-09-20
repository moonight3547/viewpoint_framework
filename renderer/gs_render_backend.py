"""Adapter for the private ``gs_render`` API used by V3.3."""
from __future__ import annotations
import numpy as np

from viewpoint_framework.renderer.scene_loader import load_gaussian_scene_data
from viewpoint_framework.renderer.types import GaussianRenderResult

# ``render``: color, alpha, semantic, depth, normal, extra_data.
# ``render_with_distance`` inserts distance before extra_data.
RGB_INDEX, ALPHA_INDEX = 0, 1
DEPTH_INDEX, NORMAL_INDEX, DISTANCE_INDEX = 3, 4, 5
RENDER_OUTPUT_COUNT = 6
DISTANCE_OUTPUT_COUNT = 7


def compute_plane_depth(rendered_normal, rendered_distance, cam_data, torch_module=None):
    """Convert gs_render plane distance to camera-space planar/Z depth."""
    torch = torch_module
    if torch is None:
        import torch
    fx, fy, cx, cy = cam_data.intrinsic.detach()
    device, dtype = rendered_normal.device, rendered_normal.dtype
    y, x = torch.meshgrid(
        torch.arange(cam_data.height, device=device, dtype=dtype),
        torch.arange(cam_data.width, device=device, dtype=dtype), indexing="ij")
    ray_x, ray_y = (x-cx)/fx, (y-cy)/fy
    denom = -(rendered_normal[0]*ray_x + rendered_normal[1]*ray_y
              + rendered_normal[2] + 1e-8)
    return rendered_distance[0]/denom


def _hwc_rgb(tensor, height, width):
    if tensor is None:
        raise ValueError("gs_render render returned no RGB tensor")
    value = np.squeeze(tensor.detach().float().cpu().numpy())
    if value.shape == (3, height, width):
        value = np.moveaxis(value, 0, -1)
    if value.shape != (height, width, 3):
        raise ValueError(f"gs_render RGB shape must be HxWx3 or 3xHxW, got {value.shape}")
    return value.astype(np.float32, copy=False)


def _hw(tensor, height, width, name):
    if tensor is None:
        raise ValueError(f"gs_render render returned no {name} tensor")
    value = np.squeeze(tensor.detach().float().cpu().numpy())
    if value.shape != (height, width):
        raise ValueError(f"gs_render {name} shape must be HxW, got {value.shape}")
    return value.astype(np.float32, copy=False)


class GsRenderRenderer:
    backend_name = "gs_render"

    def __init__(self, gaussian_ply, module, config, device="auto"):
        import torch
        self.torch, self.module, self.config = torch, module, config
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = str(device)
        self.scene_data = load_gaussian_scene_data(gaussian_ply, config)
        data = self.scene_data
        self.geometry_means_np = data.geometry_means
        self.geometry_scales_np = data.scales[data.geometry_mask]
        self.geometry_opacities_np = data.opacities[data.geometry_mask]
        self.geometry_max_scale_np = np.max(self.geometry_scales_np, axis=1)
        self.means_np, self.scales_np = self.geometry_means_np, self.geometry_scales_np
        self.opacities_np, self.max_scale_np = self.geometry_opacities_np, self.geometry_max_scale_np
        self.skybox_means_np = data.skybox_means
        self.skybox_scales_np = data.scales[data.skybox_mask]
        self.skybox_opacities_np = data.opacities[data.skybox_mask]
        self.skybox_center, self.skybox_radius = data.skybox_center.copy(), float(data.skybox_radius)
        self.skybox_metadata = data.metadata["skybox"]
        self.source_path = data.metadata["source_path"]
        self._geometry_data = self._gaussian_data(data.geometry_mask)
        self._full_data = self._gaussian_data(np.ones(len(data.means), dtype=bool))

    def _tensor(self, value):
        return self.torch.from_numpy(np.ascontiguousarray(value)).to(self.device)

    def _gaussian_data(self, mask):
        data = self.scene_data
        return self.module.GsRenderGaussianData(
            means=self._tensor(data.means[mask]), rotations=self._tensor(data.quats[mask]),
            scales=self._tensor(data.scales[mask]), opacitys=self._tensor(data.opacities[mask, None]),
            features_dc=self._tensor(data.features_dc[mask]),
            features_sh=self._tensor(data.features_sh[mask]), semantics=None)

    @staticmethod
    def resolve_size(camera, max_image_dim=None):
        if max_image_dim is None or int(max_image_dim) <= 0:
            return int(camera.width), int(camera.height)
        scale = min(1., max(16, int(max_image_dim))/float(max(camera.width, camera.height)))
        return max(16, round(camera.width*scale)), max(16, round(camera.height*scale))

    @staticmethod
    def scaled_intrinsics(camera, width, height):
        sx, sy = width/float(camera.width), height/float(camera.height)
        return np.array([[camera.fx*sx, 0., camera.cx*sx],
                         [0., camera.fy*sy, camera.cy*sy], [0., 0., 1.]], dtype=np.float32)

    def _camera_data(self, camera, width, height):
        k = self.scaled_intrinsics(camera, width, height)
        w2c = np.asarray(camera.w2c, dtype=np.float32)
        return self.module.GsRenderCameraData(
            width=int(width), height=int(height), w2c_r=self._tensor(w2c[:3, :3]),
            w2c_t=self._tensor(w2c[:3, 3]),
            intrinsic=self._tensor(np.array(
                [k[0, 0], k[1, 1], k[0, 2], k[1, 2]], dtype=np.float32)),
            exposure=None)

    def _config_data(self, *, render_normal):
        return self.module.GsRenderConfigData(
            degree=int(self.scene_data.sh_degree),
            bg_color=self._tensor(np.asarray(self.config.background, dtype=np.float32)),
            render_depth=False, render_normal=bool(render_normal),
            clamp_color_min=False, return_abs_grad=False, use_bucket=True)

    def _render(self, camera, *, max_image_dim=None, include_skybox=False,
                need_rgb=True, need_depth=False):
        width, height = self.resolve_size(camera, max_image_dim)
        cam_data = self._camera_data(camera, width, height)
        gs_data = self._full_data if include_skybox else self._geometry_data
        config_data = self._config_data(render_normal=need_depth)
        with self.torch.no_grad():
            if need_depth:
                outputs = list(self.module.GsRenderer.render_with_distance(
                    gs_data, cam_data, config_data))
                if len(outputs) != DISTANCE_OUTPUT_COUNT:
                    raise ValueError(
                        "gs_render render_with_distance must return 7 values, "
                        f"got {len(outputs)}")
                normal, distance = outputs[NORMAL_INDEX], outputs[DISTANCE_INDEX]
                if normal is None or distance is None:
                    raise ValueError(
                        "gs_render render_with_distance returned no normal/distance "
                        "with render_normal=True")
                outputs[DEPTH_INDEX] = compute_plane_depth(
                    normal, distance, cam_data, self.torch)
            else:
                outputs = list(self.module.GsRenderer.render(
                    gs_data, cam_data, config_data))
                if len(outputs) != RENDER_OUTPUT_COUNT:
                    raise ValueError(
                        "gs_render render must return 6 values, "
                        f"got {len(outputs)}")
        rgb = _hwc_rgb(outputs[RGB_INDEX], height, width) if need_rgb else None
        alpha = _hw(outputs[ALPHA_INDEX], height, width, "alpha")
        depth = None
        if need_depth:
            depth = _hw(outputs[DEPTH_INDEX], height, width, "depth")
            valid = np.isfinite(depth) & (depth > 0.) & np.isfinite(alpha) & (alpha > 1e-6)
            depth = np.where(valid, depth, 0.).astype(np.float32)
        alpha = np.where(np.isfinite(alpha), alpha, 0.).astype(np.float32)
        return GaussianRenderResult(rgb, alpha, depth, width, height)

    def render(self, camera, *, max_image_dim=None, need_rgb=True,
               need_depth=True, include_skybox=True):
        return self._render(camera, max_image_dim=max_image_dim,
                            include_skybox=include_skybox,
                            need_rgb=need_rgb, need_depth=need_depth)

    def render_geometry(self, camera, max_image_dim=None, need_rgb=True,
                        need_alpha=True, need_depth=False):
        return self._render(camera, max_image_dim=max_image_dim,
                            include_skybox=False, need_rgb=need_rgb,
                            need_depth=need_depth)

    def render_geometry_depth(self, camera, max_image_dim=None):
        return self._render(camera, max_image_dim=max_image_dim,
                            include_skybox=False, need_rgb=False, need_depth=True)

    def render_geometry_rgb_alpha(self, camera, max_image_dim=None):
        return self._render(camera, max_image_dim=max_image_dim,
                            include_skybox=False, need_rgb=True, need_depth=False)

    def render_full_rgb_alpha(self, camera, max_image_dim=None):
        return self._render(camera, max_image_dim=max_image_dim,
                            include_skybox=True, need_rgb=True, need_depth=False)

    def render_rgb(self, camera, max_image_dim=None):
        return self.render_full_rgb_alpha(camera, max_image_dim).rgb

    def render_depth(self, camera, max_image_dim=None):
        return self._render(camera, max_image_dim=max_image_dim,
                            include_skybox=True, need_rgb=False, need_depth=True)

    def camera_inside_skybox(self, camera, margin=0.):
        if not self.scene_data.skybox_present:
            return True
        distance = float(np.linalg.norm(np.asarray(camera.position)-self.skybox_center))
        return distance < self.skybox_radius-max(float(margin), 0.)


__all__ = ["GsRenderRenderer", "compute_plane_depth"]
