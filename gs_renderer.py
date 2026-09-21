#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared gsplat-backed Gaussian renderer for Stage 2 and Stage 3.

The renderer is intentionally small and only depends on the common 3DGS PLY
layout.  Stage 2 uses it for low-resolution reverse-depth probes.  Stage 3 reuses
exactly the same loaded Gaussian tensors for visibility estimation and final RGB
rendering, avoiding duplicate PLY parsing / GPU upload in the end-to-end path.

Camera convention follows :mod:`viewpoint_framework.cameras_util`:
    +X right, +Y down, +Z forward, Camera.c2w maps camera -> world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from viewpoint_framework.skybox_detection import SkyboxDetectionConfig
from viewpoint_framework.renderer.types import GaussianRenderResult
from viewpoint_framework.renderer.scene_loader import (
    activate_opacities,
    activate_scales,
    load_gaussian_scene_data,
    normalize_quaternions,
)


@dataclass
class GaussianRendererConfig:
    scale_activation: str = "exp"          # exp | identity
    opacity_activation: str = "sigmoid"    # sigmoid | identity
    max_sh_degree: Optional[int] = None     # None -> infer from PLY
    background: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    near_plane: float = 0.01
    skybox: SkyboxDetectionConfig = field(default_factory=SkyboxDetectionConfig)


@dataclass
class RendererNearPlaneConfig:
    strategy: str = "captured_radius_ratio"
    ratio: float = 0.01


def resolve_renderer_near_plane(cameras, center, up, config=None) -> float:
    """Resolve z-near from captured horizontal radii, never raw point AABB."""
    cfg = config or RendererNearPlaneConfig()
    if cfg.strategy != "captured_radius_ratio":
        raise ValueError(f"Unknown renderer near-plane strategy: {cfg.strategy}")
    center = np.asarray(center, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    up /= max(float(np.linalg.norm(up)), 1e-10)
    radii = []
    for camera in cameras:
        delta = np.asarray(camera.position, dtype=np.float64) - center
        horizontal = delta - float(np.dot(delta, up)) * up
        rho = float(np.linalg.norm(horizontal))
        if np.isfinite(rho) and rho > 1e-10:
            radii.append(rho)
    if not radii:
        raise ValueError("Cannot resolve renderer near plane without valid captured radii.")
    ratio = float(cfg.ratio)
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("Renderer near-plane ratio must be finite and positive.")
    return max(ratio * float(np.median(radii)), 1e-6)


class GsplatRenderer:
    """Load one aligned Gaussian PLY and render arbitrary framework cameras."""

    def __init__(
        self,
        gaussian_ply: str,
        config: Optional[GaussianRendererConfig] = None,
        device: str = "auto",
    ) -> None:
        self.config = config or GaussianRendererConfig()
        try:
            import torch
            from gsplat.rendering import rasterization
        except ImportError as exc:
            raise ImportError(
                "GsplatRenderer requires torch, gsplat and plyfile. "
                "Use the same environment that already renders your 3DGS."
            ) from exc

        self.torch = torch
        self.rasterization = rasterization
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = str(device)
        self._load_gaussians(gaussian_ply)

    def _load_gaussians(self, gaussian_ply: str) -> None:
        data = load_gaussian_scene_data(gaussian_ply, self.config)
        self.scene_data = data
        means = data.means
        # gsplat consumes activated values, unlike gs_render.  Activation is
        # deliberately confined to this backend boundary.
        scales = activate_scales(data.scales, self.config.scale_activation)
        opacities = activate_opacities(
            data.opacities, self.config.opacity_activation)
        quats = normalize_quaternions(data.quats)
        colors_np = data.features
        geometry_mask = data.geometry_mask
        skybox_mask = data.skybox_mask

        self.source_path = data.metadata["source_path"]
        self.skybox_metadata = data.metadata["skybox"]
        self.skybox_center = data.skybox_center.copy()
        self.skybox_radius = float(data.skybox_radius)
        self.skybox_mask_np = skybox_mask.copy()
        self.sh_degree = int(data.sh_degree)

        # Compatibility names intentionally mean geometry-only in V3.1. This
        # makes legacy Stage-3 sampling safe even before callers migrate to the
        # explicit geometry_* aliases below.
        self.geometry_means_np = means[geometry_mask]
        self.geometry_scales_np = scales[geometry_mask]
        self.geometry_opacities_np = opacities[geometry_mask]
        self.geometry_max_scale_np = np.max(self.geometry_scales_np, axis=1)
        self.means_np = self.geometry_means_np
        self.scales_np = self.geometry_scales_np
        self.opacities_np = self.geometry_opacities_np
        self.max_scale_np = self.geometry_max_scale_np

        self.skybox_means_np = means[skybox_mask]
        self.skybox_scales_np = scales[skybox_mask]
        self.skybox_opacities_np = opacities[skybox_mask]
        torch = self.torch
        # Store disjoint groups: there is no permanent full + geometry duplicate.
        self.means = torch.from_numpy(means[geometry_mask]).to(self.device)
        self.scales = torch.from_numpy(scales[geometry_mask]).to(self.device)
        self.quats = torch.from_numpy(quats[geometry_mask]).to(self.device)
        self.opacities = torch.from_numpy(opacities[geometry_mask]).to(self.device)
        self.colors = torch.from_numpy(colors_np[geometry_mask]).to(self.device)
        self.skybox_means = torch.from_numpy(means[skybox_mask]).to(self.device)
        self.skybox_scales = torch.from_numpy(scales[skybox_mask]).to(self.device)
        self.skybox_quats = torch.from_numpy(quats[skybox_mask]).to(self.device)
        self.skybox_opacities = torch.from_numpy(opacities[skybox_mask]).to(self.device)
        self.skybox_colors = torch.from_numpy(colors_np[skybox_mask]).to(self.device)

    @staticmethod
    def scaled_intrinsics(camera, width: int, height: int) -> np.ndarray:
        sx = float(width) / float(camera.width)
        sy = float(height) / float(camera.height)
        return np.array(
            [
                [camera.fx * sx, 0.0, camera.cx * sx],
                [0.0, camera.fy * sy, camera.cy * sy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def resolve_size(camera, max_image_dim: Optional[int] = None) -> tuple[int, int]:
        if max_image_dim is None or int(max_image_dim) <= 0:
            return int(camera.width), int(camera.height)
        max_dim = max(16, int(max_image_dim))
        scale = min(1.0, max_dim / float(max(camera.width, camera.height)))
        width = max(16, int(round(camera.width * scale)))
        height = max(16, int(round(camera.height * scale)))
        return width, height

    def render(
        self,
        camera,
        *,
        max_image_dim: Optional[int] = None,
        need_rgb: bool = True,
        need_depth: bool = True,
        include_skybox: bool = True,
    ) -> GaussianRenderResult:
        """Render RGB/alpha/expected-depth for one camera."""
        torch = self.torch
        width, height = self.resolve_size(camera, max_image_dim)
        K = self.scaled_intrinsics(camera, width, height)
        c2w = torch.from_numpy(camera.c2w.astype(np.float32)).to(self.device)[None]
        K_t = torch.from_numpy(K).to(self.device)[None]

        # RGB+D is also used for depth-only calls because it is broadly supported
        # across gsplat versions used in existing 3DGS environments.
        render_mode = "RGB+D" if need_depth else "RGB"
        bg = torch.tensor(self.config.background, dtype=torch.float32, device=self.device)

        means, scales, quats = self.means, self.scales, self.quats
        opacities, gaussian_colors = self.opacities, self.colors
        if include_skybox and len(self.skybox_means):
            # Concatenation is transient. Persistent GPU storage remains disjoint,
            # while full RGB keeps gsplat's exact depth-sorted compositing.
            means = torch.cat((means, self.skybox_means), dim=0)
            scales = torch.cat((scales, self.skybox_scales), dim=0)
            quats = torch.cat((quats, self.skybox_quats), dim=0)
            opacities = torch.cat((opacities, self.skybox_opacities), dim=0)
            gaussian_colors = torch.cat((gaussian_colors, self.skybox_colors), dim=0)

        with torch.no_grad():
            colors, alphas, _ = self.rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=gaussian_colors,
                viewmats=torch.linalg.inv(c2w),
                Ks=K_t,
                width=width,
                height=height,
                render_mode=render_mode,
                sh_degree=self.sh_degree,
                backgrounds=None, #bg[None],
                near_plane=float(self.config.near_plane),
            )

        alpha_t = alphas[0, ..., 0]
        alpha = alpha_t.detach().float().cpu().numpy()

        rgb = None
        if need_rgb:
            rgb_t = colors[0, ..., :3]
            rgb = rgb_t.detach().float().cpu().numpy().astype(np.float32)

        depth = None
        if need_depth:
            accum_depth = colors[0, ..., 3]
            expected = accum_depth / torch.clamp(alpha_t, min=1e-6)
            expected = torch.where(alpha_t > 1e-6, expected, torch.zeros_like(expected))
            depth = expected.detach().float().cpu().numpy().astype(np.float32)

        return GaussianRenderResult(
            rgb=rgb,
            alpha=alpha.astype(np.float32),
            depth=depth,
            width=width,
            height=height,
        )

    def render_rgb(self, camera, max_image_dim: Optional[int] = None) -> np.ndarray:
        result = self.render(
            camera,
            max_image_dim=max_image_dim,
            need_rgb=True,
            need_depth=False,
        )
        assert result.rgb is not None
        return result.rgb

    def render_depth(self, camera, max_image_dim: Optional[int] = None) -> GaussianRenderResult:
        return self.render(
            camera,
            max_image_dim=max_image_dim,
            need_rgb=False,
            need_depth=True,
        )

    def render_geometry_depth(self, camera, max_image_dim: Optional[int] = None) -> GaussianRenderResult:
        """Render depth/alpha from scene geometry, explicitly excluding skybox."""
        return self.render(
            camera,
            max_image_dim=max_image_dim,
            need_rgb=False,
            need_depth=True,
            include_skybox=False,
        )

    def render_geometry(self, camera, max_image_dim: Optional[int] = None,
                        need_rgb: bool = True, need_alpha: bool = True,
                        need_depth: bool = False) -> GaussianRenderResult:
        """Unified V3.3 geometry-only RGB/alpha/planar-Z render call."""
        return self.render(camera, max_image_dim=max_image_dim,
                           need_rgb=need_rgb, need_depth=need_depth,
                           include_skybox=False)

    def camera_inside_skybox(self, camera, margin: float = 0.0) -> bool:
        if len(self.skybox_means_np) == 0:
            return True
        distance = float(np.linalg.norm(np.asarray(camera.position) - self.skybox_center))
        return distance < self.skybox_radius - max(float(margin), 0.0)


if __name__ == "__main__":
    print("gs_renderer.py: import OK. Instantiate GsplatRenderer with a standard 3DGS PLY for runtime testing.")
