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
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from viewpoint_framework.skybox_detection import (
    SkyboxDetectionConfig,
    detect_skybox_gaussians,
)
from viewpoint_framework.renderer.types import GaussianRenderResult, GaussianSceneData


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
            from plyfile import PlyData
        except ImportError as exc:
            raise ImportError(
                "GsplatRenderer requires torch, gsplat and plyfile. "
                "Use the same environment that already renders your 3DGS."
            ) from exc

        self.torch = torch
        self.rasterization = rasterization
        self.PlyData = PlyData
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = str(device)
        self._load_gaussians(gaussian_ply)

    @staticmethod
    def _sigmoid_numpy(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _sorted_property_names(names: Sequence[str], prefix: str) -> list[str]:
        items = [name for name in names if name.startswith(prefix)]

        def key(name: str) -> int:
            try:
                return int(name[len(prefix):])
            except ValueError:
                return 10**9

        return sorted(items, key=key)

    def _load_gaussians(self, gaussian_ply: str) -> None:
        path = Path(gaussian_ply).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"3DGS PLY does not exist: {path}")

        ply = self.PlyData.read(str(path))
        if "vertex" not in ply:
            raise ValueError(f"PLY has no vertex element: {path}")
        vertex = ply["vertex"].data
        names = list(vertex.dtype.names or ())
        name_set = set(names)

        required = {
            "x", "y", "z", "opacity",
            "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3",
        }
        missing = sorted(required - name_set)
        if missing:
            raise ValueError(
                "3DGS PLY is missing geometry properties: " + ", ".join(missing)
            )

        means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
        scales = np.stack(
            [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1
        ).astype(np.float32)
        quats = np.stack(
            [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]],
            axis=1,
        ).astype(np.float32)
        opacities = np.asarray(vertex["opacity"], dtype=np.float32)

        if self.config.scale_activation == "exp":
            scales = np.exp(np.clip(scales, -20.0, 20.0))
        elif self.config.scale_activation != "identity":
            raise ValueError(f"Unknown scale_activation={self.config.scale_activation}")

        if self.config.opacity_activation == "sigmoid":
            opacities = self._sigmoid_numpy(opacities)
        elif self.config.opacity_activation != "identity":
            raise ValueError(f"Unknown opacity_activation={self.config.opacity_activation}")

        quat_norm = np.linalg.norm(quats, axis=1, keepdims=True)
        quats = quats / np.maximum(quat_norm, 1e-8)

        # Standard 3DGS SH property layout.  When unavailable we still support
        # depth/alpha rendering and emit a neutral gray RGB fallback.
        dc_names = self._sorted_property_names(names, "f_dc_")
        rest_names = self._sorted_property_names(names, "f_rest_")
        sh_coeffs = None
        sh_degree = None
        if len(dc_names) >= 3:
            dc = np.stack([vertex[name] for name in dc_names[:3]], axis=1).astype(np.float32)
            dc = dc[:, None, :]  # N,1,3
            if rest_names and len(rest_names) % 3 == 0:
                rest_flat = np.stack([vertex[name] for name in rest_names], axis=1).astype(np.float32)
                # Original 3DGS PLY flattens [3, Krest] after transpose.
                k_rest = len(rest_names) // 3
                rest = rest_flat.reshape(len(rest_flat), 3, k_rest).transpose(0, 2, 1)
                sh_coeffs = np.concatenate([dc, rest], axis=1)
            else:
                sh_coeffs = dc

            k = int(sh_coeffs.shape[1])
            inferred = int(round(np.sqrt(k) - 1))
            if (inferred + 1) ** 2 != k:
                inferred = 0
                sh_coeffs = dc
            if self.config.max_sh_degree is not None:
                inferred = min(inferred, int(self.config.max_sh_degree))
                keep = (inferred + 1) ** 2
                sh_coeffs = sh_coeffs[:, :keep]
            sh_degree = inferred

        # Skybox tail detection must see the unfiltered raw PLY order.  The
        # resulting mask is filtered only after classification.
        raw_detection = detect_skybox_gaussians(means, scales, self.config.skybox)
        finite = (
            np.all(np.isfinite(means), axis=1)
            & np.all(np.isfinite(scales), axis=1)
            & np.all(np.isfinite(quats), axis=1)
            & np.isfinite(opacities)
            & (opacities > 1e-6)
        )
        if sh_coeffs is not None:
            finite &= np.all(np.isfinite(sh_coeffs), axis=(1, 2))

        means = means[finite]
        scales = scales[finite]
        quats = quats[finite]
        opacities = opacities[finite]
        if sh_coeffs is not None:
            sh_coeffs = sh_coeffs[finite]
        if len(means) == 0:
            raise ValueError(f"No valid Gaussians found in {path}")

        detection = raw_detection
        skybox_mask = detection.skybox_mask[finite]
        geometry_mask = ~skybox_mask
        if not np.any(geometry_mask):
            raise ValueError("Skybox detector removed every Gaussian; refusing unsafe classification.")

        if sh_coeffs is None:
            colors_np = np.full((len(means), 3), 0.5, dtype=np.float32)
            sh_degree = None
        else:
            colors_np = sh_coeffs

        self.source_path = str(path)
        self.skybox_metadata = detection.diagnostics
        self.skybox_center = detection.skybox_center.astype(np.float64)
        self.skybox_radius = float(detection.skybox_radius)
        self.skybox_mask_np = skybox_mask.copy()
        self.sh_degree = sh_degree

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
        self.scene_data = GaussianSceneData(
            means, quats, scales, opacities,
            (colors_np[:, :1, :] if colors_np.ndim == 3 else colors_np[:, None, :]),
            (colors_np[:, 1:, :] if colors_np.ndim == 3 else
             np.empty((len(colors_np), 0, 3), dtype=np.float32)),
            int(sh_degree or 0),
            geometry_mask.copy(), skybox_mask.copy(),
            self.skybox_center.copy(), self.skybox_radius,
            {"source_path": self.source_path, "skybox": self.skybox_metadata},
        )

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
