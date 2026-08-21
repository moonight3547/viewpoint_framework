#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Low-resolution 3DGS depth probing used by candidate camera placement.

This module is optional at import time.  Only constructing ``GsplatDepthProbe``
requires torch, gsplat and plyfile.  The core pose generator therefore remains
usable in camera/point-cloud-only environments.

Expected PLY format is the common 3D Gaussian Splatting vertex layout:
    x y z
    opacity                 (logit by default)
    scale_0 scale_1 scale_2 (log-scale by default)
    rot_0 rot_1 rot_2 rot_3 (w, x, y, z quaternion)

Only geometry attributes are loaded; SH/color coefficients are intentionally
ignored because depth probing does not need photometric fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class DepthProbeConfig:
    strategy: str = "central_low_quantile"  # central_low_quantile | median | mean
    max_image_dim: int = 256
    central_crop_ratio: float = 0.35
    depth_quantile: float = 0.10
    alpha_threshold: float = 0.05
    min_valid_pixels: int = 32
    min_valid_ratio: float = 0.02
    scale_activation: str = "exp"       # exp | identity
    opacity_activation: str = "sigmoid" # sigmoid | identity


@dataclass
class DepthProbeResult:
    valid: bool
    depth: float
    confidence: float
    valid_pixels: int
    valid_ratio: float
    depth_q10: float
    depth_median: float
    depth_mean: float
    note: str = ""


class NullDepthProbe:
    """No-render fallback implementing the same query interface."""

    def probe(self, *args, **kwargs) -> DepthProbeResult:
        return DepthProbeResult(
            valid=False,
            depth=0.0,
            confidence=0.0,
            valid_pixels=0,
            valid_ratio=0.0,
            depth_q10=0.0,
            depth_median=0.0,
            depth_mean=0.0,
            note="3DGS depth probe disabled/unavailable",
        )


class GsplatDepthProbe:
    """Depth-only renderer backed by gsplat.rasterization."""

    def __init__(
        self,
        gaussian_ply: str,
        config: Optional[DepthProbeConfig] = None,
        device: str = "auto",
    ) -> None:
        self.config = config or DepthProbeConfig()
        if self.config.strategy not in (
            "central_low_quantile",
            "median",
            "mean",
        ):
            raise ValueError(f"Unknown depth probe strategy: {self.config.strategy}")

        try:
            import torch
            from gsplat.rendering import rasterization
            from plyfile import PlyData
        except ImportError as exc:
            raise ImportError(
                "GsplatDepthProbe requires torch, gsplat and plyfile. "
                "Install plyfile and use the same environment that renders your 3DGS."
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

    def _load_gaussians(self, gaussian_ply: str) -> None:
        path = Path(gaussian_ply).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"3DGS PLY does not exist: {path}")

        ply = self.PlyData.read(str(path))
        if "vertex" not in ply:
            raise ValueError(f"PLY has no vertex element: {path}")
        vertex = ply["vertex"].data
        names = set(vertex.dtype.names or ())

        required = {
            "x", "y", "z",
            "opacity",
            "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3",
        }
        missing = sorted(required - names)
        if missing:
            raise ValueError(
                "3DGS PLY is missing geometry properties: " + ", ".join(missing)
            )

        means = np.stack(
            [vertex["x"], vertex["y"], vertex["z"]], axis=1
        ).astype(np.float32)
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
            raise ValueError(
                f"Unknown scale_activation={self.config.scale_activation}"
            )

        if self.config.opacity_activation == "sigmoid":
            opacities = self._sigmoid_numpy(opacities)
        elif self.config.opacity_activation != "identity":
            raise ValueError(
                f"Unknown opacity_activation={self.config.opacity_activation}"
            )

        quat_norm = np.linalg.norm(quats, axis=1, keepdims=True)
        quats = quats / np.maximum(quat_norm, 1e-8)

        finite = (
            np.all(np.isfinite(means), axis=1)
            & np.all(np.isfinite(scales), axis=1)
            & np.all(np.isfinite(quats), axis=1)
            & np.isfinite(opacities)
            & (opacities > 1e-5)
        )
        means = means[finite]
        scales = scales[finite]
        quats = quats[finite]
        opacities = opacities[finite]
        if len(means) == 0:
            raise ValueError(f"No valid Gaussians found in {path}")

        torch = self.torch
        self.means = torch.from_numpy(means).to(self.device)
        self.scales = torch.from_numpy(scales).to(self.device)
        self.quats = torch.from_numpy(quats).to(self.device)
        self.opacities = torch.from_numpy(opacities).to(self.device)
        # Depth-only rasterization still expects a color/features tensor.
        self.colors = torch.zeros(
            (len(means), 3), dtype=torch.float32, device=self.device
        )

    @staticmethod
    def _scaled_intrinsics(camera, width: int, height: int) -> np.ndarray:
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

    def _probe_size(self, camera) -> tuple[int, int]:
        max_dim = max(32, int(self.config.max_image_dim))
        scale = min(1.0, max_dim / float(max(camera.width, camera.height)))
        width = max(16, int(round(camera.width * scale)))
        height = max(16, int(round(camera.height * scale)))
        return width, height

    def probe(self, camera) -> DepthProbeResult:
        """Render one camera and aggregate a conservative depth statistic."""
        torch = self.torch
        width, height = self._probe_size(camera)
        K = self._scaled_intrinsics(camera, width, height)

        c2w = torch.from_numpy(camera.c2w.astype(np.float32)).to(self.device)[None]
        K_t = torch.from_numpy(K).to(self.device)[None]

        with torch.no_grad():
            render_colors, render_alphas, _ = self.rasterization(
                means=self.means,
                quats=self.quats,
                scales=self.scales,
                opacities=self.opacities,
                colors=self.colors,
                viewmats=torch.linalg.inv(c2w),
                Ks=K_t,
                width=width,
                height=height,
                render_mode="RGB+D",
                sh_degree=None,
            )

        # Use accumulated Gaussian depth (D) and divide by alpha ourselves.
        # This keeps the expected-depth semantics explicit and avoids depending
        # on version-specific assumptions about ED normalization.
        ed = render_colors[0, ..., 3]
        alpha = render_alphas[0, ..., 0]

        if self.config.strategy == "central_low_quantile":
            crop = float(np.clip(self.config.central_crop_ratio, 0.05, 1.0))
            crop_w = max(4, int(round(width * crop)))
            crop_h = max(4, int(round(height * crop)))
            x0 = max(0, (width - crop_w) // 2)
            y0 = max(0, (height - crop_h) // 2)
            ed = ed[y0:y0 + crop_h, x0:x0 + crop_w]
            alpha = alpha[y0:y0 + crop_h, x0:x0 + crop_w]

        valid = alpha > float(self.config.alpha_threshold)
        valid_pixels = int(valid.sum().item())
        total_pixels = int(valid.numel())
        valid_ratio = valid_pixels / max(total_pixels, 1)

        if (
            valid_pixels < int(self.config.min_valid_pixels)
            or valid_ratio < float(self.config.min_valid_ratio)
        ):
            return DepthProbeResult(
                valid=False,
                depth=0.0,
                confidence=float(np.clip(valid_ratio, 0.0, 1.0)),
                valid_pixels=valid_pixels,
                valid_ratio=valid_ratio,
                depth_q10=0.0,
                depth_median=0.0,
                depth_mean=0.0,
                note="insufficient alpha-supported depth pixels",
            )

        depths = (ed[valid] / torch.clamp(alpha[valid], min=1e-6)).float()
        depths = depths[torch.isfinite(depths) & (depths > 0)]
        if depths.numel() < int(self.config.min_valid_pixels):
            return DepthProbeResult(
                valid=False,
                depth=0.0,
                confidence=0.0,
                valid_pixels=int(depths.numel()),
                valid_ratio=valid_ratio,
                depth_q10=0.0,
                depth_median=0.0,
                depth_mean=0.0,
                note="insufficient finite positive depth pixels",
            )

        q = float(np.clip(self.config.depth_quantile, 0.0, 1.0))
        q10 = float(torch.quantile(depths, q).item())
        median = float(torch.median(depths).item())
        mean = float(torch.mean(depths).item())

        if self.config.strategy == "mean":
            selected = mean
        elif self.config.strategy == "median":
            selected = median
        else:
            selected = q10

        confidence = float(np.clip(valid_ratio / 0.50, 0.0, 1.0))
        return DepthProbeResult(
            valid=True,
            depth=max(selected, 0.0),
            confidence=confidence,
            valid_pixels=int(depths.numel()),
            valid_ratio=valid_ratio,
            depth_q10=q10,
            depth_median=median,
            depth_mean=mean,
        )
