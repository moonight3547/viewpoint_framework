#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Low-resolution 3DGS depth probing used by Stage-2 camera placement.

The actual Gaussian PLY loading / rasterization is shared with Stage 3 through
:class:`viewpoint_framework.gs_renderer.GsplatRenderer`.  The old constructor
``GsplatDepthProbe(gaussian_ply=...)`` remains valid, while the end-to-end runner
passes one already-loaded renderer so Stage 2 and Stage 3 share GPU tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from viewpoint_framework.gs_renderer import GsplatRenderer, GaussianRendererConfig


@dataclass
class DepthProbeConfig:
    strategy: str = "central_low_quantile"  # central_low_quantile | median | mean
    max_image_dim: int = 256
    central_crop_ratio: float = 0.35
    depth_quantile: float = 0.10
    alpha_threshold: float = 0.05
    min_valid_pixels: int = 32
    min_valid_ratio: float = 0.02
    scale_activation: str = "exp"
    opacity_activation: str = "sigmoid"


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
    hole_ratio: float = 1.0
    center_valid: bool = False
    hole_detected: bool = False
    note: str = ""


class NullDepthProbe:
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
    def __init__(
        self,
        gaussian_ply: Optional[str] = None,
        config: Optional[DepthProbeConfig] = None,
        device: str = "auto",
        renderer: Optional[GsplatRenderer] = None,
    ) -> None:
        self.config = config or DepthProbeConfig()
        if self.config.strategy not in ("central_low_quantile", "median", "mean"):
            raise ValueError(f"Unknown depth probe strategy: {self.config.strategy}")

        if renderer is None:
            if not gaussian_ply:
                raise ValueError("Either gaussian_ply or renderer must be provided.")
            renderer = GsplatRenderer(
                gaussian_ply,
                config=GaussianRendererConfig(
                    scale_activation=self.config.scale_activation,
                    opacity_activation=self.config.opacity_activation,
                ),
                device=device,
            )
        self.renderer = renderer

    def probe(self, camera) -> DepthProbeResult:
        result = self.renderer.render_geometry_depth(
            camera,
            max_image_dim=self.config.max_image_dim,
        )
        assert result.depth is not None
        depth = result.depth
        alpha = result.alpha

        if self.config.strategy == "central_low_quantile":
            crop = float(np.clip(self.config.central_crop_ratio, 0.05, 1.0))
            crop_w = max(4, int(round(result.width * crop)))
            crop_h = max(4, int(round(result.height * crop)))
            x0 = max(0, (result.width - crop_w) // 2)
            y0 = max(0, (result.height - crop_h) // 2)
            depth = depth[y0:y0 + crop_h, x0:x0 + crop_w]
            alpha = alpha[y0:y0 + crop_h, x0:x0 + crop_w]

        valid = (
            (alpha > float(self.config.alpha_threshold))
            & np.isfinite(depth)
            & (depth > 0.0)
        )
        valid_pixels = int(np.count_nonzero(valid))
        total_pixels = int(valid.size)
        valid_ratio = valid_pixels / max(total_pixels, 1)
        cy, cx = (valid.shape[0] - 1) // 2, (valid.shape[1] - 1) // 2
        center_valid = bool(np.any(valid[
            max(0, cy - 1):min(valid.shape[0], cy + 2),
            max(0, cx - 1):min(valid.shape[1], cx + 2),
        ]))
        hole_ratio = 1.0 - valid_ratio
        hole_detected = bool(hole_ratio > 0.5 and not center_valid)

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
                hole_ratio=hole_ratio,
                center_valid=center_valid,
                hole_detected=hole_detected,
                note="insufficient alpha-supported depth pixels",
            )

        depths = depth[valid].astype(np.float64)
        q = float(np.clip(self.config.depth_quantile, 0.0, 1.0))
        q_depth = float(np.quantile(depths, q))
        median = float(np.median(depths))
        mean = float(np.mean(depths))

        if self.config.strategy == "mean":
            selected = mean
        elif self.config.strategy == "median":
            selected = median
        else:
            selected = q_depth

        confidence = float(np.clip(valid_ratio / 0.50, 0.0, 1.0))
        return DepthProbeResult(
            valid=True,
            depth=max(selected, 0.0),
            confidence=confidence,
            valid_pixels=valid_pixels,
            valid_ratio=valid_ratio,
            depth_q10=q_depth,
            depth_median=median,
            depth_mean=mean,
            hole_ratio=hole_ratio,
            center_valid=center_valid,
            hole_detected=hole_detected,
        )


if __name__ == "__main__":
    print("gs_depth_probe.py: import OK; runtime test requires a Gaussian PLY.")
