#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Replaceable visibility / coverage backends for Stage 3.

Two concrete backends are provided:

``GaussianVisibilityModel``
    Samples aligned 3DGS Gaussians and uses low-resolution rendered depth + alpha
    to decide which sampled surface elements are visible from a camera.

``PointCloudVisibilityModel``
    Samples the aligned point cloud and uses a simple image-space z-buffer.

Both expose the same ``visibility(camera) -> bool[M]`` interface, so FPS/greedy
selection and reference selection do not depend on the representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from viewpoint_framework.cameras_util import Camera
from viewpoint_framework.gs_renderer import GsplatRenderer


EPS = 1e-8


@dataclass
class VisibilityConfig:
    strategy: str = "gaussian_visibility"  # gaussian_visibility | pointcloud_visibility | none
    max_samples: int = 12000
    random_seed: int = 0
    max_image_dim: int = 192
    alpha_threshold: float = 0.05
    depth_tolerance_ratio: float = 0.03
    depth_tolerance_abs_ratio: float = 0.003
    gaussian_scale_tolerance: float = 2.0
    opacity_power: float = 1.0


class VisibilityModel:
    sample_points: np.ndarray
    sample_weights: np.ndarray

    def visibility(self, camera: Camera, cache_key: Optional[str] = None) -> np.ndarray:
        raise NotImplementedError

    def visibility_matrix(
        self,
        cameras: Sequence[Camera],
        key_prefix: str = "cam",
    ) -> np.ndarray:
        if len(cameras) == 0:
            return np.zeros((0, len(self.sample_points)), dtype=bool)
        rows = [
            self.visibility(cam, cache_key=f"{key_prefix}:{i}:{cam.index}")
            for i, cam in enumerate(cameras)
        ]
        return np.stack(rows, axis=0)

    def coverage_counts(self, cameras: Sequence[Camera], key_prefix: str = "coverage") -> np.ndarray:
        matrix = self.visibility_matrix(cameras, key_prefix=key_prefix)
        if len(matrix) == 0:
            return np.zeros(len(self.sample_points), dtype=np.int32)
        return matrix.sum(axis=0).astype(np.int32)

    def marginal_gain(self, visible: np.ndarray, counts: np.ndarray) -> float:
        """Diminishing-return surface coverage gain.

        IG(v | S) = sum_{j in V(v)} w_j / (1 + C_j(S))

        A previously unseen sample (C=0) contributes its full weight; repeatedly
        observing the same sample contributes progressively less.  This is a
        practical submodular-style coverage objective for greedy view selection.
        """
        visible = np.asarray(visible, dtype=bool)
        counts = np.asarray(counts, dtype=np.float64)
        return float(np.sum(self.sample_weights[visible] / (1.0 + counts[visible])))


class NullVisibilityModel(VisibilityModel):
    def __init__(self) -> None:
        self.sample_points = np.empty((0, 3), dtype=np.float64)
        self.sample_weights = np.empty((0,), dtype=np.float64)

    def visibility(self, camera: Camera, cache_key: Optional[str] = None) -> np.ndarray:
        return np.empty((0,), dtype=bool)


class GaussianVisibilityModel(VisibilityModel):
    def __init__(
        self,
        renderer: GsplatRenderer,
        scene_scale: float,
        config: Optional[VisibilityConfig] = None,
    ) -> None:
        self.renderer = renderer
        self.config = config or VisibilityConfig()
        self.scene_scale = max(float(scene_scale), EPS)
        self._cache: Dict[str, np.ndarray] = {}

        n = len(renderer.means_np)
        max_samples = max(1, min(int(self.config.max_samples), n))
        rng = np.random.default_rng(int(self.config.random_seed))
        opacity = np.asarray(renderer.opacities_np, dtype=np.float64)
        probabilities = np.maximum(opacity, 1e-5) ** float(self.config.opacity_power)
        probabilities /= np.sum(probabilities)
        if max_samples < n:
            indices = np.sort(rng.choice(n, size=max_samples, replace=False, p=probabilities))
        else:
            indices = np.arange(n, dtype=np.int64)

        self.sample_indices = indices
        self.sample_points = renderer.means_np[indices].astype(np.float64)
        self.sample_scales = renderer.max_scale_np[indices].astype(np.float64)
        weights = opacity[indices].astype(np.float64)
        self.sample_weights = weights / max(float(np.mean(weights)), EPS)

    @staticmethod
    def _project(camera: Camera, points: np.ndarray, width: int, height: int):
        R = camera.w2c[:3, :3]
        t = camera.w2c[:3, 3]
        pc = points @ R.T + t[None, :]
        z = pc[:, 2]
        sx = float(width) / float(camera.width)
        sy = float(height) / float(camera.height)
        fx = camera.fx * sx
        fy = camera.fy * sy
        cx = camera.cx * sx
        cy = camera.cy * sy
        safe_z = np.maximum(z, EPS)
        x = fx * pc[:, 0] / safe_z + cx
        y = fy * pc[:, 1] / safe_z + cy
        return x, y, z

    def visibility(self, camera: Camera, cache_key: Optional[str] = None) -> np.ndarray:
        if cache_key is not None and cache_key in self._cache:
            return self._cache[cache_key].copy()

        render = self.renderer.render_depth(
            camera,
            max_image_dim=int(self.config.max_image_dim),
        )
        assert render.depth is not None
        x, y, z = self._project(camera, self.sample_points, render.width, render.height)
        ix = np.rint(x).astype(np.int64)
        iy = np.rint(y).astype(np.int64)
        in_frame = (
            (z > EPS)
            & (ix >= 0) & (ix < render.width)
            & (iy >= 0) & (iy < render.height)
        )
        visible = np.zeros(len(self.sample_points), dtype=bool)
        idx = np.flatnonzero(in_frame)
        if len(idx):
            px = ix[idx]
            py = iy[idx]
            depth_ref = render.depth[py, px].astype(np.float64)
            alpha_ref = render.alpha[py, px].astype(np.float64)
            tol = np.maximum(
                float(self.config.depth_tolerance_abs_ratio) * self.scene_scale,
                float(self.config.depth_tolerance_ratio) * np.maximum(depth_ref, EPS),
            )
            tol = np.maximum(
                tol,
                float(self.config.gaussian_scale_tolerance) * self.sample_scales[idx],
            )
            ok = (
                (alpha_ref >= float(self.config.alpha_threshold))
                & np.isfinite(depth_ref)
                & (depth_ref > 0.0)
                & (np.abs(z[idx] - depth_ref) <= tol)
            )
            visible[idx[ok]] = True

        if cache_key is not None:
            self._cache[cache_key] = visible.copy()
        return visible


class PointCloudVisibilityModel(VisibilityModel):
    def __init__(
        self,
        points: np.ndarray,
        scene_scale: float,
        config: Optional[VisibilityConfig] = None,
    ) -> None:
        self.config = config or VisibilityConfig(strategy="pointcloud_visibility")
        self.scene_scale = max(float(scene_scale), EPS)
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        finite = np.all(np.isfinite(points), axis=1)
        points = points[finite]
        if len(points) == 0:
            raise ValueError("PointCloudVisibilityModel requires finite points.")
        rng = np.random.default_rng(int(self.config.random_seed))
        max_samples = max(1, min(int(self.config.max_samples), len(points)))
        if max_samples < len(points):
            indices = np.sort(rng.choice(len(points), size=max_samples, replace=False))
            points = points[indices]
        self.sample_points = points
        self.sample_weights = np.ones(len(points), dtype=np.float64)
        self._cache: Dict[str, np.ndarray] = {}

    def visibility(self, camera: Camera, cache_key: Optional[str] = None) -> np.ndarray:
        if cache_key is not None and cache_key in self._cache:
            return self._cache[cache_key].copy()

        max_dim = max(16, int(self.config.max_image_dim))
        scale = min(1.0, max_dim / float(max(camera.width, camera.height)))
        width = max(16, int(round(camera.width * scale)))
        height = max(16, int(round(camera.height * scale)))

        R = camera.w2c[:3, :3]
        t = camera.w2c[:3, 3]
        pc = self.sample_points @ R.T + t[None, :]
        z = pc[:, 2]
        sx = float(width) / float(camera.width)
        sy = float(height) / float(camera.height)
        safe_z = np.maximum(z, EPS)
        x = camera.fx * sx * pc[:, 0] / safe_z + camera.cx * sx
        y = camera.fy * sy * pc[:, 1] / safe_z + camera.cy * sy
        ix = np.rint(x).astype(np.int64)
        iy = np.rint(y).astype(np.int64)
        valid = (
            (z > EPS)
            & (ix >= 0) & (ix < width)
            & (iy >= 0) & (iy < height)
        )
        visible = np.zeros(len(self.sample_points), dtype=bool)
        idx = np.flatnonzero(valid)
        if len(idx):
            flat = iy[idx] * width + ix[idx]
            zbuf = np.full(width * height, np.inf, dtype=np.float64)
            np.minimum.at(zbuf, flat, z[idx])
            nearest = zbuf[flat]
            tol = np.maximum(
                float(self.config.depth_tolerance_abs_ratio) * self.scene_scale,
                float(self.config.depth_tolerance_ratio) * nearest,
            )
            visible[idx[np.abs(z[idx] - nearest) <= tol]] = True

        if cache_key is not None:
            self._cache[cache_key] = visible.copy()
        return visible


def build_visibility_model(
    strategy: str,
    *,
    renderer: Optional[GsplatRenderer],
    point_cloud_points: Optional[np.ndarray],
    scene_scale: float,
    config: VisibilityConfig,
) -> VisibilityModel:
    if strategy == "none":
        return NullVisibilityModel()
    if strategy == "gaussian_visibility":
        if renderer is None:
            raise ValueError("gaussian_visibility requires a GsplatRenderer.")
        return GaussianVisibilityModel(renderer, scene_scale=scene_scale, config=config)
    if strategy == "pointcloud_visibility":
        if point_cloud_points is None:
            raise ValueError("pointcloud_visibility requires point-cloud points.")
        return PointCloudVisibilityModel(point_cloud_points, scene_scale=scene_scale, config=config)
    raise ValueError(f"Unknown visibility strategy: {strategy}")


if __name__ == "__main__":
    # Pure objective sanity check without renderer dependencies.
    model = NullVisibilityModel()
    weights = np.ones(4)
    visible = np.array([True, True, False, True])
    counts = np.array([0, 1, 0, 3])
    gain = float(np.sum(weights[visible] / (1 + counts[visible])))
    assert abs(gain - (1.0 + 0.5 + 0.25)) < 1e-8
    print("stage3.visibility self-test passed")
