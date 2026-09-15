"""Conservative detection of the fixed spherical Gaussian skybox shell."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict
import warnings

import numpy as np


EPS = 1e-10


@dataclass
class SkyboxDetectionConfig:
    enabled: bool = False
    candidate_inner_ratio: float = 0.80
    # V3.1 contract: classify by distance to skybox_center within 0.5% radius.
    radial_band_ratio: float = 0.005
    irls_iterations: int = 6
    mad_multiplier: float = 4.0
    use_scale_consistency: bool = True
    min_shell_points: int = 32
    max_axis_anisotropy: float = 0.15
    low_confidence_behavior: str = "conservative"  # conservative | disable


@dataclass
class SkyboxDetectionResult:
    skybox_mask: np.ndarray
    skybox_center: np.ndarray
    skybox_radius: float
    confidence: float
    diagnostics: Dict[str, Any]


def _mad(values: np.ndarray, axis=None) -> np.ndarray:
    median = np.median(values, axis=axis, keepdims=True)
    return np.median(np.abs(values - median), axis=axis)


def _fit_sphere(points: np.ndarray, weights: np.ndarray | None = None):
    a = np.concatenate((2.0 * points, np.ones((len(points), 1))), axis=1)
    b = np.sum(points * points, axis=1)
    if weights is not None:
        root = np.sqrt(np.maximum(weights, 0.0))[:, None]
        a = a * root
        b = b * root[:, 0]
    solution, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    if rank < 4:
        raise ValueError("Skybox sphere fit is rank deficient.")
    center = solution[:3]
    radius2 = float(solution[3] + np.dot(center, center))
    if not np.isfinite(radius2) or radius2 <= EPS:
        raise ValueError("Skybox sphere fit produced an invalid radius.")
    return center, float(np.sqrt(radius2))


def _robust_sphere(points: np.ndarray, center0: np.ndarray, radius0: float, iterations: int):
    center, radius = center0.copy(), float(radius0)
    try:
        for _ in range(max(0, int(iterations))):
            residual = np.abs(np.linalg.norm(points - center[None], axis=1) - radius)
            scale = max(float(_mad(residual)), radius * 1e-7, EPS)
            u = residual / (4.685 * scale)
            weights = np.square(np.maximum(0.0, 1.0 - u * u))
            if np.count_nonzero(weights > 0) < 4:
                break
            updated_center, updated_radius = _fit_sphere(points, weights)
            if np.linalg.norm(updated_center - center) <= radius * 1e-9:
                center, radius = updated_center, updated_radius
                break
            center, radius = updated_center, updated_radius
    except (ValueError, np.linalg.LinAlgError):
        center, radius = center0.copy(), float(radius0)
    return center, radius


def detect_skybox_gaussians(
    means: np.ndarray,
    scales: np.ndarray,
    config: SkyboxDetectionConfig | None = None,
) -> SkyboxDetectionResult:
    """Detect only high-confidence members of an external spherical shell.

    The shell center is deliberately named ``skybox_center`` and is estimated
    solely from Gaussian geometry; the reconstruction scene center is not used.
    """
    cfg = config or SkyboxDetectionConfig()
    means = np.asarray(means, dtype=np.float64).reshape(-1, 3)
    scales = np.asarray(scales, dtype=np.float64).reshape(-1, 3)
    n = len(means)
    empty = np.zeros(n, dtype=bool)
    if n == 0:
        raise ValueError("Skybox detection requires at least one Gaussian.")

    xyz_min, xyz_max = np.min(means, axis=0), np.max(means, axis=0)
    skybox_center0 = 0.5 * (xyz_min + xyz_max)
    half_extent = 0.5 * (xyz_max - xyz_min)
    radius0 = float(np.median(half_extent))
    anisotropy = float((np.max(half_extent) - np.min(half_extent)) / max(radius0, EPS))
    base = {
        "enabled": bool(cfg.enabled),
        "strategy": "outer_sphere_robust",
        "total_gaussians": n,
        "axis_half_extent": half_extent.tolist(),
        "axis_anisotropy": anisotropy,
        "candidate_inner_ratio": float(cfg.candidate_inner_ratio),
        "radial_band_ratio": float(cfg.radial_band_ratio),
    }
    if not cfg.enabled:
        base.update({
            "skybox_center": skybox_center0.tolist(), "skybox_radius": radius0,
            "skybox_gaussians": 0, "geometry_gaussians": n,
            "skybox_fraction": 0.0, "detection_confidence": 0.0,
            "status": "disabled", "depth_includes_skybox": True,
            "visibility_includes_skybox": True, "rgb_includes_skybox": True,
        })
        return SkyboxDetectionResult(empty, skybox_center0, radius0, 0.0, base)

    distances0 = np.linalg.norm(means - skybox_center0[None], axis=1)
    seed_mask = distances0 >= float(cfg.candidate_inner_ratio) * radius0
    seeds = means[seed_mask]
    if radius0 <= EPS or len(seeds) < 4:
        skybox_center, skybox_radius = skybox_center0, radius0
    else:
        skybox_center, skybox_radius = _robust_sphere(
            seeds, skybox_center0, radius0, cfg.irls_iterations
        )

    radii = np.linalg.norm(means - skybox_center[None], axis=1)
    residual = np.abs(radii - skybox_radius)
    radial_tolerance = max(float(cfg.radial_band_ratio) * skybox_radius, EPS)
    # Direct V3.1 classification rule requested by the dataset owner.
    radial_mask = residual <= radial_tolerance
    radial_values = residual[radial_mask]
    residual_median = float(np.median(radial_values)) if len(radial_values) else float("inf")
    residual_mad = float(_mad(radial_values)) if len(radial_values) else float("inf")

    scale_mask = np.ones(n, dtype=bool)
    scale_median = np.zeros(3, dtype=np.float64)
    scale_mad = np.zeros(3, dtype=np.float64)
    if np.any(radial_mask):
        log_scales = np.log(np.maximum(scales, EPS))
        shell_scales = log_scales[radial_mask]
        scale_median = np.median(shell_scales, axis=0)
        scale_mad = _mad(shell_scales, axis=0)
        scale_tolerance = np.maximum(float(cfg.mad_multiplier) * scale_mad, 1e-4)
        scale_mask = np.all(np.abs(log_scales - scale_median[None]) <= scale_tolerance[None], axis=1)
    skybox_mask = radial_mask & scale_mask if cfg.use_scale_consistency else radial_mask

    shell_count = int(np.count_nonzero(skybox_mask))
    radial_support = int(np.count_nonzero(radial_mask))
    enough = shell_count >= int(cfg.min_shell_points)
    spherical = anisotropy <= float(cfg.max_axis_anisotropy)
    precision = shell_count / max(radial_support, 1)
    residual_score = max(0.0, 1.0 - residual_median / max(radial_tolerance, EPS))
    confidence = float(np.clip(
        (0.35 * float(enough) + 0.25 * float(spherical) + 0.20 * precision + 0.20 * residual_score),
        0.0, 1.0,
    ))
    low_confidence = not enough or not spherical
    if low_confidence and cfg.low_confidence_behavior == "disable":
        skybox_mask[:] = False
        shell_count = 0
    elif low_confidence and cfg.low_confidence_behavior != "conservative":
        raise ValueError(f"Unknown low_confidence_behavior={cfg.low_confidence_behavior}")

    diagnostics = dict(base)
    diagnostics.update({
        "skybox_center": skybox_center.tolist(),
        "skybox_radius": float(skybox_radius),
        "outer_seed_gaussians": int(len(seeds)),
        "radial_candidate_gaussians": radial_support,
        "skybox_gaussians": shell_count,
        "geometry_gaussians": int(n - shell_count),
        "skybox_fraction": shell_count / max(n, 1),
        "radial_residual_median": residual_median,
        "radial_residual_mad": residual_mad,
        "radial_band_threshold": radial_tolerance,
        "skybox_scale_median": scale_median.tolist(),
        "skybox_scale_mad": scale_mad.tolist(),
        "detection_confidence": confidence,
        "low_confidence": bool(low_confidence),
        "status": "low_confidence" if low_confidence else "detected",
        "depth_includes_skybox": False,
        "visibility_includes_skybox": False,
        "rgb_includes_skybox": True,
        "config": asdict(cfg),
    })
    if low_confidence:
        warnings.warn(
            "[GS:SKYBOX] low-confidence shell detection; using conservative mask "
            f"center={skybox_center.tolist()} radius={skybox_radius:.6g} "
            f"skybox={shell_count}/{n} confidence={confidence:.3f}",
            RuntimeWarning,
        )
    print(
        "[GS:SKYBOX] "
        f"center={skybox_center.tolist()} radius={skybox_radius:.6g} total={n} "
        f"skybox={shell_count} geometry={n-shell_count} confidence={confidence:.3f}"
    )
    return SkyboxDetectionResult(skybox_mask, skybox_center, skybox_radius, confidence, diagnostics)
