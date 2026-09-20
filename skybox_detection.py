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
    strategy: str = "legacy"  # legacy | tail_strict
    expected_tail_count: int = 40962
    strict_scale_axis_ratio: float = 0.02
    strict_scale_cv: float = 0.02
    strict_radial_coverage: float = 0.995
    strict_angular_coverage: float = 0.95


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

    if cfg.strategy == "tail_strict":
        count = int(cfg.expected_tail_count)
        candidate = np.zeros(n, dtype=bool)
        if n >= count:
            candidate[n-count:] = True
        center = skybox_center0

        def evaluate(mask):
            points, candidate_scales = means[mask], scales[mask]
            radii = (np.linalg.norm(points-center[None], axis=1)
                     if len(points) else np.empty(0))
            radius = float(np.median(radii)) if len(radii) else radius0
            radial_coverage = (float(np.mean(np.abs(radii-radius) <=
                               cfg.radial_band_ratio*max(radius, EPS)))
                               if len(radii) else 0.0)
            axis_ratio = (np.max(candidate_scales, axis=1) / np.maximum(
                np.min(candidate_scales, axis=1), EPS)) if len(candidate_scales) else np.empty(0)
            scale_axis_ok = bool(len(axis_ratio) and
                                 np.quantile(axis_ratio-1.0, .99) <= cfg.strict_scale_axis_ratio)
            scalar_scale = (np.mean(candidate_scales, axis=1)
                            if len(candidate_scales) else np.empty(0))
            scale_cv = (float(np.std(scalar_scale)/max(np.mean(scalar_scale), EPS))
                        if len(scalar_scale) else float("inf"))
            angular_coverage = 0.0
            if len(points) and radius > EPS:
                unit = (points-center[None]) / np.maximum(radii[:, None], EPS)
                az = (np.arctan2(unit[:, 0], unit[:, 2]) + np.pi) / (2*np.pi)
                el = (np.arcsin(np.clip(unit[:, 1], -1., 1.)) + np.pi/2) / np.pi
                ai = np.minimum((az*36).astype(int), 35)
                ei = np.minimum((el*18).astype(int), 17)
                angular_coverage = len(set(zip(ai.tolist(), ei.tolist()))) / (36*18)
            ok = bool(len(points) >= cfg.min_shell_points
                      and radial_coverage >= cfg.strict_radial_coverage
                      and scale_axis_ok and scale_cv <= cfg.strict_scale_cv
                      and angular_coverage >= cfg.strict_angular_coverage)
            return ok, radius, radial_coverage, scale_axis_ok, scale_cv, angular_coverage

        accepted, radius, radial_coverage, scale_axis_ok, scale_cv, angular_coverage = evaluate(candidate)
        source = "tail"
        if not accepted:
            # Raw-order prior failed: permit a skybox only if all strict
            # geometric signatures agree on the outer AABB-centered shell.
            all_radii = np.linalg.norm(means-center[None], axis=1)
            outer_radius = float(np.max(all_radii))
            candidate = np.abs(all_radii-outer_radius) <= cfg.radial_band_ratio*max(outer_radius, EPS)
            accepted, radius, radial_coverage, scale_axis_ok, scale_cv, angular_coverage = evaluate(candidate)
            source = "strict_geometry"
        if accepted:
            all_radii = np.linalg.norm(means-center[None], axis=1)
            radial_member = np.abs(all_radii-radius) <= cfg.radial_band_ratio*max(radius, EPS)
            per_axis_ratio = np.max(scales, axis=1) / np.maximum(np.min(scales, axis=1), EPS)
            scale_member = per_axis_ratio-1.0 <= cfg.strict_scale_axis_ratio
            mask = candidate & radial_member & scale_member
        else:
            mask = empty.copy()
        status = (f"{source}_detected" if accepted else
                  "tail_and_strict_geometry_rejected")
        diagnostics = dict(base)
        diagnostics.update({
            "strategy": "raw_ply_tail_40962_strict",
            "status": status,
            "tail_count_expected": count,
            "tail_count_available": min(n, count),
            "skybox_center": center.tolist(),
            "skybox_radius": radius,
            "detection_source": source,
            "radial_coverage": radial_coverage,
            "scale_axis_consistent": scale_axis_ok,
            "scale_population_cv": scale_cv,
            "angular_coverage": angular_coverage,
            "skybox_gaussians": int(np.count_nonzero(mask)),
            "geometry_gaussians": int(n-np.count_nonzero(mask)),
            "skybox_fraction": float(np.mean(mask)),
            "detection_confidence": 1.0 if accepted else 0.0,
            "depth_includes_skybox": not accepted,
            "visibility_includes_skybox": not accepted,
            "rgb_includes_skybox": True,
            "config": asdict(cfg),
        })
        print("[GS:SKYBOX] "
              f"strategy=raw_tail count={count} status={status} "
              f"radial={diagnostics['radial_coverage']:.4f} "
              f"angular={angular_coverage:.4f} scale_cv={scale_cv:.6g}")
        return SkyboxDetectionResult(mask, center, radius,
                                     1.0 if accepted else 0.0, diagnostics)
    if cfg.strategy != "legacy":
        raise ValueError(f"Unknown skybox strategy={cfg.strategy}")

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
