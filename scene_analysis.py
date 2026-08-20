#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Scene-center fitting and per-camera acquisition-mode analysis.

The design intentionally exposes legacy and improved strategies through one
stable API so experiments can switch algorithms without changing pipeline
code.

Camera convention is inherited from ``cameras_util.Camera``:
    +X : image right
    +Y : image down
    +Z : camera forward
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from viewpoint_framework.cameras_util import Camera
from viewpoint_framework.scene_types import (
    CameraMode,
    CameraSceneRelation,
    CenterFitResult,
    GlobalCollectionMode,
    ModeSummary,
)


EPS = 1e-10

CENTER_STRATEGIES = (
    "legacy_check_alignment",
    "legacy_augmented_normal_eq",
    "projected_ls",
    "robust_irls",
)

MODE_STRATEGIES = (
    "legacy_sign",
    "robust",
)

GLOBAL_MODE_STRATEGIES = (
    "legacy_majority",
    "weighted",
)


@dataclass
class CenterEstimationConfig:
    """Configuration for scene-center estimation."""

    strategy: str = "robust_irls"

    # Legacy parity: the previous implementation rounded translations to 1e-6
    # and solved normal equations over [center, lambda_0, ..., lambda_N].
    legacy_round_decimals: int = 6
    legacy_fallback_to_lstsq: bool = True

    # Robust IRLS parameters.
    robust_loss: str = "huber"  # huber | tukey
    robust_delta: float = 1.5
    max_iters: int = 10
    convergence_tol: float = 1e-7
    min_robust_scale: float = 1e-6
    inlier_weight_threshold: float = 0.2

    # Numerical degeneracy warning threshold.
    singular_value_eps: float = 1e-10


@dataclass
class ModeAnalysisConfig:
    """Configuration for camera mode classification."""

    strategy: str = "robust"
    global_strategy: str = "weighted"

    # Robust per-camera classification.
    max_alignment_deg: float = 20.0
    max_residual_ratio: float = 0.35
    min_fit_weight: float = 0.2
    min_radius: float = 1e-8

    # Global mode decision.
    dominant_ratio: float = 0.75


def _extract_camera_lines(
    cameras: Sequence[Camera],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return camera centers and normalized forward directions."""

    if len(cameras) == 0:
        raise ValueError("At least one camera is required.")

    positions = np.stack(
        [np.asarray(camera.position, dtype=np.float64) for camera in cameras],
        axis=0,
    )
    directions = np.stack(
        [np.asarray(camera.forward, dtype=np.float64) for camera in cameras],
        axis=0,
    )

    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= EPS):
        bad = np.where(norms <= EPS)[0].tolist()
        raise ValueError(f"Near-zero camera forward vectors at indices: {bad}")

    directions = directions / norms[:, None]
    return positions, directions


def _line_projectors(directions: np.ndarray) -> np.ndarray:
    """Build P_i = I - d_i d_i^T for unit line directions."""

    identity = np.eye(3, dtype=np.float64)[None, :, :]
    return identity - directions[:, :, None] * directions[:, None, :]


def _line_residuals(
    center: np.ndarray,
    positions: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """Perpendicular distance from ``center`` to each sight line."""

    projectors = _line_projectors(directions)
    delta = center[None, :] - positions
    perpendicular = np.einsum("nij,nj->ni", projectors, delta)
    return np.linalg.norm(perpendicular, axis=1)


def _residual_stats(residuals: np.ndarray) -> Tuple[float, float]:
    """Return median and robust MAD scale of residuals."""

    if len(residuals) == 0:
        return 0.0, 0.0

    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    return median, mad


def _legacy_augmented_center(
    positions: np.ndarray,
    directions: np.ndarray,
    config: CenterEstimationConfig,
) -> CenterFitResult:
    """Reproduce the previous center estimator as closely as practical.

    Previous implementation:
        center - lambda_i * direction_i = camera_position_i

    Unknowns are [center_xyz, lambda_0, ..., lambda_N].  It then solves the
    normal equations (A^T A) x = A^T b.  We keep that exact path as an ablation
    baseline and only fall back to lstsq when the normal equations are singular
    and ``legacy_fallback_to_lstsq`` is enabled.
    """

    n = len(positions)
    rounded_positions = np.round(
        positions,
        decimals=config.legacy_round_decimals,
    )

    A = np.zeros((3 * n, 3 + n), dtype=np.float64)
    b = np.zeros(3 * n, dtype=np.float64)

    for i in range(n):
        row = 3 * i
        A[row : row + 3, 0:3] = np.eye(3, dtype=np.float64)
        A[row : row + 3, 3 + i] = -directions[i]
        b[row : row + 3] = rounded_positions[i]

    notes: List[str] = []
    solver = "normal_equations"

    try:
        AtA = A.T @ A
        Atb = A.T @ b
        solution = np.linalg.solve(AtA, Atb)
    except np.linalg.LinAlgError:
        if not config.legacy_fallback_to_lstsq:
            raise
        solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        solver = "lstsq_fallback"
        notes.append(
            "Legacy normal equations were singular; np.linalg.lstsq fallback used."
        )

    center = solution[:3]
    residuals = _line_residuals(center, positions, directions)
    median_residual, mad_residual = _residual_stats(residuals)

    singular_values = np.linalg.svd(A, compute_uv=False)
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if len(singular_values) > 0 and singular_values[-1] > config.singular_value_eps
        else float("inf")
    )

    return CenterFitResult(
        center=center,
        residuals=residuals,
        robust_weights=np.ones(n, dtype=np.float64),
        inlier_mask=np.ones(n, dtype=bool),
        strategy="legacy_augmented_normal_eq",
        solver=solver,
        converged=True,
        iterations=1,
        condition_number=condition_number,
        singular_values=singular_values,
        median_residual=median_residual,
        mad_residual=mad_residual,
        notes=notes,
    )


def _solve_projected_center(
    positions: np.ndarray,
    directions: np.ndarray,
    weights: np.ndarray,
    singular_value_eps: float,
) -> Tuple[np.ndarray, float, np.ndarray, str]:
    """Solve weighted common-line center using a 3x3 projected system."""

    projectors = _line_projectors(directions)
    weighted_projectors = weights[:, None, None] * projectors

    system = np.sum(weighted_projectors, axis=0)
    rhs = np.sum(
        np.einsum("nij,nj->ni", weighted_projectors, positions),
        axis=0,
    )

    center, _, _, singular_values = np.linalg.lstsq(system, rhs, rcond=None)
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if len(singular_values) > 0 and singular_values[-1] > singular_value_eps
        else float("inf")
    )

    return center, condition_number, singular_values, "projected_3x3_lstsq"



def _legacy_check_alignment_center(
    positions: np.ndarray,
    directions: np.ndarray,
    config: CenterEstimationConfig,
) -> CenterFitResult:
    """Reproduce the previous two-pass ``check_camera_alignment`` behavior.

    Historical pipeline:
        target = check_camera_alignment(c2ws, np.zeros(3))
        target = check_camera_alignment(c2ws, target)

    ``check_camera_alignment`` inspected only the first camera.  If its forward
    axis was already exactly parallel/anti-parallel to the target direction
    (atol=1e-5 on dot product), the supplied target was accepted.  Otherwise
    ``compute_sight_center`` was called.  This strategy intentionally preserves
    that behavior for strict ablation; it is *not* recommended as the default.
    """

    def one_pass(target: np.ndarray):
        cam_pos = positions[0]
        forward = directions[0]
        target_vec = target - cam_pos
        norm = float(np.linalg.norm(target_vec))
        if norm <= EPS:
            return None, False
        target_dir = target_vec / norm
        dot = float(np.dot(forward, target_dir))
        aligned = bool(
            np.isclose(dot, 1.0, atol=1e-5)
            or np.isclose(dot, -1.0, atol=1e-5)
        )
        if aligned:
            return np.asarray(target, dtype=np.float64), True
        return None, False

    target = np.zeros(3, dtype=np.float64)
    notes: List[str] = []
    used_sight_center = False

    for pass_index in range(2):
        accepted, aligned = one_pass(target)
        if aligned:
            target = accepted
            notes.append(
                f"Legacy pass {pass_index + 1}: first camera accepted supplied target."
            )
        else:
            fit = _legacy_augmented_center(positions, directions, config)
            target = fit.center
            used_sight_center = True
            notes.append(
                f"Legacy pass {pass_index + 1}: first camera not aligned; compute_sight_center baseline used."
            )

    residuals = _line_residuals(target, positions, directions)
    median_residual, mad_residual = _residual_stats(residuals)

    # Diagnostics use the projected 3x3 geometry even though the returned center
    # follows the historical control flow.
    _, condition_number, singular_values, _ = _solve_projected_center(
        positions,
        directions,
        np.ones(len(positions), dtype=np.float64),
        config.singular_value_eps,
    )

    notes.append(
        "Strict legacy baseline: only the first camera decides whether a supplied target is accepted."
    )

    return CenterFitResult(
        center=target,
        residuals=residuals,
        robust_weights=np.ones(len(positions), dtype=np.float64),
        inlier_mask=np.ones(len(positions), dtype=bool),
        strategy="legacy_check_alignment",
        solver=(
            "legacy_check_alignment+augmented_normal_eq"
            if used_sight_center
            else "legacy_check_alignment_target"
        ),
        converged=True,
        iterations=2,
        condition_number=condition_number,
        singular_values=singular_values,
        median_residual=median_residual,
        mad_residual=mad_residual,
        notes=notes,
    )

def _projected_ls_center(
    positions: np.ndarray,
    directions: np.ndarray,
    config: CenterEstimationConfig,
) -> CenterFitResult:
    """Improved non-robust 3x3 point-to-line least squares center."""

    n = len(positions)
    weights = np.ones(n, dtype=np.float64)
    center, condition_number, singular_values, solver = _solve_projected_center(
        positions,
        directions,
        weights,
        config.singular_value_eps,
    )

    residuals = _line_residuals(center, positions, directions)
    median_residual, mad_residual = _residual_stats(residuals)

    notes: List[str] = []
    if not np.isfinite(condition_number):
        notes.append(
            "Projected center system is degenerate or nearly singular; center may be weakly constrained."
        )

    return CenterFitResult(
        center=center,
        residuals=residuals,
        robust_weights=weights,
        inlier_mask=np.ones(n, dtype=bool),
        strategy="projected_ls",
        solver=solver,
        converged=True,
        iterations=1,
        condition_number=condition_number,
        singular_values=singular_values,
        median_residual=median_residual,
        mad_residual=mad_residual,
        notes=notes,
    )


def _robust_weights(
    residuals: np.ndarray,
    config: CenterEstimationConfig,
) -> Tuple[np.ndarray, float]:
    """Compute Huber/Tukey IRLS weights from residual magnitudes."""

    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    robust_scale = max(1.4826 * mad, config.min_robust_scale)

    z = residuals / robust_scale
    delta = max(config.robust_delta, EPS)

    if config.robust_loss == "huber":
        weights = np.ones_like(z)
        mask = z > delta
        weights[mask] = delta / np.maximum(z[mask], EPS)

    elif config.robust_loss == "tukey":
        u = z / delta
        weights = np.zeros_like(u)
        mask = u < 1.0
        weights[mask] = (1.0 - u[mask] ** 2) ** 2

    else:
        raise ValueError(
            f"Unknown robust_loss={config.robust_loss!r}; expected 'huber' or 'tukey'."
        )

    return weights, robust_scale


def _robust_irls_center(
    positions: np.ndarray,
    directions: np.ndarray,
    config: CenterEstimationConfig,
) -> CenterFitResult:
    """Robust point-to-sight-line center estimation using IRLS."""

    n = len(positions)
    weights = np.ones(n, dtype=np.float64)

    center, condition_number, singular_values, solver = _solve_projected_center(
        positions,
        directions,
        weights,
        config.singular_value_eps,
    )

    converged = False
    iterations = 0
    notes: List[str] = []

    for iteration in range(config.max_iters):
        iterations = iteration + 1
        residuals = _line_residuals(center, positions, directions)
        new_weights, _ = _robust_weights(residuals, config)

        # Avoid a pathological all-zero Tukey state.  Falling back to the previous
        # weights keeps the estimator inspectable rather than silently producing NaN.
        if float(np.sum(new_weights)) <= EPS:
            notes.append(
                "IRLS produced zero total weight; previous iteration retained."
            )
            break

        new_center, condition_number, singular_values, solver = _solve_projected_center(
            positions,
            directions,
            new_weights,
            config.singular_value_eps,
        )

        center_shift = float(np.linalg.norm(new_center - center))
        scene_scale = max(
            float(np.median(np.linalg.norm(positions - center[None, :], axis=1))),
            1.0,
        )

        center = new_center
        weights = new_weights

        if center_shift <= config.convergence_tol * scene_scale:
            converged = True
            break

    residuals = _line_residuals(center, positions, directions)
    # Recompute final weights so diagnostics correspond to the final center.
    weights, _ = _robust_weights(residuals, config)
    inlier_mask = weights >= config.inlier_weight_threshold

    median_residual, mad_residual = _residual_stats(residuals)

    if not np.isfinite(condition_number):
        notes.append(
            "Robust projected center system is degenerate or nearly singular."
        )

    return CenterFitResult(
        center=center,
        residuals=residuals,
        robust_weights=weights,
        inlier_mask=inlier_mask,
        strategy="robust_irls",
        solver=solver,
        converged=converged,
        iterations=iterations,
        condition_number=condition_number,
        singular_values=singular_values,
        median_residual=median_residual,
        mad_residual=mad_residual,
        notes=notes,
    )


def estimate_scene_center(
    cameras: Sequence[Camera],
    config: CenterEstimationConfig | None = None,
) -> CenterFitResult:
    """Estimate a common sight center with a selectable strategy."""

    config = config or CenterEstimationConfig()
    if config.strategy not in CENTER_STRATEGIES:
        raise ValueError(
            f"Unknown center strategy {config.strategy!r}; choices={CENTER_STRATEGIES}"
        )

    positions, directions = _extract_camera_lines(cameras)

    if config.strategy == "legacy_check_alignment":
        return _legacy_check_alignment_center(positions, directions, config)

    if config.strategy == "legacy_augmented_normal_eq":
        return _legacy_augmented_center(positions, directions, config)

    if config.strategy == "projected_ls":
        return _projected_ls_center(positions, directions, config)

    return _robust_irls_center(positions, directions, config)


def _compute_relation_geometry(
    cameras: Sequence[Camera],
    center_fit: CenterFitResult,
) -> List[dict]:
    """Compute strategy-independent camera-to-center geometry."""

    center = np.asarray(center_fit.center, dtype=np.float64)
    relations: List[dict] = []

    for local_index, camera in enumerate(cameras):
        position = np.asarray(camera.position, dtype=np.float64)
        forward = np.asarray(camera.forward, dtype=np.float64)
        forward = forward / max(float(np.linalg.norm(forward)), EPS)

        center_vec = center - position
        radius = float(np.linalg.norm(center_vec))

        if radius <= EPS:
            radial_direction = np.zeros(3, dtype=np.float64)
            lambda_center = 0.0
            alignment_deg = 90.0
            residual_ratio = float("inf")
        else:
            radial_direction = (position - center) / radius
            lambda_center = float(np.dot(forward, center_vec))
            cos_abs = float(np.clip(abs(lambda_center) / radius, 0.0, 1.0))
            alignment_deg = float(np.degrees(np.arccos(cos_abs)))
            residual_ratio = float(center_fit.residuals[local_index] / radius)

        relations.append(
            {
                "local_index": local_index,
                "camera_index": int(camera.index),
                "position": position,
                "forward": forward,
                "radius": radius,
                "radial_direction": radial_direction,
                "lambda_center": lambda_center,
                "sight_residual": float(center_fit.residuals[local_index]),
                "residual_ratio": residual_ratio,
                "alignment_deg": alignment_deg,
                "robust_weight": float(center_fit.robust_weights[local_index]),
                "fit_inlier": bool(center_fit.inlier_mask[local_index]),
            }
        )

    return relations


def classify_camera_modes(
    cameras: Sequence[Camera],
    center_fit: CenterFitResult,
    config: ModeAnalysisConfig | None = None,
) -> Tuple[List[CameraSceneRelation], ModeSummary]:
    """Classify every camera as outside-in / inside-out / ambiguous / outlier.

    ``legacy_sign`` reproduces the previous sign-only behavior: every camera is
    outside-in or inside-out according to the sign of its forward projection to
    the fitted center.  ``robust`` adds fit-weight, residual-ratio, and alignment
    gates while preserving the same sign definition for accepted cameras.
    """

    config = config or ModeAnalysisConfig()

    if config.strategy not in MODE_STRATEGIES:
        raise ValueError(
            f"Unknown mode strategy {config.strategy!r}; choices={MODE_STRATEGIES}"
        )
    if config.global_strategy not in GLOBAL_MODE_STRATEGIES:
        raise ValueError(
            "Unknown global mode strategy "
            f"{config.global_strategy!r}; choices={GLOBAL_MODE_STRATEGIES}"
        )

    geometry = _compute_relation_geometry(cameras, center_fit)
    relations: List[CameraSceneRelation] = []

    for item in geometry:
        lambda_center = item["lambda_center"]

        if config.strategy == "legacy_sign":
            mode = (
                CameraMode.OUTSIDE_IN
                if lambda_center > 0.0
                else CameraMode.INSIDE_OUT
            )
            confidence = 1.0

        else:
            is_outlier = (
                not item["fit_inlier"]
                or item["robust_weight"] < config.min_fit_weight
                or item["radius"] <= config.min_radius
                or item["residual_ratio"] > config.max_residual_ratio
            )

            if is_outlier:
                mode = CameraMode.OUTLIER
                confidence = 0.0

            elif item["alignment_deg"] > config.max_alignment_deg:
                mode = CameraMode.AMBIGUOUS
                alignment_factor = max(
                    0.0,
                    1.0 - item["alignment_deg"] / 90.0,
                )
                confidence = float(item["robust_weight"] * alignment_factor)

            else:
                mode = (
                    CameraMode.OUTSIDE_IN
                    if lambda_center > 0.0
                    else CameraMode.INSIDE_OUT
                )
                alignment_factor = float(
                    np.cos(np.radians(item["alignment_deg"]))
                )
                confidence = float(
                    np.clip(
                        item["robust_weight"] * alignment_factor,
                        0.0,
                        1.0,
                    )
                )

        relations.append(
            CameraSceneRelation(
                camera_index=item["camera_index"],
                position=item["position"],
                forward=item["forward"],
                radius=item["radius"],
                radial_direction=item["radial_direction"],
                lambda_center=item["lambda_center"],
                sight_residual=item["sight_residual"],
                residual_ratio=item["residual_ratio"],
                alignment_deg=item["alignment_deg"],
                robust_weight=item["robust_weight"],
                mode=mode,
                confidence=confidence,
            )
        )

    summary = summarize_collection_mode(relations, config)
    return relations, summary


def summarize_collection_mode(
    relations: Sequence[CameraSceneRelation],
    config: ModeAnalysisConfig | None = None,
) -> ModeSummary:
    """Aggregate per-camera labels into a global collection-mode summary."""

    config = config or ModeAnalysisConfig()

    outside = [r for r in relations if r.mode == CameraMode.OUTSIDE_IN]
    inside = [r for r in relations if r.mode == CameraMode.INSIDE_OUT]
    ambiguous = [r for r in relations if r.mode == CameraMode.AMBIGUOUS]
    outliers = [r for r in relations if r.mode == CameraMode.OUTLIER]

    if config.global_strategy == "legacy_majority":
        outside_weight = float(len(outside))
        inside_weight = float(len(inside))
        denominator = outside_weight + inside_weight

        # Exact previous majority semantics: ties resolve to outside-in because
        # old code used ``inside_out > outside_in``.
        if denominator <= EPS:
            dominant_mode = GlobalCollectionMode.UNKNOWN
            dominant_confidence = 0.0
        elif inside_weight > outside_weight:
            dominant_mode = GlobalCollectionMode.INSIDE_OUT
            dominant_confidence = inside_weight / denominator
        else:
            dominant_mode = GlobalCollectionMode.OUTSIDE_IN
            dominant_confidence = outside_weight / denominator

    else:
        outside_weight = float(sum(max(r.confidence, 0.0) for r in outside))
        inside_weight = float(sum(max(r.confidence, 0.0) for r in inside))
        denominator = outside_weight + inside_weight

        if denominator <= EPS:
            dominant_mode = GlobalCollectionMode.UNKNOWN
            dominant_confidence = 0.0
        else:
            outside_ratio = outside_weight / denominator
            inside_ratio = inside_weight / denominator

            if outside_ratio >= config.dominant_ratio:
                dominant_mode = GlobalCollectionMode.OUTSIDE_IN
                dominant_confidence = outside_ratio
            elif inside_ratio >= config.dominant_ratio:
                dominant_mode = GlobalCollectionMode.INSIDE_OUT
                dominant_confidence = inside_ratio
            else:
                dominant_mode = GlobalCollectionMode.MIXED
                dominant_confidence = max(outside_ratio, inside_ratio)

    denominator = outside_weight + inside_weight
    outside_ratio = outside_weight / denominator if denominator > EPS else 0.0
    inside_ratio = inside_weight / denominator if denominator > EPS else 0.0

    return ModeSummary(
        dominant_mode=dominant_mode,
        dominant_confidence=float(dominant_confidence),
        outside_in_count=len(outside),
        inside_out_count=len(inside),
        ambiguous_count=len(ambiguous),
        outlier_count=len(outliers),
        outside_in_weight=outside_weight,
        inside_out_weight=inside_weight,
        outside_in_ratio=float(outside_ratio),
        inside_out_ratio=float(inside_ratio),
        strategy=config.global_strategy,
    )
