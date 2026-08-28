#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Top-level orchestration for scene input understanding.

This is intentionally a *thin* pipeline.  Individual algorithms remain in:
    - scene_analysis.py  : center + camera mode
    - view_space.py      : spherical frame + mode-specific bbox
    - radius_field.py    : directional radius strategies

The resulting SceneUnderstandingResult exposes both a serializable SceneProfile
and runtime DirectionalRadiusField objects for later viewpoint generation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from viewpoint_framework.cameras_util import Camera
from viewpoint_framework.radius_field import (
    DirectionalRadiusField,
    RadiusFieldConfig,
)
from viewpoint_framework.scene_analysis import (
    CenterEstimationConfig,
    ModeAnalysisConfig,
    classify_camera_modes,
    estimate_scene_center,
)
from viewpoint_framework.scene_types import (
    CameraMode,
    CameraSceneRelation,
    RadiusFieldSample,
    SceneProfile,
    to_jsonable,
)
from viewpoint_framework.view_space import (
    ViewSpaceConfig,
    azimuth_elevation_to_direction,
    build_spherical_frame,
    estimate_mode_view_bboxes,
    sample_circular_interval,
)


@dataclass
class RadiusSamplingConfig:
    """Grid used only to materialize/debug the continuous radius field."""

    azimuth_step_deg: float = 20.0
    elevation_step_deg: float = 20.0
    include_invalid_samples: bool = True


@dataclass
class CenterRefinementConfig:
    """Optional one-shot center refinement from strict Stage-1 mode support."""

    strategy: str = "none"  # none | dominant_mode_once
    min_support_ratio: float = 0.20


@dataclass
class SceneUnderstandingConfig:
    """Strategy-composable scene-understanding configuration."""

    center: CenterEstimationConfig
    mode: ModeAnalysisConfig
    view_space: ViewSpaceConfig
    radius: RadiusFieldConfig
    radius_sampling: RadiusSamplingConfig
    center_refinement: CenterRefinementConfig = field(
        default_factory=CenterRefinementConfig
    )

    @classmethod
    def default(cls) -> "SceneUnderstandingConfig":
        return cls(
            center=CenterEstimationConfig(),
            mode=ModeAnalysisConfig(),
            view_space=ViewSpaceConfig(),
            radius=RadiusFieldConfig(),
            radius_sampling=RadiusSamplingConfig(),
            center_refinement=CenterRefinementConfig(),
        )

    @classmethod
    def from_dict(cls, data: dict) -> "SceneUnderstandingConfig":
        """Build config from nested JSON/dict for reproducible ablations."""

        return cls(
            center=CenterEstimationConfig(**data.get("center", {})),
            mode=ModeAnalysisConfig(**data.get("mode", {})),
            view_space=ViewSpaceConfig(**data.get("view_space", {})),
            radius=RadiusFieldConfig(**data.get("radius", {})),
            radius_sampling=RadiusSamplingConfig(**data.get("radius_sampling", {})),
            center_refinement=CenterRefinementConfig(
                **data.get("center_refinement", {})
            ),
        )

    @classmethod
    def legacy_camera_only(cls) -> "SceneUnderstandingConfig":
        """Convenience baseline close to the previous pose-only semantics."""

        return cls(
            center=CenterEstimationConfig(
                strategy="legacy_check_alignment",
            ),
            mode=ModeAnalysisConfig(
                strategy="legacy_sign",
                global_strategy="legacy_majority",
            ),
            view_space=ViewSpaceConfig(
                strategy="legacy_minmax",
                extension_mode="fixed",
            ),
            radius=RadiusFieldConfig(
                strategy="global_median",
            ),
            radius_sampling=RadiusSamplingConfig(),
            center_refinement=CenterRefinementConfig(),
        )


@dataclass
class SceneUnderstandingResult:
    """Scene profile plus runtime radius-field query objects."""

    profile: SceneProfile
    radius_fields: Dict[str, DirectionalRadiusField]


def _strict_dominant_mode(relations: Sequence[CameraSceneRelation]) -> CameraMode:
    outside = sum(r.mode == CameraMode.OUTSIDE_IN for r in relations)
    inside = sum(r.mode == CameraMode.INSIDE_OUT for r in relations)
    return CameraMode.INSIDE_OUT if inside > outside else CameraMode.OUTSIDE_IN


def _refresh_relation_geometry(
    cameras: Sequence[Camera],
    relations: Sequence[CameraSceneRelation],
    center: np.ndarray,
) -> List[CameraSceneRelation]:
    """Refresh center-relative geometry without re-running four-way labels."""

    center = np.asarray(center, dtype=np.float64)
    refreshed: List[CameraSceneRelation] = []
    for camera, old in zip(cameras, relations):
        position = np.asarray(camera.position, dtype=np.float64)
        forward = np.asarray(camera.forward, dtype=np.float64)
        forward /= max(float(np.linalg.norm(forward)), 1e-10)
        center_vec = center - position
        radius = float(np.linalg.norm(center_vec))
        if radius <= 1e-10:
            radial = np.zeros(3, dtype=np.float64)
            lam = 0.0
            residual = 0.0
            alignment = 90.0
            residual_ratio = float("inf")
        else:
            radial = (position - center) / radius
            lam = float(np.dot(forward, center_vec))
            residual = float(np.linalg.norm(center_vec - lam * forward))
            residual_ratio = residual / radius
            alignment = float(
                np.degrees(np.arccos(np.clip(abs(lam) / radius, 0.0, 1.0)))
            )
        refreshed.append(
            CameraSceneRelation(
                camera_index=old.camera_index,
                position=position,
                forward=forward,
                radius=radius,
                radial_direction=radial,
                lambda_center=lam,
                sight_residual=residual,
                residual_ratio=residual_ratio,
                alignment_deg=alignment,
                robust_weight=old.robust_weight,
                mode=old.mode,
                confidence=old.confidence,
            )
        )
    return refreshed


def _sample_radius_field(
    mode: CameraMode,
    field: DirectionalRadiusField,
    bbox,
    frame,
    config: RadiusSamplingConfig,
) -> List[RadiusFieldSample]:
    """Materialize radius estimates over a debug grid inside one view bbox."""

    if config.azimuth_step_deg <= 0 or config.elevation_step_deg <= 0:
        raise ValueError("Radius sampling steps must be >0.")

    azimuths = sample_circular_interval(
        bbox.generation_azimuth,
        step_deg=config.azimuth_step_deg,
        include_end=True,
    )

    elevation_min, elevation_max = bbox.generation_elevation_deg

    if elevation_max - elevation_min <= 1e-9:
        elevations = np.asarray([elevation_min], dtype=np.float64)
    else:
        elevations = np.arange(
            elevation_min,
            elevation_max + 0.5 * config.elevation_step_deg,
            config.elevation_step_deg,
            dtype=np.float64,
        )
        if elevations[-1] < elevation_max - 1e-8:
            elevations = np.concatenate([elevations, [elevation_max]])
        elevations[-1] = min(elevations[-1], elevation_max)

    samples: List[RadiusFieldSample] = []

    for elevation_deg in elevations:
        for azimuth_deg in azimuths:
            direction = azimuth_elevation_to_direction(
                azimuth_deg=float(azimuth_deg),
                elevation_deg=float(elevation_deg),
                frame=frame,
            )
            estimate = field.query(direction)

            if not estimate.valid and not config.include_invalid_samples:
                continue

            position = (
                field.center + estimate.nominal * direction
                if estimate.valid
                else None
            )

            samples.append(
                RadiusFieldSample(
                    mode=mode,
                    azimuth_deg=float(azimuth_deg),
                    elevation_deg=float(elevation_deg),
                    direction=direction,
                    position=position,
                    estimate=estimate,
                )
            )

    return samples


def understand_scene(
    cameras: Sequence[Camera],
    config: Optional[SceneUnderstandingConfig] = None,
    point_cloud_points: Optional[np.ndarray] = None,
    legacy_view_limits: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> SceneUnderstandingResult:
    """Run the complete first-stage scene input understanding pipeline."""

    config = config or SceneUnderstandingConfig.default()

    if len(cameras) == 0:
        raise ValueError("At least one captured camera is required.")

    # ------------------------------------------------------------------
    # 1. Common sight center.
    # ------------------------------------------------------------------
    center_fit = estimate_scene_center(
        cameras=cameras,
        config=config.center,
    )

    # ------------------------------------------------------------------
    # 2. Per-camera mode analysis.
    # ------------------------------------------------------------------
    camera_relations, mode_summary = classify_camera_modes(
        cameras=cameras,
        center_fit=center_fit,
        config=config.mode,
    )

    refinement_metadata = {
        "strategy": config.center_refinement.strategy,
        "applied": False,
        "initial_center": center_fit.center.astype(float).tolist(),
    }
    if config.center_refinement.strategy == "dominant_mode_once":
        dominant = _strict_dominant_mode(camera_relations)
        support_indices = [
            i for i, relation in enumerate(camera_relations)
            if relation.mode == dominant
        ]
        support_ratio = len(support_indices) / float(len(cameras))
        refinement_metadata.update(
            {
                "dominant_mode": dominant.value,
                "support_count": len(support_indices),
                "support_ratio": support_ratio,
            }
        )
        if support_ratio >= float(config.center_refinement.min_support_ratio):
            initial_center = center_fit.center.copy()
            center_fit = estimate_scene_center(
                cameras=[cameras[i] for i in support_indices],
                config=config.center,
            )
            camera_relations = _refresh_relation_geometry(
                cameras, camera_relations, center_fit.center
            )
            refinement_metadata.update(
                {
                    "applied": True,
                    "refined_center": center_fit.center.astype(float).tolist(),
                    "center_shift": float(
                        np.linalg.norm(center_fit.center - initial_center)
                    ),
                    "note": (
                        "Four-way Stage-1 labels were preserved; only "
                        "center-relative geometry was refreshed."
                    ),
                }
            )
        else:
            refinement_metadata["note"] = (
                "Skipped: strict dominant-mode support is below min_support_ratio."
            )
    elif config.center_refinement.strategy != "none":
        raise ValueError(
            "Unknown center_refinement strategy: "
            f"{config.center_refinement.strategy}"
        )

    # ------------------------------------------------------------------
    # 3. Spherical scene frame + per-mode bbox.
    # ------------------------------------------------------------------
    frame = build_spherical_frame(config.view_space)
    view_bboxes = estimate_mode_view_bboxes(
        relations=camera_relations,
        frame=frame,
        config=config.view_space,
        legacy_view_limits=legacy_view_limits,
    )

    # ------------------------------------------------------------------
    # 4. Per-mode directional radius field.
    # ------------------------------------------------------------------
    radius_fields: Dict[str, DirectionalRadiusField] = {}
    radius_samples: Dict[str, List[RadiusFieldSample]] = {}

    for mode in (CameraMode.OUTSIDE_IN, CameraMode.INSIDE_OUT):
        key = mode.value
        if (
            key not in view_bboxes
            and config.radius.support_strategy != "all_cameras"
        ):
            continue

        field = DirectionalRadiusField(
            center=center_fit.center,
            mode=mode,
            relations=camera_relations,
            config=config.radius,
            point_cloud_points=point_cloud_points,
            radius_bounds=(
                view_bboxes[key].observed_radius
                if key in view_bboxes
                else None
            ),
        )
        radius_fields[key] = field

        if key in view_bboxes:
            radius_samples[key] = _sample_radius_field(
                mode=mode,
                field=field,
                bbox=view_bboxes[key],
                frame=frame,
                config=config.radius_sampling,
            )

    strategy_config = {
        "center": asdict(config.center),
        "mode": asdict(config.mode),
        "view_space": asdict(config.view_space),
        "radius": asdict(config.radius),
        "radius_sampling": asdict(config.radius_sampling),
        "center_refinement": asdict(config.center_refinement),
    }

    profile = SceneProfile(
        center_fit=center_fit,
        mode_summary=mode_summary,
        coordinate_frame=frame,
        camera_relations=camera_relations,
        view_bboxes=view_bboxes,
        radius_field_samples=radius_samples,
        strategy_config=to_jsonable(strategy_config),
        metadata={
            **dict(metadata or {}),
            "center_refinement": refinement_metadata,
        },
    )

    return SceneUnderstandingResult(
        profile=profile,
        radius_fields=radius_fields,
    )
