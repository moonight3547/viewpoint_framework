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

from dataclasses import asdict, dataclass
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
class SceneUnderstandingConfig:
    """Strategy-composable scene-understanding configuration."""

    center: CenterEstimationConfig
    mode: ModeAnalysisConfig
    view_space: ViewSpaceConfig
    radius: RadiusFieldConfig
    radius_sampling: RadiusSamplingConfig

    @classmethod
    def default(cls) -> "SceneUnderstandingConfig":
        return cls(
            center=CenterEstimationConfig(),
            mode=ModeAnalysisConfig(),
            view_space=ViewSpaceConfig(),
            radius=RadiusFieldConfig(),
            radius_sampling=RadiusSamplingConfig(),
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
        )


@dataclass
class SceneUnderstandingResult:
    """Scene profile plus runtime radius-field query objects."""

    profile: SceneProfile
    radius_fields: Dict[str, DirectionalRadiusField]


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
        if key not in view_bboxes:
            continue

        field = DirectionalRadiusField(
            center=center_fit.center,
            mode=mode,
            relations=camera_relations,
            config=config.radius,
            point_cloud_points=point_cloud_points,
            radius_bounds=view_bboxes[key].observed_radius,
        )
        radius_fields[key] = field

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
    }

    profile = SceneProfile(
        center_fit=center_fit,
        mode_summary=mode_summary,
        coordinate_frame=frame,
        camera_relations=camera_relations,
        view_bboxes=view_bboxes,
        radius_field_samples=radius_samples,
        strategy_config=to_jsonable(strategy_config),
        metadata=dict(metadata or {}),
    )

    return SceneUnderstandingResult(
        profile=profile,
        radius_fields=radius_fields,
    )
