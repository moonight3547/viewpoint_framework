#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure viewport transforms for final camera outputs.

These helpers deliberately operate after viewpoint generation and selection.
They change the pixel coordinate basis without moving the camera center or
changing its world-space optical axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from viewpoint_framework.utils.cameras import Camera


PortraitOutputMode = Literal["off", "auto", "auto_cw90", "auto_ccw90"]
VALID_PORTRAIT_OUTPUT_MODES = ("off", "auto", "auto_cw90", "auto_ccw90")


@dataclass
class OutputTransformConfig:
    """Configuration for final image-plane output transforms."""

    portrait_output: PortraitOutputMode = "off"

    def __post_init__(self) -> None:
        if self.portrait_output not in VALID_PORTRAIT_OUTPUT_MODES:
            raise ValueError(
                "portrait_output must be one of "
                f"{VALID_PORTRAIT_OUTPUT_MODES}, got {self.portrait_output!r}"
            )


def resolve_output_transform(camera: Camera, mode: PortraitOutputMode) -> str:
    """Return the viewport transform actually applied to one output frame."""

    if mode not in VALID_PORTRAIT_OUTPUT_MODES:
        raise ValueError(f"Unknown portrait output mode: {mode!r}")
    if mode == "off" or int(camera.width) <= int(camera.height):
        return "none"
    # The former rotation names remain accepted as compatibility aliases for
    # batch commands written before the portrait contract was clarified.
    return "portrait_viewport"


def transform_camera_viewport(camera: Camera, transform: str) -> Camera:
    """Swap a landscape viewport to portrait without rotating the camera.

    Focal lengths and extrinsics are preserved.  The principal-point offset
    from the image center is preserved while the canvas changes from W x H to
    H x W.  This narrows horizontal coverage and extends vertical coverage
    around the same optical-axis content.
    """

    if transform == "none":
        return Camera(
            index=int(camera.index),
            fx=float(camera.fx),
            fy=float(camera.fy),
            cx=float(camera.cx),
            cy=float(camera.cy),
            width=int(camera.width),
            height=int(camera.height),
            w2c=np.asarray(camera.w2c, dtype=np.float64).copy(),
            c2w=np.asarray(camera.c2w, dtype=np.float64).copy(),
        )
    if transform != "portrait_viewport":
        raise ValueError("transform must be 'none' or 'portrait_viewport'")
    old_center_x = 0.5 * (float(camera.width) - 1.0)
    old_center_y = 0.5 * (float(camera.height) - 1.0)
    new_width = int(camera.height)
    new_height = int(camera.width)
    new_center_x = 0.5 * (float(new_width) - 1.0)
    new_center_y = 0.5 * (float(new_height) - 1.0)
    return Camera(
        index=int(camera.index),
        fx=float(camera.fx),
        fy=float(camera.fy),
        cx=new_center_x + (float(camera.cx) - old_center_x),
        cy=new_center_y + (float(camera.cy) - old_center_y),
        width=new_width,
        height=new_height,
        w2c=np.asarray(camera.w2c, dtype=np.float64).copy(),
        c2w=np.asarray(camera.c2w, dtype=np.float64).copy(),
    )


def transform_camera_for_output(
    camera: Camera,
    mode: PortraitOutputMode,
) -> tuple[Camera, str]:
    """Apply the configured automatic portrait transform to one camera."""

    transform = resolve_output_transform(camera, mode)
    return transform_camera_viewport(camera, transform), transform


__all__ = [
    "OutputTransformConfig",
    "PortraitOutputMode",
    "VALID_PORTRAIT_OUTPUT_MODES",
    "resolve_output_transform",
    "transform_camera_viewport",
    "transform_camera_for_output",
]
