#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure image-plane transforms for final camera/raster outputs.

These helpers deliberately operate after viewpoint generation and selection.
They change the pixel coordinate basis without moving the camera center or
changing its world-space optical axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from viewpoint_framework.utils.cameras import Camera


PortraitOutputMode = Literal["off", "auto_cw90", "auto_ccw90"]
VALID_PORTRAIT_OUTPUT_MODES = ("off", "auto_cw90", "auto_ccw90")


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


def resolve_output_rotation(camera: Camera, mode: PortraitOutputMode) -> str:
    """Return the rotation actually applied to one output frame."""

    if mode not in VALID_PORTRAIT_OUTPUT_MODES:
        raise ValueError(f"Unknown portrait output mode: {mode!r}")
    if mode == "off" or int(camera.width) <= int(camera.height):
        return "none"
    return "cw90" if mode == "auto_cw90" else "ccw90"


def rotate_camera_image_plane(camera: Camera, rotation: str) -> Camera:
    """Return a camera expressed in a rotated image coordinate basis.

    The public camera convention uses OpenCV axes (+X right, +Y down, +Z
    forward) and 0-based pixel centers.  ``rotation`` is the visual rotation
    applied to the raster.
    """

    if rotation == "none":
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
    if rotation == "cw90":
        q = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        cx = float(camera.height - 1) - float(camera.cy)
        cy = float(camera.cx)
    elif rotation == "ccw90":
        q = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
        cx = float(camera.cy)
        cy = float(camera.width - 1) - float(camera.cx)
    else:
        raise ValueError("rotation must be 'none', 'cw90', or 'ccw90'")

    q4 = np.eye(4, dtype=np.float64)
    q4[:3, :3] = q
    w2c = q4 @ np.asarray(camera.w2c, dtype=np.float64)
    c2w = np.asarray(camera.c2w, dtype=np.float64) @ q4.T
    return Camera(
        index=int(camera.index),
        fx=float(camera.fy),
        fy=float(camera.fx),
        cx=cx,
        cy=cy,
        width=int(camera.height),
        height=int(camera.width),
        w2c=w2c,
        c2w=c2w,
    )


def transform_camera_for_output(
    camera: Camera,
    mode: PortraitOutputMode,
) -> tuple[Camera, str]:
    """Apply the configured automatic portrait transform to one camera."""

    rotation = resolve_output_rotation(camera, mode)
    return rotate_camera_image_plane(camera, rotation), rotation


def rotate_raster_for_output(array: np.ndarray | None, rotation: str):
    """Rotate an HxW[...] raster exactly like its output camera."""

    if array is None:
        return None
    value = np.asarray(array)
    if value.ndim < 2:
        raise ValueError("Raster arrays must have at least two dimensions")
    if rotation == "none":
        return value
    if rotation == "cw90":
        return np.ascontiguousarray(np.rot90(value, k=-1, axes=(0, 1)))
    if rotation == "ccw90":
        return np.ascontiguousarray(np.rot90(value, k=1, axes=(0, 1)))
    raise ValueError("rotation must be 'none', 'cw90', or 'ccw90'")


__all__ = [
    "OutputTransformConfig",
    "PortraitOutputMode",
    "VALID_PORTRAIT_OUTPUT_MODES",
    "resolve_output_rotation",
    "rotate_camera_image_plane",
    "rotate_raster_for_output",
    "transform_camera_for_output",
]
