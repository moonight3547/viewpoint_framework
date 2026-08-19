#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Camera data structures and IO utilities.

Unified camera JSON format
--------------------------

The JSON top-level object is a list.

Each camera is an 18-D list:

[
    fx, fy, cx, cy, width, height,
    w2c_00, w2c_01, w2c_02, w2c_03,
    w2c_10, w2c_11, w2c_12, w2c_13,
    w2c_20, w2c_21, w2c_22, w2c_23
]

Equivalent to:

    camera[:6]
        = [fx, fy, cx, cy, width, height]

    camera[6:18]
        = w2c[:3, :4].flatten()

Camera convention
-----------------

Camera local coordinate:

    +X : image right
    +Y : image down
    +Z : camera forward

No implicit axis flipping or rotation correction is performed.
"""

import json
import warnings

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np


# ==============================================================================
# Camera
# ==============================================================================

@dataclass
class Camera:
    """
    Unified internal camera representation.
    """

    index: int

    fx: float
    fy: float
    cx: float
    cy: float

    width: int
    height: int

    w2c: np.ndarray
    c2w: np.ndarray

    # --------------------------------------------------------------------------
    # Extrinsic properties
    # --------------------------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        """
        Camera center in world coordinates.

        Shape:
            [3]
        """

        return self.c2w[:3, 3]

    @property
    def rotation_c2w(self) -> np.ndarray:
        """
        Camera-to-world rotation matrix.

        Shape:
            [3,3]
        """

        return self.c2w[:3, :3]

    @property
    def rotation_w2c(self) -> np.ndarray:
        """
        World-to-camera rotation matrix.

        Shape:
            [3,3]
        """

        return self.w2c[:3, :3]

    @property
    def translation_w2c(self) -> np.ndarray:
        """
        World-to-camera translation.

        Shape:
            [3]
        """

        return self.w2c[:3, 3]

    # --------------------------------------------------------------------------
    # Camera axes in world coordinates
    # --------------------------------------------------------------------------

    @property
    def right(self) -> np.ndarray:
        """
        Camera local +X direction in world coordinates.
        """

        return self.rotation_c2w[:, 0]

    @property
    def down(self) -> np.ndarray:
        """
        Camera local +Y direction in world coordinates.
        """

        return self.rotation_c2w[:, 1]

    @property
    def forward(self) -> np.ndarray:
        """
        Camera local +Z direction in world coordinates.
        """

        return self.rotation_c2w[:, 2]

    # --------------------------------------------------------------------------
    # Intrinsics
    # --------------------------------------------------------------------------

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """
        Return standard pinhole intrinsic matrix K.

        Shape:
            [3,3]
        """

        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


# ==============================================================================
# Parsing
# ==============================================================================

def parse_camera(
    values: Sequence[float],
    index: int = 0,
    check_rotation: bool = True,
) -> Camera:
    """
    Parse one camera from the unified 18-D representation.

    Args:
        values:
            18-D camera values.

        index:
            Camera index in the original sequence.

        check_rotation:
            Whether to warn if det(R) is not close to +1.

    Returns:
        Camera.
    """

    if len(values) != 18:

        raise ValueError(
            f"Camera {index}: expected 18 values, "
            f"got {len(values)}."
        )

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    fx, fy, cx, cy, width, height = values[:6]

    # --------------------------------------------------------------------------
    # Intrinsic validation
    # --------------------------------------------------------------------------

    if fx <= 0 or fy <= 0:

        raise ValueError(
            f"Camera {index}: invalid focal length: "
            f"fx={fx}, fy={fy}"
        )

    if width <= 0 or height <= 0:

        raise ValueError(
            f"Camera {index}: invalid image size: "
            f"{width} x {height}"
        )

    # --------------------------------------------------------------------------
    # Restore 4x4 w2c
    # --------------------------------------------------------------------------

    w2c = np.eye(
        4,
        dtype=np.float64,
    )

    w2c[:3, :4] = values[6:18].reshape(
        3,
        4,
    )

    # --------------------------------------------------------------------------
    # Validate rotation.
    #
    # Important:
    # Do NOT automatically repair / flip / rotate the input matrix.
    # --------------------------------------------------------------------------

    if check_rotation:

        R = w2c[:3, :3]

        det_R = float(
            np.linalg.det(R)
        )

        if abs(det_R - 1.0) > 1e-2:

            warnings.warn(
                f"Camera {index}: det(R)={det_R:.6f}. "
                "A rigid rotation normally has det(R) ~= +1. "
                "The matrix will NOT be modified."
            )

    # --------------------------------------------------------------------------
    # w2c -> c2w
    # --------------------------------------------------------------------------

    try:

        c2w = np.linalg.inv(
            w2c
        )

    except np.linalg.LinAlgError as exc:

        raise ValueError(
            f"Camera {index}: w2c matrix is singular."
        ) from exc

    return Camera(
        index=index,

        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),

        width=int(round(width)),
        height=int(round(height)),

        w2c=w2c,
        c2w=c2w,
    )


# ==============================================================================
# Camera JSON IO
# ==============================================================================

def load_cameras_json(
    json_path: str,
    check_rotation: bool = True,
) -> List[Camera]:
    """
    Load camera sequence from JSON.

    Args:
        json_path:
            Unified 18-D camera JSON path.

        check_rotation:
            Whether to validate rotation determinants.

    Returns:
        List[Camera].
    """

    json_path = Path(
        json_path
    ).expanduser().resolve()

    if not json_path.exists():

        raise FileNotFoundError(
            f"Camera JSON does not exist: {json_path}"
        )

    if not json_path.is_file():

        raise ValueError(
            f"Camera JSON path is not a file: {json_path}"
        )

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if not isinstance(data, list):

        raise ValueError(
            "Camera JSON top-level object must be a list: "
            f"{json_path}"
        )

    cameras = []

    for index, values in enumerate(data):

        if not isinstance(
            values,
            (list, tuple),
        ):

            raise ValueError(
                f"Camera {index} must be a list."
            )

        cameras.append(
            parse_camera(
                values,
                index=index,
                check_rotation=check_rotation,
            )
        )

    return cameras


# ==============================================================================
# Camera sequence operations
# ==============================================================================

def downsample_cameras(
    cameras: Sequence[Camera],
    stride: int = 4,
) -> List[Camera]:
    """
    Uniformly downsample a camera sequence.

    Example:

        stride = 4

    means:

        cameras[::4]

    Camera.index remains the index in the original input sequence.
    """

    if stride <= 0:

        raise ValueError(
            f"stride must be > 0, got {stride}"
        )

    return list(
        cameras[::stride]
    )


def camera_positions(
    cameras: Sequence[Camera],
) -> np.ndarray:
    """
    Extract camera centers.

    Args:
        cameras:
            Camera sequence.

    Returns:
        Camera centers with shape [N,3].
    """

    if len(cameras) == 0:

        return np.empty(
            (0, 3),
            dtype=np.float64,
        )

    return np.stack(
        [
            camera.position
            for camera in cameras
        ],
        axis=0,
    )