#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Point-cloud data structures and IO utilities.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

from viewpoint_framework.geometry_util import (
    bounding_box_center,
    bounding_box_diagonal,
    bounding_box_extent,
)


# ==============================================================================
# Point-cloud representation
# ==============================================================================

@dataclass
class PointCloudData:
    """
    Lightweight NumPy-based point-cloud representation.

    Attributes
    ----------
    points:
        Shape [N,3].

    colors:
        Shape [N,3], floating point RGB in [0,1],
        or None.
    """

    points: np.ndarray
    colors: Optional[np.ndarray] = None

    @property
    def num_points(self) -> int:
        """
        Number of points.
        """

        return len(self.points)

    @property
    def min_bound(self) -> np.ndarray:
        """
        AABB minimum XYZ.
        """

        return self.points.min(
            axis=0
        )

    @property
    def max_bound(self) -> np.ndarray:
        """
        AABB maximum XYZ.
        """

        return self.points.max(
            axis=0
        )

    @property
    def center(self) -> np.ndarray:
        """
        AABB center.
        """

        return bounding_box_center(
            self.min_bound,
            self.max_bound,
        )

    @property
    def extent(self) -> np.ndarray:
        """
        AABB extent.
        """

        return bounding_box_extent(
            self.min_bound,
            self.max_bound,
        )

    @property
    def diagonal(self) -> float:
        """
        AABB diagonal length.

        Degenerate point clouds return 1.0 to avoid
        downstream visualization scale collapse.
        """

        diagonal = bounding_box_diagonal(
            self.min_bound,
            self.max_bound,
        )

        if diagonal <= 1e-8:
            return 1.0

        return diagonal


# ==============================================================================
# Point-cloud processing
# ==============================================================================

def remove_invalid_points(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> PointCloudData:
    """
    Remove points containing NaN / Inf.

    Corresponding colors are removed using the same mask.
    """

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    if points.ndim != 2 or points.shape[1] != 3:

        raise ValueError(
            f"points must have shape [N,3], "
            f"got {points.shape}"
        )

    valid_mask = np.all(
        np.isfinite(points),
        axis=1,
    )

    points = points[
        valid_mask
    ]

    if colors is not None:

        colors = np.asarray(
            colors,
            dtype=np.float64,
        )

        if colors.shape != (
            len(valid_mask),
            3,
        ):

            raise ValueError(
                "colors must have shape [N,3] matching points, "
                f"got {colors.shape}"
            )

        colors = colors[
            valid_mask
        ]

        # Replace abnormal color values instead of dropping geometry.
        colors = np.nan_to_num(
            colors,
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        )

        colors = np.clip(
            colors,
            0.0,
            1.0,
        )

    return PointCloudData(
        points=points,
        colors=colors,
    )


def random_downsample_point_cloud(
    point_cloud: PointCloudData,
    max_points: int,
    random_seed: int = 0,
) -> PointCloudData:
    """
    Randomly downsample a point cloud.

    Primarily intended for visualization.

    Args:
        point_cloud:
            Input point cloud.

        max_points:
            Maximum number of retained points.

            max_points <= 0:
                disable downsampling.

        random_seed:
            Random sampling seed.

    Returns:
        Downsampled PointCloudData.
    """

    if (
        max_points <= 0
        or point_cloud.num_points <= max_points
    ):

        return point_cloud

    rng = np.random.default_rng(
        random_seed
    )

    indices = rng.choice(
        point_cloud.num_points,
        size=max_points,
        replace=False,
    )

    points = point_cloud.points[
        indices
    ]

    colors = None

    if point_cloud.colors is not None:

        colors = point_cloud.colors[
            indices
        ]

    return PointCloudData(
        points=points,
        colors=colors,
    )


# ==============================================================================
# PLY IO
# ==============================================================================

def load_ply_point_cloud(
    ply_path: str,
    max_points: int = 0,
    random_seed: int = 0,
) -> PointCloudData:
    """
    Load PLY point cloud using Open3D.

    Args:
        ply_path:
            Input PLY file.

        max_points:
            Maximum retained points.

            <= 0:
                retain all points.

        random_seed:
            Random downsampling seed.

    Returns:
        PointCloudData.
    """

    ply_path = Path(
        ply_path
    ).expanduser().resolve()

    if not ply_path.exists():

        raise FileNotFoundError(
            f"Point cloud does not exist: {ply_path}"
        )

    if not ply_path.is_file():

        raise ValueError(
            f"Point cloud path is not a file: {ply_path}"
        )

    pcd = o3d.io.read_point_cloud(
        str(ply_path)
    )

    points = np.asarray(
        pcd.points,
        dtype=np.float64,
    )

    if len(points) == 0:

        raise ValueError(
            f"Point cloud contains no points: {ply_path}"
        )

    colors = None

    if pcd.has_colors():

        colors = np.asarray(
            pcd.colors,
            dtype=np.float64,
        )

    # --------------------------------------------------------------------------
    # Remove invalid geometry
    # --------------------------------------------------------------------------

    point_cloud = remove_invalid_points(
        points=points,
        colors=colors,
    )

    if point_cloud.num_points == 0:

        raise ValueError(
            f"Point cloud contains no valid points: {ply_path}"
        )

    # --------------------------------------------------------------------------
    # Visualization / optional downsampling
    # --------------------------------------------------------------------------

    point_cloud = random_downsample_point_cloud(
        point_cloud=point_cloud,
        max_points=max_points,
        random_seed=random_seed,
    )

    return point_cloud