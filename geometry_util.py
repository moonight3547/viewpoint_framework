#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic geometry utilities.

Design principle
----------------
This module is the lowest-level geometry layer of the viewpoint framework.

It should remain independent from:
    - Camera
    - PointCloudData
    - Open3D
    - Plotly

Only generic NumPy-based geometry operations should be placed here.
"""

import numpy as np


EPS = 1e-8


# ==============================================================================
# Vector operations
# ==============================================================================

def normalize_vector(
    vector: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:
    """
    Normalize one vector.

    Args:
        vector:
            Shape [D].

        eps:
            Minimum allowed vector norm.

    Returns:
        Normalized vector.

    Raises:
        ValueError:
            If vector norm is too small.
    """

    vector = np.asarray(
        vector,
        dtype=np.float64,
    )

    norm = np.linalg.norm(vector)

    if norm < eps:
        raise ValueError(
            f"Cannot normalize near-zero vector: {vector}"
        )

    return vector / norm


def angle_between_vectors(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    degrees: bool = False,
) -> float:
    """
    Compute angle between two vectors.

    Args:
        vector_a:
            Shape [D].

        vector_b:
            Shape [D].

        degrees:
            False:
                return radians.

            True:
                return degrees.

    Returns:
        Angle between vectors.
    """

    a = normalize_vector(vector_a)
    b = normalize_vector(vector_b)

    cos_angle = np.clip(
        np.dot(a, b),
        -1.0,
        1.0,
    )

    angle = float(
        np.arccos(cos_angle)
    )

    if degrees:
        angle = float(
            np.degrees(angle)
        )

    return angle


# ==============================================================================
# Homogeneous coordinates
# ==============================================================================

def to_homogeneous(
    points: np.ndarray,
) -> np.ndarray:
    """
    Convert points to homogeneous coordinates.

    Args:
        points:
            Shape [N, D].

    Returns:
        Shape [N, D+1].
    """

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    if points.ndim != 2:
        raise ValueError(
            f"points must be 2-D, got shape {points.shape}"
        )

    ones = np.ones(
        (len(points), 1),
        dtype=np.float64,
    )

    return np.concatenate(
        [
            points,
            ones,
        ],
        axis=1,
    )


def transform_points(
    points: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    """
    Apply a 4x4 homogeneous transformation to 3D points.

    Args:
        points:
            Shape [N, 3].

        transform:
            Shape [4, 4].

    Returns:
        Transformed points with shape [N, 3].
    """

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    transform = np.asarray(
        transform,
        dtype=np.float64,
    )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points must have shape [N, 3], got {points.shape}"
        )

    if transform.shape != (4, 4):
        raise ValueError(
            f"transform must have shape [4, 4], "
            f"got {transform.shape}"
        )

    homogeneous = to_homogeneous(
        points
    )

    transformed = (
        transform
        @ homogeneous.T
    ).T

    return transformed[:, :3]


def transform_direction(
    direction: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    """
    Transform a direction vector using only rotation.

    Translation is ignored.

    Args:
        direction:
            Shape [3].

        transform:
            Shape [3,3] or [4,4].

    Returns:
        Transformed direction [3].
    """

    direction = np.asarray(
        direction,
        dtype=np.float64,
    )

    transform = np.asarray(
        transform,
        dtype=np.float64,
    )

    if direction.shape != (3,):
        raise ValueError(
            f"direction must have shape [3], "
            f"got {direction.shape}"
        )

    if transform.shape == (4, 4):

        rotation = transform[:3, :3]

    elif transform.shape == (3, 3):

        rotation = transform

    else:

        raise ValueError(
            "transform must have shape [3,3] or [4,4], "
            f"got {transform.shape}"
        )

    return rotation @ direction


# ==============================================================================
# Pinhole camera geometry
# ==============================================================================

def pinhole_image_corner_rays(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    depth: float = 1.0,
) -> np.ndarray:
    """
    Compute image-corner points in camera coordinates.

    Camera convention:

        +X : image right
        +Y : image down
        +Z : camera forward

    The four points lie on:

        Z = depth

    Returns:
        Shape [4, 3].

    Order:
        0: top-left
        1: top-right
        2: bottom-right
        3: bottom-left
    """

    if fx <= 0 or fy <= 0:
        raise ValueError(
            f"Invalid focal length: fx={fx}, fy={fy}"
        )

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid image size: {width} x {height}"
        )

    if depth <= 0:
        raise ValueError(
            f"depth must be > 0, got {depth}"
        )

    image_corners = np.array(
        [
            [0.0, 0.0],
            [float(width), 0.0],
            [float(width), float(height)],
            [0.0, float(height)],
        ],
        dtype=np.float64,
    )

    camera_points = []

    for u, v in image_corners:

        x = (
            (u - cx)
            / fx
            * depth
        )

        y = (
            (v - cy)
            / fy
            * depth
        )

        z = depth

        camera_points.append(
            [
                x,
                y,
                z,
            ]
        )

    return np.asarray(
        camera_points,
        dtype=np.float64,
    )


def build_camera_frustum_world(
    c2w: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    depth: float,
) -> np.ndarray:
    """
    Build pinhole camera frustum vertices in world coordinates.

    Args:
        c2w:
            Camera-to-world transform [4,4].

        fx, fy, cx, cy:
            Camera intrinsics.

        width, height:
            Image resolution.

        depth:
            Visualization depth of the image plane.

    Returns:
        Vertices [5,3].

    Vertex definition:
        0: camera center
        1: top-left
        2: top-right
        3: bottom-right
        4: bottom-left
    """

    c2w = np.asarray(
        c2w,
        dtype=np.float64,
    )

    if c2w.shape != (4, 4):
        raise ValueError(
            f"c2w must have shape [4,4], "
            f"got {c2w.shape}"
        )

    corners = pinhole_image_corner_rays(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        depth=depth,
    )

    camera_center = np.zeros(
        (1, 3),
        dtype=np.float64,
    )

    camera_vertices = np.concatenate(
        [
            camera_center,
            corners,
        ],
        axis=0,
    )

    world_vertices = transform_points(
        camera_vertices,
        c2w,
    )

    return world_vertices


# ==============================================================================
# Bounding-box geometry
# ==============================================================================

def bounding_box_center(
    min_bound: np.ndarray,
    max_bound: np.ndarray,
) -> np.ndarray:
    """
    Compute AABB center.
    """

    min_bound = np.asarray(
        min_bound,
        dtype=np.float64,
    )

    max_bound = np.asarray(
        max_bound,
        dtype=np.float64,
    )

    return (
        min_bound
        + max_bound
    ) / 2.0


def bounding_box_extent(
    min_bound: np.ndarray,
    max_bound: np.ndarray,
) -> np.ndarray:
    """
    Compute AABB extent along XYZ.
    """

    min_bound = np.asarray(
        min_bound,
        dtype=np.float64,
    )

    max_bound = np.asarray(
        max_bound,
        dtype=np.float64,
    )

    return (
        max_bound
        - min_bound
    )


def bounding_box_diagonal(
    min_bound: np.ndarray,
    max_bound: np.ndarray,
) -> float:
    """
    Compute AABB diagonal length.
    """

    extent = bounding_box_extent(
        min_bound,
        max_bound,
    )

    return float(
        np.linalg.norm(extent)
    )


# ==============================================================================
# Visualization geometry
# ==============================================================================

def estimate_camera_visualization_depth(
    scene_diagonal: float,
    camera_positions: np.ndarray,
    scene_scale: float = 0.015,
    spacing_scale: float = 0.35,
    min_scene_scale: float = 0.002,
    max_scene_scale: float = 0.02,
    eps: float = EPS,
) -> float:
    """
    Estimate adaptive camera-frustum visualization depth.

    Camera visualization size is jointly constrained by:

        1. point-cloud scene scale
        2. camera spatial density

    Motivation
    ----------
    Scene-scale-only visualization can produce very large camera
    frustums when camera poses are densely sampled.

    We therefore use:

        scene_based_depth
            = scene_diagonal * scene_scale

        spacing_based_depth
            = median_nearest_camera_spacing * spacing_scale

        depth
            = min(scene_based_depth, spacing_based_depth)

    followed by scene-relative min/max clipping.

    Args:
        scene_diagonal:
            Point-cloud bounding-box diagonal.

        camera_positions:
            Camera centers [N,3].

        scene_scale:
            Preferred visualization size relative to scene diagonal.

        spacing_scale:
            Maximum camera-frustum depth relative to the typical
            nearest-neighbor camera distance.

        min_scene_scale:
            Minimum depth relative to scene diagonal.

        max_scene_scale:
            Maximum depth relative to scene diagonal.

    Returns:
        Frustum visualization depth in world coordinates.
    """

    if scene_scale <= 0:
        raise ValueError(
            f"scene_scale must be > 0, got {scene_scale}"
        )

    if spacing_scale <= 0:
        raise ValueError(
            f"spacing_scale must be > 0, got {spacing_scale}"
        )

    if min_scene_scale <= 0:
        raise ValueError(
            "min_scene_scale must be > 0, "
            f"got {min_scene_scale}"
        )

    if max_scene_scale <= 0:
        raise ValueError(
            "max_scene_scale must be > 0, "
            f"got {max_scene_scale}"
        )

    if min_scene_scale > max_scene_scale:
        raise ValueError(
            "min_scene_scale must not exceed "
            "max_scene_scale."
        )

    if scene_diagonal <= eps:
        scene_diagonal = 1.0

    scene_based_depth = (
        scene_diagonal
        * scene_scale
    )

    min_depth = (
        scene_diagonal
        * min_scene_scale
    )

    max_depth = (
        scene_diagonal
        * max_scene_scale
    )

    camera_positions = np.asarray(
        camera_positions,
        dtype=np.float64,
    )

    # --------------------------------------------------------------------------
    # No meaningful camera-spacing information.
    # --------------------------------------------------------------------------

    if camera_positions.size == 0:

        return float(
            np.clip(
                scene_based_depth,
                min_depth,
                max_depth,
            )
        )

    if (
        camera_positions.ndim != 2
        or camera_positions.shape[1] != 3
    ):
        raise ValueError(
            "camera_positions must have shape [N,3], "
            f"got {camera_positions.shape}"
        )

    if len(camera_positions) < 2:

        return float(
            np.clip(
                scene_based_depth,
                min_depth,
                max_depth,
            )
        )

    # --------------------------------------------------------------------------
    # Pairwise camera-center distances.
    #
    # Camera count is normally much smaller than point count, so a dense
    # pairwise distance matrix is acceptable for visualization use.
    # --------------------------------------------------------------------------

    diff = (
        camera_positions[:, None, :]
        - camera_positions[None, :, :]
    )

    distances = np.linalg.norm(
        diff,
        axis=-1,
    )

    # Ignore distance from a camera to itself.
    np.fill_diagonal(
        distances,
        np.inf,
    )

    nearest_distances = np.min(
        distances,
        axis=1,
    )

    # Ignore duplicated / invalid camera centers.
    valid_mask = (
        np.isfinite(nearest_distances)
        & (nearest_distances > eps)
    )

    nearest_distances = nearest_distances[
        valid_mask
    ]

    if len(nearest_distances) == 0:

        return float(
            np.clip(
                scene_based_depth,
                min_depth,
                max_depth,
            )
        )

    # Median is robust to a few isolated cameras.
    typical_spacing = float(
        np.median(
            nearest_distances
        )
    )

    spacing_based_depth = (
        typical_spacing
        * spacing_scale
    )

    depth = min(
        scene_based_depth,
        spacing_based_depth,
    )

    depth = np.clip(
        depth,
        min_depth,
        max_depth,
    )

    return float(depth)