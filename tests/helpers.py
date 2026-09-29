import numpy as np

from viewpoint_framework.utils.cameras import Camera


def camera_at(index, position, forward):
    """Build a synthetic camera using the framework's +Z-forward convention."""

    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray(forward, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(up, forward))) > 0.95:
        up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = -true_up
    c2w[:3, 2] = forward
    c2w[:3, 3] = position
    return Camera(
        index=index,
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        width=100,
        height=100,
        w2c=np.linalg.inv(c2w),
        c2w=c2w,
    )
