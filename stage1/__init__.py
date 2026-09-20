"""Stage-1 scene understanding and view-domain analysis."""

from viewpoint_framework.scene_understanding import (  # compatibility during migration
    SceneUnderstandingConfig,
    SceneUnderstandingResult,
    understand_scene,
)

__all__ = ["SceneUnderstandingConfig", "SceneUnderstandingResult", "understand_scene"]
