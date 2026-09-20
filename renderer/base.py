"""Structural renderer protocol used by Stage 2 and Stage 3."""
from typing import Protocol

class RendererBackend(Protocol):
    backend_name: str
    backend_version: str
    def render_geometry(self, camera, max_image_dim=None, *, need_rgb=True,
                        need_alpha=True, need_depth=False): ...
    def render_geometry_depth(self, camera, max_image_dim=None): ...
