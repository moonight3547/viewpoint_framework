"""Private gs_render uses its exact package-level create_renderer contract."""
from .factory import _make_gs_render
__all__ = ["_make_gs_render"]
