"""Strict V3.3 renderer selection and runtime-failure boundary."""
from __future__ import annotations
import importlib
from importlib import metadata as package_metadata


class NoRendererAvailable(RuntimeError):
    pass


class RendererRasterizationFailure(RuntimeError):
    pass


def _version(name, module):
    value = getattr(module, "__version__", None)
    if value:
        return str(value)
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
        return "unknown"


class _FailureBoundary:
    """Never switch backend after selection; normalize rasterization errors."""
    def __init__(self, renderer, backend, version):
        object.__setattr__(self, "_renderer", renderer)
        object.__setattr__(self, "backend_name", backend)
        object.__setattr__(self, "backend_version", version)

    def __getattr__(self, name):
        value = getattr(self._renderer, name)
        if not callable(value) or not name.startswith("render"):
            return value
        def guarded(*args, **kwargs):
            try:
                return value(*args, **kwargs)
            except Exception as exc:
                raise RendererRasterizationFailure(
                    f"Renderer Rasterization Failure ({self.backend_name}): {exc}") from exc
        return guarded

    def __setattr__(self, name, value):
        setattr(self._renderer, name, value)


def _make_gsplat(gaussian_ply, config, device):
    module = importlib.import_module("gsplat")
    from viewpoint_framework.gs_renderer import GsplatRenderer
    return GsplatRenderer(gaussian_ply, config=config, device=device), module


def _make_gs_render(module, gaussian_ply, config, device):
    from viewpoint_framework.renderer.gs_render_backend import GsRenderRenderer
    return GsRenderRenderer(gaussian_ply, module, config, device=device)


def create_renderer(gaussian_ply, *, backend="auto", config=None, device="auto"):
    if backend not in ("auto", "gs_render", "gsplat"):
        raise ValueError("renderer backend must be auto, gs_render, or gsplat")
    if config is None:
        from viewpoint_framework.gs_renderer import GaussianRendererConfig
        config = GaussianRendererConfig()
    print(f"[GS:RENDERER] requested={backend}")
    if backend in ("auto", "gs_render"):
        try:
            module = importlib.import_module("gs_render")
        except ModuleNotFoundError as exc:
            if exc.name != "gs_render":
                raise
            if backend == "gs_render":
                raise NoRendererAvailable(
                    f"No Renderer Available. gs_render: {exc}") from exc
            print(f"[GS:RENDERER] gs_render unavailable: {exc}")
        else:
            # Import success locks this invocation to gs_render. Any adapter
            # initialization failure propagates; gsplat is never attempted.
            renderer = _make_gs_render(module, gaussian_ply, config, device)
            version = _version("gs_render", module)
            print(f"[GS:RENDERER] backend=gs_render version={version} device={renderer.device}")
            wrapped = _FailureBoundary(renderer, "gs_render", version)
            wrapped.metadata = {"backend": "gs_render", "version": version}
            return wrapped
    try:
        renderer, module = _make_gsplat(gaussian_ply, config, device)
    except (ImportError, ModuleNotFoundError) as exc:
        raise NoRendererAvailable(f"No Renderer Available. gsplat: {exc}") from exc
    version = _version("gsplat", module)
    if backend == "auto":
        print("[GS:RENDERER] fallback backend=gsplat")
    print(f"[GS:RENDERER] backend=gsplat version={version} device={renderer.device}")
    wrapped = _FailureBoundary(renderer, "gsplat", version)
    wrapped.metadata = {"backend": "gsplat", "version": version}
    return wrapped
