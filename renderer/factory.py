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


def _make_gs_render(gaussian_ply, config, device):
    module = importlib.import_module("gs_render")
    # Private backend contract is deliberately exact, not capability-probed.
    renderer = module.create_renderer(gaussian_ply, config=config, device=device)
    return renderer, module


def create_renderer(gaussian_ply, *, backend="auto", config=None, device="auto"):
    if backend not in ("auto", "gs_render", "gsplat"):
        raise ValueError("renderer backend must be auto, gs_render, or gsplat")
    attempts = ("gs_render", "gsplat") if backend == "auto" else (backend,)
    failures = []
    for name in attempts:
        try:
            renderer, module = (_make_gs_render(gaussian_ply, config, device)
                                if name == "gs_render"
                                else _make_gsplat(gaussian_ply, config, device))
        except (ImportError, ModuleNotFoundError) as exc:
            failures.append(f"{name}: {exc}")
            continue
        version = _version(name, module)
        print(f"[GS:RENDERER] backend={name} version={version}")
        wrapped = _FailureBoundary(renderer, name, version)
        wrapped.metadata = {"backend": name, "version": version}
        return wrapped
    raise NoRendererAvailable("No Renderer Available. " + " | ".join(failures))
