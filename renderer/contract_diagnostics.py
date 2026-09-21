"""Read-only tensor-contract diagnostics for real gs_render smoke tests."""
from __future__ import annotations
import json
import numpy as np


GAUSSIAN_FIELDS = (
    "means", "rotations", "scales", "opacitys",
    "features_dc", "features_sh", "semantics",
)
CAMERA_FIELDS = ("width", "height", "w2c_r", "w2c_t", "intrinsic", "exposure")


def tensor_summary(value):
    if value is None:
        return {"present": False}
    if not hasattr(value, "shape"):
        return {"present": True, "type": type(value).__name__, "value": value}
    result = {
        "present": True,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(getattr(value, "device", "cpu")),
        "contiguous": bool(value.is_contiguous()) if hasattr(value, "is_contiguous") else None,
    }
    try:
        import torch
        tensor = value.detach()
        finite = torch.isfinite(tensor) if tensor.is_floating_point() else torch.ones_like(tensor, dtype=torch.bool)
        result["finite"] = bool(torch.all(finite).item())
        result["finite_count"] = int(torch.count_nonzero(finite).item())
        result["numel"] = int(tensor.numel())
        if tensor.numel() and tensor.is_floating_point() and torch.any(finite):
            valid = tensor[finite]
            result.update(min=float(valid.min().item()), max=float(valid.max().item()),
                          mean=float(valid.float().mean().item()))
    except (ImportError, TypeError, RuntimeError, AttributeError):
        array = np.asarray(value)
        finite = np.isfinite(array) if np.issubdtype(array.dtype, np.number) else np.ones(array.shape, bool)
        result.update(finite=bool(np.all(finite)), finite_count=int(np.count_nonzero(finite)),
                      numel=int(array.size))
    return result


def object_contract(value, fields):
    return {field: tensor_summary(getattr(value, field, None)) for field in fields}


def print_contract(label, gaussian_data, camera_data, config_data):
    payload = {
        "label": label,
        "gaussian": object_contract(gaussian_data, GAUSSIAN_FIELDS),
        "camera": object_contract(camera_data, CAMERA_FIELDS),
        "config": {
            key: getattr(config_data, key, None)
            for key in ("degree", "render_depth", "render_normal",
                        "clamp_color_min", "return_abs_grad", "use_bucket")
        },
    }
    payload["config"]["bg_color"] = tensor_summary(
        getattr(config_data, "bg_color", None))
    print("[GS:CONTRACT] " + json.dumps(payload, ensure_ascii=False, default=str), flush=True)
    return payload
