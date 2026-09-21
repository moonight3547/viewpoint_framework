"""Backend-independent standard-3DGS PLY loading and V3.3 skybox split."""
from pathlib import Path
import numpy as np

from viewpoint_framework.renderer.types import GaussianSceneData
from viewpoint_framework.skybox_detection import detect_skybox_gaussians


def _sorted(names, prefix):
    return sorted((name for name in names if name.startswith(prefix)),
                  key=lambda name: int(name[len(prefix):]))


def activate_scales(values, activation="exp"):
    values = np.asarray(values, dtype=np.float32)
    if activation == "exp":
        with np.errstate(over="ignore", invalid="ignore"):
            return np.exp(values).astype(np.float32, copy=False)
    if activation == "identity":
        return values.copy()
    raise ValueError(f"Unknown scale_activation={activation}")


def activate_opacities(values, activation="sigmoid"):
    values = np.asarray(values, dtype=np.float32)
    if activation == "identity":
        return values.copy()
    if activation != "sigmoid":
        raise ValueError(f"Unknown opacity_activation={activation}")
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1. / (1. + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1. + exp_values)
    return output


def normalize_quaternions(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def load_gaussian_scene_data(gaussian_ply, config):
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise ImportError("Loading 3DGS PLY requires plyfile.") from exc
    path = Path(gaussian_ply).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"3DGS PLY does not exist: {path}")
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError(f"PLY has no vertex element: {path}")
    vertex = ply["vertex"].data
    names = list(vertex.dtype.names or ())
    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"}
    missing = sorted(required-set(names))
    if missing:
        raise ValueError("3DGS PLY is missing properties: " + ", ".join(missing))
    means = np.stack([vertex[x] for x in ("x", "y", "z")], 1).astype(np.float32)
    scales = np.stack([vertex[f"scale_{i}"] for i in range(3)], 1).astype(np.float32)
    quats = np.stack([vertex[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)
    opacities = np.asarray(vertex["opacity"], dtype=np.float32)
    dc_names, rest_names = _sorted(names, "f_dc_"), _sorted(names, "f_rest_")
    dc = np.stack([vertex[name] for name in dc_names[:3]], 1).astype(np.float32)[:, None, :]
    if rest_names and len(rest_names) % 3 == 0:
        k = len(rest_names)//3
        rest = np.stack([vertex[name] for name in rest_names], 1).astype(np.float32)
        rest = rest.reshape(len(rest), 3, k).transpose(0, 2, 1)
    else:
        rest = np.empty((len(means), 0, 3), dtype=np.float32)
    degree = int(round(np.sqrt(1+rest.shape[1])-1))
    if (degree+1)**2 != 1+rest.shape[1]:
        raise ValueError("PLY spherical-harmonic coefficient count is invalid.")
    if config.max_sh_degree is not None:
        degree = min(degree, int(config.max_sh_degree))
        rest = rest[:, :((degree+1)**2-1), :]
    # Canonical arrays remain raw.  Skybox classification alone needs physical
    # scale, so activation is local and never written back to ``scales``.
    actual_scales = activate_scales(scales, config.scale_activation)
    actual_opacities = activate_opacities(opacities, config.opacity_activation)
    detection = detect_skybox_gaussians(means, actual_scales, config.skybox)
    finite = (np.all(np.isfinite(means), 1) & np.all(np.isfinite(scales), 1)
              & np.all(np.isfinite(quats), 1) & np.isfinite(opacities)
              & np.all(np.isfinite(actual_scales), 1)
              & np.isfinite(actual_opacities) & (actual_opacities > 1e-6)
              & np.all(np.isfinite(dc), (1, 2))
              & np.all(np.isfinite(rest), (1, 2)))
    skybox = detection.skybox_mask[finite]
    geometry = ~skybox
    if not np.any(geometry):
        raise ValueError("Skybox detector removed every Gaussian.")
    return GaussianSceneData(
        means[finite], quats[finite], scales[finite], opacities[finite],
        dc[finite], rest[finite], degree, geometry, skybox,
        detection.skybox_center.astype(np.float64), float(detection.skybox_radius),
        {"source_path": str(path), "skybox": detection.diagnostics,
         "representation": {"quats": "raw", "scales": "log",
                            "opacities": "logit"}},
    )


__all__ = ["load_gaussian_scene_data", "activate_scales",
           "activate_opacities", "normalize_quaternions"]
