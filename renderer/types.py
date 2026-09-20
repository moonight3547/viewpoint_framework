from dataclasses import dataclass
from typing import Any, Optional
import numpy as np


@dataclass
class GaussianSceneData:
    """Backend-independent activated Gaussian values and one canonical split."""
    means: np.ndarray
    quats: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray
    features_dc: np.ndarray
    features_sh: np.ndarray
    sh_degree: int
    geometry_mask: np.ndarray
    skybox_mask: np.ndarray
    skybox_center: np.ndarray
    skybox_radius: float
    metadata: dict[str, Any]

    @property
    def geometry_means(self):
        return self.means[self.geometry_mask]

    @property
    def skybox_means(self):
        return self.means[self.skybox_mask]

    @property
    def skybox_present(self):
        return bool(np.any(self.skybox_mask))

    @property
    def features(self):
        return np.concatenate((self.features_dc, self.features_sh), axis=1)


@dataclass
class GaussianRenderResult:
    rgb: Optional[np.ndarray]
    alpha: np.ndarray
    depth: Optional[np.ndarray]
    width: int
    height: int
