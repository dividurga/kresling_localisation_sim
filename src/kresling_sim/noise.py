"""sensor noise models, used by fisher.py (CRB) and montecarlo.py (sampling).

fisher_contribution(jacobian) is the common interface but the math isnt generic.
I = J^T Sigma^-1 J only holds for gaussian noise. a non gaussian model needs its own
derivation, otherwise raise NotImplementedError (it can still be sampled for monte carlo)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class NoiseModel(ABC):
    """base noise model, needs sample (monte carlo) and fisher_contribution (CRB)"""

    @abstractmethod
    def sample(self, n_sensors: int, rng: np.random.Generator) -> np.ndarray:
        """one noise draw, (n_sensors,3) mT"""

    @abstractmethod
    def fisher_contribution(self, jacobian: np.ndarray) -> np.ndarray:
        """(3,3) fisher info from the (N,3,3) jacobian, depends on the distribution"""


class GaussianNoise(NoiseModel):
    """independent gaussian noise per sensor per axis. sigma is a scalar, (3,) per axis
    (z is noisier on the MLX90393, see hardware_notes.md), or (n_sensors, 3)
    """

    def __init__(self, sigma: float | np.ndarray):
        self.sigma = np.asarray(sigma, dtype=float)

    def _sigma_per_sensor(self, n_sensors: int) -> np.ndarray:
        """sigma broadcast to (n_sensors, 3)"""
        sigma = self.sigma
        if sigma.ndim == 0:
            return np.full((n_sensors, 3), float(sigma))
        if sigma.shape == (3,):
            return np.tile(sigma, (n_sensors, 1))
        if sigma.shape == (n_sensors, 3):
            return sigma
        raise ValueError(
            f"sigma shape {sigma.shape} is not broadcastable to "
            f"(n_sensors=3,) or ({n_sensors}, 3)"
        )

    def precision(self, n_sensors: int) -> np.ndarray:
        """(3n,3n) inverse covariance, diagonal. order is sensor 0 xyz, sensor 1 xyz etc. gaussian only"""
        variances = self._sigma_per_sensor(n_sensors).reshape(-1) ** 2  # (3N,)
        return np.diag(1.0 / variances)

    def sample(self, n_sensors: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(scale=self._sigma_per_sensor(n_sensors))

    def fisher_contribution(self, jacobian: np.ndarray) -> np.ndarray:
        """I = J^T Sigma^-1 J. b_meas = b(theta) + n, n ~ N(0, Sigma), so
        log p = -1/2 r^T Sigma^-1 r + const with r = b_meas - b(theta).
        score is J^T Sigma^-1 r, second derivative is -J^T Sigma^-1 J + (d2b)^T Sigma^-1 r.
        E[r] = 0 kills the second term so I = -E[hessian] = J^T Sigma^-1 J.
        only true for gaussian, other distributions need the derivation redone"""
        n_sensors = jacobian.shape[0]
        jacobian_flat = jacobian.reshape(-1, jacobian.shape[-1])  # (3N, 3)
        precision = self.precision(n_sensors)  # (3N, 3N)
        return jacobian_flat.T @ precision @ jacobian_flat
