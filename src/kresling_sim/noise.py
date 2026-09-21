"""sensor noise models. shared by fisher.py (needs a fisher-information
contribution for the CRB) and montecarlo.py (needs to draw noise samples),
so there's one definition of "what does sensor noise look like" instead
of a bare `sigma: float` hardcoded everywhere.

every NoiseModel implements `fisher_contribution(jacobian)`, so fisher.py
and montecarlo.py don't need to know which distribution `noise` actually
is -- they just call it and get a (3,3) matrix back. that part's generic.

the MATH isn't generic though. I = J^T Sigma^-1 J only falls out because
gaussian log-likelihood is quadratic in the residual -- a different
distribution needs its own derivation. GaussianNoise below has the real
one. if a non-gaussian model shows up without a derived formula, raise
NotImplementedError in its fisher_contribution rather than guess -- it can
still be sampled for monte carlo, it just can't feed the CRB path.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class NoiseModel(ABC):
    """base for any sensor noise model -- must support drawing a
    realization (monte carlo) and its own fisher_contribution (CRB); see
    module docstring for why the latter isn't generic."""

    @abstractmethod
    def sample(self, n_sensors: int, rng: np.random.Generator) -> np.ndarray:
        """one noise realization, (n_sensors,3) mT -- matches
        stacked_forward_prediction's output shape so it can just be added
        on."""

    @abstractmethod
    def fisher_contribution(self, jacobian: np.ndarray) -> np.ndarray:
        """given the (N,3,3) forward-model jacobian, return this noise
        model's (3,3) contribution to the fisher info matrix. formula
        depends on the distribution's log-likelihood -- see module
        docstring."""


class GaussianNoise(NoiseModel):
    """independent (per-sensor, per-axis) gaussian noise. sigma can be:
      - scalar: same noise on every axis of every sensor
      - (3,): per-axis, e.g. [sigma_xy, sigma_xy, sigma_z] -- MLX90393
        datasheet shows Z is noisier than X/Y, see hardware_notes.md
      - (n_sensors, 3): fully per-sensor-per-axis
    """

    def __init__(self, sigma: float | np.ndarray):
        self.sigma = np.asarray(sigma, dtype=float)

    def _sigma_per_sensor(self, n_sensors: int) -> np.ndarray:
        """broadcast whatever shape `sigma` is into (n_sensors, 3)."""
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
        """(3n,3n) inverse noise covariance, diagonal since noise is
        independent across sensors/axes. row/col order matches fisher.py's
        flattening: sensor 0's [x,y,z] first, then sensor 1's, etc.
        gaussian-specific, not part of the abstract interface."""
        variances = self._sigma_per_sensor(n_sensors).reshape(-1) ** 2  # (3N,)
        return np.diag(1.0 / variances)

    def sample(self, n_sensors: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(scale=self._sigma_per_sensor(n_sensors))

    def fisher_contribution(self, jacobian: np.ndarray) -> np.ndarray:
        """I = J^T Sigma^-1 J -- derivation, since this isn't just asserted:

        model: b_measured = b(theta) + n, n ~ N(0, Sigma). so b_measured is
        gaussian with mean b(theta), log-likelihood

            log p(b_measured; theta) = -1/2 (b_measured-b(theta))^T Sigma^-1 (b_measured-b(theta)) + const

        (const is the normalizing prefactor, doesn't depend on theta,
        vanishes on differentiation)

        let r = b_measured - b(theta), J = db/dtheta. differentiating once
        (chain rule, dr/dtheta_j = -db/dtheta_j) gives the score:

            d(log p)/dtheta_j = J_j^T Sigma^-1 r

        differentiating again:

            d2(log p)/(dtheta_j dtheta_k) = -J_j^T Sigma^-1 J_k + (d2b/dtheta_j dtheta_k)^T Sigma^-1 r

        taking E[.]: E[r]=0 (b_measured's mean is exactly b(theta)), so the
        second term vanishes:

            -E[d2(log p)/(dtheta_j dtheta_k)] = J_j^T Sigma^-1 J_k = (J^T Sigma^-1 J)_jk

        i.e. fisher information (negative expected hessian of the
        log-likelihood, the general definition) equals J^T Sigma^-1 J
        specifically because the noise is gaussian -- a different
        distribution needs this same derivation redone against its own
        log-likelihood, which is why this isn't on the abstract NoiseModel."""
        n_sensors = jacobian.shape[0]
        jacobian_flat = jacobian.reshape(-1, jacobian.shape[-1])  # (3N, 3)
        precision = self.precision(n_sensors)  # (3N, 3N)
        return jacobian_flat.T @ precision @ jacobian_flat
