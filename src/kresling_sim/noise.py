"""Sensor noise models, shared by fisher.py (needs a precision/weighting
matrix for the CRB) and montecarlo.py (needs to draw noise realizations) so
there is exactly one definition of "what does sensor noise look like" in
this codebase, instead of a bare `sigma: float` hardcoded into each.

Important asymmetry, worth understanding before adding a new model: EVERY
noise model can be *sampled* (used to simulate noisy measurements for Monte
Carlo), but only *Gaussian* noise models have a valid *precision* matrix for
the Fisher-information/CRB computation. That's not a design limitation to
work around -- it's a real mathematical fact. The I = J^T Sigma^-1 J formula
in fisher.py was derived (see project notes) specifically from the Gaussian
log-likelihood; it is not a generic "plug in any noise model" formula. A
non-Gaussian NoiseModel (e.g. uniform or Laplace-distributed sensor noise)
is legitimate for Monte Carlo -- it'll tell you how the real solver behaves
under realistic non-Gaussian noise -- but it should NOT be handed to
fisher_information_matrix, since JᵀΣ⁻¹J would silently compute a number that
is not actually a valid Cramer-Rao bound for that distribution. This module
enforces that distinction structurally: `precision()` only exists on
`GaussianNoise`, not on the base `NoiseModel`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class NoiseModel(ABC):
    """Base for any sensor noise model. Every noise model must support
    drawing a realization (for Monte Carlo); that's the only capability
    that's valid regardless of the underlying distribution."""

    @abstractmethod
    def sample(self, n_sensors: int, rng: np.random.Generator) -> np.ndarray:
        """One noise realization, shape (n_sensors, 3) mT -- matching
        stacked_forward_prediction's output shape, so it can be added
        directly to a clean forward-model prediction."""


class GaussianNoise(NoiseModel):
    """Independent (per-sensor, per-axis) Gaussian noise. sigma may be:
      - a scalar: same noise on every axis of every sensor
      - shape (3,): per-axis (e.g. [sigma_xy, sigma_xy, sigma_z] -- the
        MLX90393 datasheet shows Z-axis noise is genuinely higher than X/Y,
        see hardware_notes.md), same for every sensor
      - shape (n_sensors, 3): fully per-sensor-per-axis

    This is the only noise model with a `precision()` method, because it's
    the only one for which fisher.py's I = J^T Sigma^-1 J is mathematically
    valid (see module docstring).
    """

    def __init__(self, sigma: float | np.ndarray):
        self.sigma = np.asarray(sigma, dtype=float)

    def _sigma_per_sensor(self, n_sensors: int) -> np.ndarray:
        """Broadcast whatever shape `sigma` was given into (n_sensors, 3)."""
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
        """(3*n_sensors, 3*n_sensors) inverse noise covariance, diagonal
        since noise is assumed independent across sensors and axes. Row/col
        order matches fisher.py's jacobian flattening: sensor 0's [x,y,z]
        rows first, then sensor 1's, etc."""
        variances = self._sigma_per_sensor(n_sensors).reshape(-1) ** 2  # (3N,)
        return np.diag(1.0 / variances)

    def sample(self, n_sensors: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(scale=self._sigma_per_sensor(n_sensors))
