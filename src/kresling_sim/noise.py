"""Sensor noise models, shared by fisher.py (needs a Fisher-information
contribution for the CRB) and montecarlo.py (needs to draw noise
realizations) so there is exactly one definition of "what does sensor
noise look like" in this codebase, instead of a bare `sigma: float`
hardcoded into each.

making every NoiseModel implement `fisher_contribution(jacobian)` DOES make
fisher.py and montecarlo.py fully agnostic at the type level -- neither one
needs to know or care which distribution `noise` actually is anymore, they
just call `noise.fisher_contribution(jacobian)` and get a (3,3) matrix back.
that part's a real, clean win.

what it does NOT do is make the MATH generic. Fisher information is a real
concept for any distribution (I(theta) = E[(d/dtheta log p)^2]), but the
clean I = J^T Sigma^-1 J formula only falls out because Gaussian
log-likelihood is a quadratic in the residual. a different distribution
(uniform, Laplace, whatever) has a genuinely different formula, and it has
to be derived by hand for that distribution -- there's no generic
"fisher_contribution" implementation that works for an arbitrary p(noise)
without knowing p(noise)'s actual shape. GaussianNoise below has the real,
correct implementation. if a non-Gaussian NoiseModel gets added later and
someone hasn't worked out its Fisher information formula, the honest move
is to raise NotImplementedError in its fisher_contribution rather than
guess -- it can still be sampled (Monte Carlo works fine on it), it just
can't feed the CRB path until someone does that derivation. one option if
there's truly no closed form: estimate I(theta) numerically by Monte-Carlo
averaging (d/dtheta log p)^2 over many drawn samples -- doable, but it
turns "fast analytical CRB" into "another Monte Carlo loop", which defeats
the reason we wanted the fast/slow two-lane split in the first place. so:
architecture's agnostic now, the math per-distribution is still homework.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class NoiseModel(ABC):
    """Base for any sensor noise model. Every noise model must support
    drawing a realization (for Monte Carlo) AND providing its own
    Fisher-information contribution (for the CRB) -- see the module
    docstring for why the latter can't have one generic implementation."""

    @abstractmethod
    def sample(self, n_sensors: int, rng: np.random.Generator) -> np.ndarray:
        """One noise realization, shape (n_sensors, 3) mT -- matching
        stacked_forward_prediction's output shape, so it can be added
        directly to a clean forward-model prediction."""

    @abstractmethod
    def fisher_contribution(self, jacobian: np.ndarray) -> np.ndarray:
        """Given the (N,3,3) forward-model Jacobian [sensor, field_component,
        pose_param], return this noise model's (3,3) contribution to the
        Fisher information matrix for the position parameters. The correct
        formula depends on the noise distribution's own log-likelihood --
        see module docstring."""


class GaussianNoise(NoiseModel):
    """Independent (per-sensor, per-axis) Gaussian noise. sigma may be:
      - a scalar: same noise on every axis of every sensor
      - shape (3,): per-axis (e.g. [sigma_xy, sigma_xy, sigma_z] -- the
        MLX90393 datasheet shows Z-axis noise is genuinely higher than X/Y,
        see hardware_notes.md), same for every sensor
      - shape (n_sensors, 3): fully per-sensor-per-axis
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
        rows first, then sensor 1's, etc. Gaussian-specific -- not part of
        the abstract NoiseModel interface, see module docstring."""
        variances = self._sigma_per_sensor(n_sensors).reshape(-1) ** 2  # (3N,)
        return np.diag(1.0 / variances)

    def sample(self, n_sensors: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(scale=self._sigma_per_sensor(n_sensors))

    def fisher_contribution(self, jacobian: np.ndarray) -> np.ndarray:
        """I = J^T Sigma^-1 J. Derivation, since this formula isn't just
        asserted -- it falls straight out of the Gaussian log-likelihood:

        Model: b_measured = b(theta) + n, n ~ N(0, Sigma). So b_measured is
        Gaussian with mean b(theta), and its log-likelihood is

            log p(b_measured; theta)
                = -1/2 (b_measured - b(theta))^T Sigma^-1 (b_measured - b(theta)) + const

        (the "+const" is the normalizing prefactor -(k/2)log(2*pi) - 1/2 log|Sigma|,
        which doesn't depend on theta and vanishes on differentiation).

        Let r = b_measured - b(theta) and J = db/dtheta. Differentiating once
        (chain rule, dr/dtheta_j = -db/dtheta_j) gives the score:

            d(log p)/dtheta_j = J_j^T Sigma^-1 r

        Differentiating again:

            d2(log p)/(dtheta_j dtheta_k) = -J_j^T Sigma^-1 J_k + (d2b/dtheta_j dtheta_k)^T Sigma^-1 r

        Taking E[.] over the noise: E[r] = 0 (b_measured's mean is exactly
        b(theta)), so the second term vanishes, leaving

            -E[d2(log p)/(dtheta_j dtheta_k)] = J_j^T Sigma^-1 J_k = (J^T Sigma^-1 J)_jk

        i.e. Fisher information (negative expected Hessian of the
        log-likelihood, the general definition for any distribution) equals
        J^T Sigma^-1 J specifically because the noise is Gaussian -- a
        different noise distribution would need this same derivation redone
        against its own log-likelihood, which is exactly why this method
        isn't on the abstract NoiseModel (see module docstring)."""
        n_sensors = jacobian.shape[0]
        jacobian_flat = jacobian.reshape(-1, jacobian.shape[-1])  # (3N, 3)
        precision = self.precision(n_sensors)  # (3N, 3N)
        return jacobian_flat.T @ precision @ jacobian_flat
