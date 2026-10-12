"""pluggable reverse solver for the localisation problem"""

from __future__ import annotations
from abc import ABC, abstractmethod
import enum
from kresling_sim import forward
import numpy as np
from dataclasses import dataclass
from kresling_sim.kresling import Kresling
from kresling_sim.noise import GaussianNoise, NoiseModel
from kresling_sim.sensors import Sensor
from scipy.optimize import least_squares

class SolverResult(enum.Enum):
    """result of a position solver"""
    SUCCESS = 0
    FAILURE = 1

@dataclass
class SolverReturn:
    """return type for a position solver"""
    status: SolverResult
    position_estimate: np.ndarray | None = None  # (3,) mm, or None if failed

class PositionSolver(ABC):
    """abstract base class for a position solver, must implement solve()"""

    @abstractmethod
    def solve(
        self,
        sensors: list[Sensor],
        sensor_measurements: np.ndarray,
        noise: NoiseModel,
        initial_guess: np.ndarray,
        magnet_kwargs: dict | None = None,
        rng: np.random.Generator | None = None,
    ) -> SolverReturn:
        """estimate the magnet's position from sensors (position+orientation)
        and measurements (N,3) mT. noise weights the fit, initial_guess
        (3,) mm seeds it. magnet_kwargs (diameter/height/remanence) goes to
        Kresling(...) for the forward model -- must match the magnet that
        actually generated sensor_measurements, or the fit solves for the
        wrong dipole moment entirely. rng: optional, for solvers that draw
        randomness (e.g. multi-start jitter) -- ignored by solvers that
        don't need it. returns .status and .position_estimate
        (None if failed)."""
        return SolverReturn(status=SolverResult.FAILURE, position_estimate=None)

    



class LeastSquaresSolver(PositionSolver):
    """simple least squares solver, will experiment with fancier solvers later.

    needs GaussianNoise specifically, not just any NoiseModel -- squared-
    residual weighting by 1/sigma is only maximum-likelihood for gaussian
    noise, and GaussianNoise is the only NoiseModel that exposes the
    precision() this needs. see noise.py's module docstring.

    multi-start + residual sanity check, added per the kresling_2026_supp
    ghost-solution finding (divijas_notes.md): an in-plane-magnetized
    source, at a position aligned with a sensor row/column, has a real but
    extremely narrow basin of attraction around the true minimum (~5-6mm
    observed) sitting right next to a huge, nearly-flat false-convergence
    plateau. A single warm-started solve isn't reliably inside that basin.
    """

    def __init__(self, n_starts: int = 1, start_spread_mm: float = 20.0, max_residual_ratio: float = 5.0):
        """
        n_starts: independent starting guesses per solve() call. 1 (default)
            = today's exact behavior (uses initial_guess only, no jitter).
            >1 = also try (n_starts-1) extra starts scattered around
            initial_guess, keep whichever converged attempt has the lowest
            final cost.
        start_spread_mm: std dev (mm) of the extra scattered starts, drawn
            as initial_guess + rng.normal(0, start_spread_mm, size=3). Needs
            to be wide enough that at least one attempt plausibly lands
            inside the true basin even when initial_guess itself didn't --
            picking a good value is a tuning question (the observed basin
            edge was only 5-6mm; too narrow a spread mostly re-tries near
            the same bad start, too wide wastes attempts far from anything).
        max_residual_ratio: reject a result (return FAILURE) if its best
            final weighted cost exceeds max_residual_ratio times the
            expected noise-only cost. Expected cost under correct
            convergence: each of the 3N weighted residuals should be
            ~N(0,1) if the fit is right, so E[cost] = E[0.5*sum(r_i^2)] =
            0.5*3N (chi-squared with 3N dof). This is the cheap safeguard
            from the notes -- scipy's own result.success only checks step
            size/cost-change tolerances, never that the residual is
            actually small, which is exactly how the ghost false-converges.
        """
        self.n_starts = n_starts
        self.start_spread_mm = start_spread_mm
        self.max_residual_ratio = max_residual_ratio

    def solve(
        self,
        sensors: list[Sensor],
        sensor_measurements: np.ndarray,
        noise: NoiseModel,
        initial_guess: np.ndarray,
        magnet_kwargs: dict | None = None,
        rng: np.random.Generator | None = None,
    ) -> SolverReturn:
        """same signature/contract as PositionSolver.solve(), plus:
        - runs self.n_starts independent least_squares solves (first from
          initial_guess exactly; rest from initial_guess + rng-drawn
          jitter -- fall back to a fresh np.random.default_rng() if rng is
          None, but callers doing real Monte Carlo work should pass their
          own for reproducibility, same as simulate_measurement already
          requires elsewhere in the pipeline).
        - among attempts with result.success True, keep the one with the
          lowest result.cost (scipy provides this: 0.5*sum(residuals**2)
          at the returned x).
        - if the best result's cost still exceeds
          0.5*3*n_sensors*max_residual_ratio, return FAILURE instead of a
          silently-bad SUCCESS -- the residual sanity check, independent
          of n_starts.
        - if no attempt succeeded at all, FAILURE as before.
        noise is kept as the general NoiseModel type for Liskov-
        compatibility, but must actually be a GaussianNoise at runtime
        (checked below, raises a clear TypeError instead of a confusing
        AttributeError otherwise)."""
        if not isinstance(noise, GaussianNoise):
            raise TypeError(
                f"LeastSquaresSolver requires GaussianNoise (needs .precision() for "
                f"maximum-likelihood weighting), got {type(noise).__name__}"
            )

        n_sensors = len(sensors)
        # weight = 1/sigma = sqrt(precision), so weighted residual squared
        # and summed = (predicted-measured)^2 / sigma^2 -- the weighted
        # least-squares objective that's maximum-likelihood under gaussian noise
        weights = np.sqrt(np.diag(noise.precision(n_sensors)))  # (3N,)
        magnet_kwargs = magnet_kwargs or {}

        # residual fn for least_squares to minimize
        def residuals(position):
            predicted_measurements = forward.stacked_forward_prediction(
                Kresling(position=position, **magnet_kwargs), sensors
            )
            raw_residuals = (predicted_measurements - sensor_measurements).flatten()  # (3N,)
            return raw_residuals * weights

        rng = rng if rng is not None else np.random.default_rng()

        starts = [initial_guess]
        for _ in range(self.n_starts - 1):
            starts.append(initial_guess + rng.normal(0, self.start_spread_mm, size=3))

        best_result = None
        for start in starts:
            result = least_squares(residuals, start)
            if result.success and (best_result is None or result.cost < best_result.cost):
                best_result = result

        if best_result is None:
            return SolverReturn(status=SolverResult.FAILURE, position_estimate=None)

        # residual sanity check: reject a "converged" result whose fit is
        # still much worse than pure sensor noise would produce -- catches
        # the ghost-plateau false convergence even when it's the best (or
        # only) attempt available
        expected_cost = 0.5 * 3 * n_sensors
        if best_result.cost > self.max_residual_ratio * expected_cost:
            return SolverReturn(status=SolverResult.FAILURE, position_estimate=None)

        return SolverReturn(status=SolverResult.SUCCESS, position_estimate=best_result.x)


        