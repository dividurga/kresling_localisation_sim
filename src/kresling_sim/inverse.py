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
        """estimate magnet position from sensors and measurements (N,3) mT. noise weights the fit,
        initial_guess (3,) mm seeds it. magnet_kwargs must match the magnet that made the
        measurements. rng only used by solvers that randomise (multi start)"""
        return SolverReturn(status=SolverResult.FAILURE, position_estimate=None)

    



class LeastSquaresSolver(PositionSolver):
    """weighted least squares. needs GaussianNoise (1/sigma weighting is only max likelihood for gaussian, see noise.py).

    multi start + residual check added for the kresling_2026_supp ghost solutions (divijas_notes.md).
    in plane magnet has a narrow basin (~5-6mm) next to a flat false plateau so one warm start isnt reliable
    """

    def __init__(self, n_starts: int = 1, start_spread_mm: float = 20.0, max_residual_ratio: float = 5.0):
        """
        n_starts: starting guesses per solve. 1 = initial_guess only, >1 adds jittered starts, keeps lowest cost
        start_spread_mm: stdev of the jitter, initial_guess + normal(0, spread). basin is ~5-6mm so tune this
        max_residual_ratio: FAILURE if best cost > ratio * expected noise cost (0.5*3N, chi squared).
            scipy success doesnt check the residual is small, thats how the ghost converges
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
        """n_starts least_squares solves (first from initial_guess, rest jittered, pass rng for repeatability).
        keeps the lowest cost converged one, FAILURE if none converged or cost fails the residual check.
        noise must be GaussianNoise at runtime, TypeError otherwise"""
        if not isinstance(noise, GaussianNoise):
            raise TypeError(
                f"LeastSquaresSolver requires GaussianNoise (needs .precision() for "
                f"maximum-likelihood weighting), got {type(noise).__name__}"
            )

        n_sensors = len(sensors)
        # weight = 1/sigma so the cost is sum (pred - meas)^2 / sigma^2
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

        # reject "converged" fits much worse than noise alone would give (ghost plateau)
        expected_cost = 0.5 * 3 * n_sensors
        if best_result.cost > self.max_residual_ratio * expected_cost:
            return SolverReturn(status=SolverResult.FAILURE, position_estimate=None)

        return SolverReturn(status=SolverResult.SUCCESS, position_estimate=best_result.x)


        