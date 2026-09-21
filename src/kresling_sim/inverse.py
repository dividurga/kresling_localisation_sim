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
    ) -> SolverReturn:
        """estimate the magnet's position from sensors (position+orientation)
        and measurements (N,3) mT. noise weights the fit, initial_guess
        (3,) mm seeds it. magnet_kwargs (diameter/height/remanence) goes to
        Kresling(...) for the forward model -- must match the magnet that
        actually generated sensor_measurements, or the fit solves for the
        wrong dipole moment entirely. returns .status and .position_estimate
        (None if failed)."""
        return SolverReturn(status=SolverResult.FAILURE, position_estimate=None)

    



class LeastSquaresSolver(PositionSolver):
    """simple least squares solver, will experiment with fancier solvers later.

    needs GaussianNoise specifically, not just any NoiseModel -- squared-
    residual weighting by 1/sigma is only maximum-likelihood for gaussian
    noise, and GaussianNoise is the only NoiseModel that exposes the
    precision() this needs. see noise.py's module docstring."""

    def solve(
        self,
        sensors: list[Sensor],
        sensor_measurements: np.ndarray,
        noise: NoiseModel,
        initial_guess: np.ndarray,
        magnet_kwargs: dict | None = None,
    ) -> SolverReturn:
        """same signature/contract as PositionSolver.solve(). noise is kept
        as the general NoiseModel type for Liskov-compatibility, but must
        actually be a GaussianNoise at runtime (checked below, raises a
        clear TypeError instead of a confusing AttributeError otherwise)."""
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

        result = least_squares(residuals, initial_guess)

        if result.success:
            return SolverReturn(status=SolverResult.SUCCESS, position_estimate=result.x)
        return SolverReturn(status=SolverResult.FAILURE, position_estimate=None)


        