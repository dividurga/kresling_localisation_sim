"""This file implements a pluggable reverse solver for the localisation problem"""

from __future__ import annotations
from abc import ABC, abstractmethod
import enum
from kresling_sim import forward
import numpy as np
from dataclasses import dataclass
from kresling_sim.kresling import Kresling
from kresling_sim.noise import GaussianNoise, NoiseModel
from scipy.optimize import least_squares

class SolverResult(enum.Enum):
    """Enum for the result of a position solver"""
    SUCCESS = 0
    FAILURE = 1

@dataclass
class SolverReturn:
    """Return type for a position solver"""
    status: SolverResult
    position_estimate: np.ndarray | None = None  # (3,) mm, or None if failed

class PositionSolver(ABC):
    """abstract base class for a position solver, must implement solve()"""

    @abstractmethod
    def solve(
        self,
        sensor_positions: np.ndarray,
        sensor_measurements: np.ndarray,
        noise: NoiseModel,
        initial_guess: np.ndarray,
        magnet_kwargs: dict | None = None,
    ) -> SolverReturn:
        """Given a set of sensor positions and measurements, return the estimated position of the magnet.
        Args:
            sensor_positions (np.ndarray) (N,3): The positions of the sensors in space (mm).
            sensor_measurements (np.ndarray) (N,3): The magnetic field measurements from the sensors (mT).
            noise (NoiseModel): The noise model for the sensors.
            initial_guess (np.ndarray) (3,): Starting position (mm) for the solve.
            magnet_kwargs (dict | None): diameter/height/remanence passed to
                Kresling(...) for the forward model used inside the solve --
                must match the magnet that actually generated
                sensor_measurements, or the fit is solving for the wrong
                dipole moment entirely.
        Returns:
            SolverReturn: .status (SolverResult.SUCCESS/FAILURE) and
                .position_estimate (np.ndarray (3,) mm, or None if failed).
        """
        return SolverReturn(status=SolverResult.FAILURE, position_estimate=None)

    



class LeastSquaresSolver(PositionSolver):
    """Simple least squares olsver, will experriment wiuth more advanced solvers later.

    Requires GaussianNoise specifically (not just any NoiseModel) -- squared-
    residual weighting by 1/sigma is only actually maximum-likelihood for
    Gaussian noise, and GaussianNoise is the only NoiseModel that exposes
    the precision() this needs. See noise.py's module docstring."""

    def solve(
        self,
        sensor_positions: np.ndarray,
        sensor_measurements: np.ndarray,
        noise: NoiseModel,
        initial_guess: np.ndarray,
        magnet_kwargs: dict | None = None,
    ) -> SolverReturn:
        """Given a set of sensor positions and measurements, return the estimated position of the magnet.
        Args:
            sensor_positions (np.ndarray) (N,3): The positions of the sensors in space (mm).
            sensor_measurements (np.ndarray) (N,3): The magnetic field measurements from the sensors (mT).
            noise (NoiseModel): the sensor noise model, used to weight residuals. Must
                actually be a GaussianNoise at runtime (checked below) -- kept as the
                general NoiseModel in the signature so this override stays
                Liskov-compatible with PositionSolver.solve(), and fails with a clear
                message instead of a confusing AttributeError if it isn't.
            initial_guess (np.ndarray) (3,): Starting position (mm) for the solve.
            magnet_kwargs (dict | None): diameter/height/remanence for the
                forward model used inside the solve -- must match the magnet
                that generated sensor_measurements (see PositionSolver.solve).
        Returns:
            SolverReturn: .status (SolverResult.SUCCESS/FAILURE) and
                .position_estimate (np.ndarray (3,) mm, or None if failed)."""
        if not isinstance(noise, GaussianNoise):
            raise TypeError(
                f"LeastSquaresSolver requires GaussianNoise (needs .precision() for "
                f"maximum-likelihood weighting), got {type(noise).__name__}"
            )

        n_sensors = sensor_positions.shape[0]
        # per-measurement weight = 1/sigma = sqrt(precision), so that
        # weighted residual squared and summed = (predicted-measured)^2 / sigma^2,
        # i.e. exactly the weighted least-squares objective that is
        # maximum-likelihood under this Gaussian noise model.
        weights = np.sqrt(np.diag(noise.precision(n_sensors)))  # (3N,)
        magnet_kwargs = magnet_kwargs or {}

        # least squares needs a residual function that needs to be minimised
        def residuals(position):
            # compute the predicted magnetic field at the sensor positions
            predicted_measurements = forward.stacked_forward_prediction(
                Kresling(position=position, **magnet_kwargs), sensor_positions
            )
            # compute the residuals (difference between predicted and actual measurements)
            raw_residuals = (predicted_measurements - sensor_measurements).flatten()  # (3N,)
            return raw_residuals * weights

        result = least_squares(residuals, initial_guess)

        if result.success:
            return SolverReturn(status=SolverResult.SUCCESS, position_estimate=result.x)
        return SolverReturn(status=SolverResult.FAILURE, position_estimate=None)


        