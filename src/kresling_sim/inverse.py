"""This file implements a pluggable reverse solver for the localisation problem"""

from __future__ import annotations
from abc import ABC, abstractmethod
import enum
import numpy as np
from dataclasses import dataclass
from kresling_sim.noise import NoiseModel

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
    def solve(self, sensor_positions: np.ndarray, sensor_measurements: np.ndarray, noise: NoiseModel, initial_guess: np.ndarray) -> SolverReturn:
        """Given a set of sensor positions and measurements, return the estimated position of the Kresling magnet.
        Args:
            sensor_positions (np.ndarray) (N,3): The positions of the sensors in space (mm).
            sensor_measurements (np.ndarray) (N,3): The magnetic field measurements from the sensors (mT).
            noise (NoiseModel): The noise model for the sensors.
            initial_guess (np.ndarray) (3,): Starting position (mm) for the solve.
        Returns:
            SolverReturn: .status (SolverResult.SUCCESS/FAILURE) and
                .position_estimate (np.ndarray (3,) mm, or None if failed).
        """
        return SolverReturn(status=SolverResult.FAILURE, position_estimate=None)

    



    