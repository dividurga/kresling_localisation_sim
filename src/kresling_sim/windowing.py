"""local window solve, only use an n x n block of sensors around the robot.
window is picked from the solvers initial guess (previous estimate when tracking).
sensors are the row major list from sensor_grid (y outer, x inner), square grid
"""
from __future__ import annotations

import math

import numpy as np

from kresling_sim.inverse import LeastSquaresSolver, PositionSolver, SolverReturn
from kresling_sim.noise import GaussianNoise, NoiseModel
from kresling_sim.sensors import Sensor, positions_array


def select_window(positions: np.ndarray, center_xy: np.ndarray, n: int) -> np.ndarray:
    """indices of the n x n block of sensors whose center is nearest center_xy.
    near an edge the block shifts inward so it is always n x n (robot ends up off center).
    positions (N,3) from a square sensor_grid. ties go to the lower index (floor(v + 0.5))"""
    n_sensors = len(positions)
    side = int(round(math.sqrt(n_sensors)))
    if side * side != n_sensors:
        raise ValueError(f"need a square grid, got {n_sensors} sensors")
    if not 1 <= n <= side:
        raise ValueError(f"n must be in [1, {side}], got {n}")

    xs = positions[:side, 0]    # first row, x varies fastest
    ys = positions[::side, 1]   # first column
    pitch_x, pitch_y = xs[1] - xs[0], ys[1] - ys[0]
    # fractional index of the center, block start is that minus half the block, rounded
    i0 = math.floor((center_xy[0] - xs[0]) / pitch_x - (n - 1) / 2 + 0.5)
    j0 = math.floor((center_xy[1] - ys[0]) / pitch_y - (n - 1) / 2 + 0.5)
    i0 = min(max(i0, 0), side - n)
    j0 = min(max(j0, 0), side - n)

    rows = np.arange(j0, j0 + n)
    cols = np.arange(i0, i0 + n)
    return (rows[:, None] * side + cols[None, :]).ravel()


def slice_noise(noise: NoiseModel, idx: np.ndarray) -> NoiseModel:
    """noise for a subset of sensors. per sensor sigma (N,3) gets sliced, scalar or (3,) already works for any count"""
    if isinstance(noise, GaussianNoise) and noise.sigma.ndim == 2:
        return GaussianNoise(noise.sigma[idx])
    return noise


class WindowedSolver(PositionSolver):
    """least squares using only the n x n sensors around the initial guess.
    window is fixed per solve, extra multi start jitter in the inner solver is around the same guess"""

    def __init__(self, n: int, inner: PositionSolver | None = None):
        self.n = n
        self.inner = inner if inner is not None else LeastSquaresSolver()

    def solve(
        self,
        sensors: list[Sensor],
        sensor_measurements: np.ndarray,
        noise: NoiseModel,
        initial_guess: np.ndarray,
        magnet_kwargs: dict | None = None,
        rng: np.random.Generator | None = None,
    ) -> SolverReturn:
        idx = select_window(positions_array(sensors), np.asarray(initial_guess)[:2], self.n)
        return self.inner.solve(
            sensors=[sensors[i] for i in idx],
            sensor_measurements=np.asarray(sensor_measurements)[idx],
            noise=slice_noise(noise, idx),
            initial_guess=initial_guess,
            magnet_kwargs=magnet_kwargs,
            rng=rng,
        )
