"""Monte Carlo validation of CRB-predicted localization accuracy.

simulate noisy Hall-sensor measurements at
known true magnet positions, recover each trial's position estimate via
`kresling_sim.inverse`, and compare the empirical RMS position error against
the Cramer-Rao bound from `fisher.py`. If the two track each other on the
top CRB-ranked sensor configs, the CRB sweep can be trusted as a fast proxy
for the (much slower) full nonlinear solve when searching sensor
count/placement.

Position recovery is pluggable: see `kresling_sim.inverse.PositionSolver`.
This module depends only on that interface (default: `LeastSquaresSolver`,
still unimplemented -- see inverse.py), not on any specific algorithm, so a
different solving strategy can be swapped in via the `solver` argument
without changing anything here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kresling_sim import fisher, forward
from kresling_sim.inverse import LeastSquaresSolver, PositionSolver, SolverResult
from kresling_sim.kresling import Kresling
from kresling_sim.noise import GaussianNoise, NoiseModel

# Melexis MLX90393 datasheet (Rev 012, Nov 2025), section 13.1 "Noise vs
# Conversion Time", Figures 2 & 3, GAIN_SEL=0x7: RMS noise stdev read off
# the chart at the ~10-14ms conversion time operating point (chosen as a
# middle ground between actuation-off sensing-window length and noise
# floor -- see hardware_notes.md). XY ~= 8 mGauss = 0.0008 mT, Z ~= 12-17
# mGauss = 0.0014 mT (Z is consistently noisier than XY at every
# conversion time in the datasheet, not specific to this operating point).
DEFAULT_NOISE = GaussianNoise(np.array([0.0008, 0.0008, 0.0014]))


@dataclass
class MonteCarloResult:
    """Aggregated outcome of repeated noisy solves at one true position,
    for one sensor layout."""

    true_position: np.ndarray  # (3,) mm
    estimates: np.ndarray  # (n_trials, 3) mm, one row per trial
    converged: np.ndarray  # (n_trials,) bool
    errors_mm: np.ndarray  # (n_converged,) position error norm, converged trials only
    rmse_mm: float
    crb_rmse_mm: float


def simulate_measurement(
    true_position: np.ndarray,
    points: np.ndarray,
    noise: NoiseModel,
    magnet_kwargs: dict | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """One noisy field snapshot (N,3, mT) at `points` from a magnet at
    `true_position`. Accepts any NoiseModel -- sampling is defined for every
    noise model, unlike the CRB path below (see noise.py)."""
    rng = rng if rng is not None else np.random.default_rng()
    magnet = Kresling(position=np.asarray(true_position, dtype=float), **(magnet_kwargs or {}))
    clean = forward.stacked_forward_prediction(magnet, points)
    return clean + noise.sample(n_sensors=clean.shape[0], rng=rng)


def crb_position_rmse(true_position: np.ndarray, points: np.ndarray, noise: NoiseModel) -> float:
    """sqrt(trace(inv(FIM))): the CRB's lower bound on achievable RMS
    position error (mm), for comparison against the empirical MC result.
    Type-agnostic to which NoiseModel this is (fisher.fisher_information_matrix
    just delegates to noise.fisher_contribution) -- but that only actually
    works if `noise` has a real fisher_contribution implementation; a
    non-Gaussian model without one will raise NotImplementedError here,
    same as it would inside fisher.py. See noise.py's module docstring."""
    jacobian = fisher.position_jacobian(np.asarray(true_position, dtype=float), points)
    fim = fisher.fisher_information_matrix(jacobian, noise)
    return float(np.sqrt(np.trace(np.linalg.inv(fim))))


def default_initial_guess(points: np.ndarray, z_guess: float = 15.0) -> np.ndarray:
    """A trial-independent starting point for the nonlinear solve: sensor-
    grid centroid in (x, y), fixed nominal height in z. Deliberately NOT the
    true position -- seeding at the answer would hide convergence failures
    that a real localization run would actually hit."""
    centroid_xy = np.asarray(points, dtype=float)[:, :2].mean(axis=0)
    return np.array([centroid_xy[0], centroid_xy[1], z_guess])


def run_monte_carlo(
    true_position: np.ndarray,
    points: np.ndarray,
    noise: NoiseModel = DEFAULT_NOISE,
    solver: PositionSolver = LeastSquaresSolver(),
    n_trials: int = 500,
    initial_guess: np.ndarray | None = None,
    magnet_kwargs: dict | None = None,
    rng: np.random.Generator | None = None,
) -> MonteCarloResult:
    """Run `n_trials` independent noisy solves for one true magnet position
    and one sensor layout, and summarize the recovered-position error
    against the CRB predicted for the same config. `noise` is typed as the
    general NoiseModel now -- both the simulation and the CRB comparison
    below accept any noise model, but the CRB side (crb_position_rmse) will
    raise NotImplementedError if `noise.fisher_contribution` isn't actually
    implemented for whatever distribution you pass (see noise.py). `solver`
    is any PositionSolver (default LeastSquaresSolver, see inverse.py) --
    swap it to try a different position-recovery strategy without changing
    this function."""
    rng = rng if rng is not None else np.random.default_rng()
    true_position = np.asarray(true_position, dtype=float)
    points = np.asarray(points, dtype=float)
    x0 = default_initial_guess(points) if initial_guess is None else np.asarray(initial_guess, dtype=float)

    estimates = np.empty((n_trials, 3))
    converged = np.empty(n_trials, dtype=bool)

    for i in range(n_trials):
        measurement = simulate_measurement(true_position, points, noise, magnet_kwargs, rng)
        result = solver.solve(
            sensor_positions=points,
            sensor_measurements=measurement,
            noise=noise,
            initial_guess=x0,
        )
        converged[i] = result.status == SolverResult.SUCCESS
        if converged[i]:
            estimates[i] = result.position_estimate

    if not converged.any():
        raise RuntimeError(f"all {n_trials} inverse solves failed to converge at true_position={true_position}")

    errors_mm = np.linalg.norm(estimates[converged] - true_position, axis=1)
    rmse_mm = float(np.sqrt(np.mean(errors_mm**2)))

    return MonteCarloResult(
        true_position=true_position,
        estimates=estimates,
        converged=converged,
        errors_mm=errors_mm,
        rmse_mm=rmse_mm,
        crb_rmse_mm=crb_position_rmse(true_position, points, noise),
    )


def sweep_true_positions(
    true_positions: np.ndarray,
    points: np.ndarray,
    noise: NoiseModel = DEFAULT_NOISE,
    solver: PositionSolver = LeastSquaresSolver(),
    n_trials: int = 500,
    **kwargs,
) -> list[MonteCarloResult]:
    """Run `run_monte_carlo` at each of several true positions (e.g. sampled
    across the workspace) for one fixed sensor layout, so CRB-vs-empirical
    agreement can be checked over the whole volume rather than one point."""
    true_positions = np.atleast_2d(np.asarray(true_positions, dtype=float))
    return [
        run_monte_carlo(pos, points, noise=noise, solver=solver, n_trials=n_trials, **kwargs)
        for pos in true_positions
    ]


def __main__():
    # Example usage: a small planar sensor grid at the z=0 sensor plane
    # (workspace z-range per confirmed v1 scope is ~3mm to ~28.3mm above
    # this plane), one true magnet position, 200 noisy trials.
    xs, ys = np.meshgrid(np.linspace(-20, 20, 5), np.linspace(-20, 20, 5))
    points = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    true_position = np.array([3.0, -2.0, 15.0])

    result = run_monte_carlo(true_position, points, n_trials=200)
    print(f"true position: {result.true_position} mm")
    print(f"converged: {result.converged.sum()}/{len(result.converged)} trials")
    print(f"empirical RMSE: {result.rmse_mm:.4f} mm")
    print(f"CRB RMSE:       {result.crb_rmse_mm:.4f} mm")


if __name__ == "__main__":
    __main__()
