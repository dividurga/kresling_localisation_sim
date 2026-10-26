"""sanity checks for windowing.py, asserts + printed values, not pytest

uv run python tests/sanity_check_windowing.py
"""
from __future__ import annotations

import numpy as np

from kresling_sim import fisher, kresling_2026_supp_config as cfg
from kresling_sim.inverse import LeastSquaresSolver
from kresling_sim.montecarlo import DEFAULT_NOISE, simulate_measurement
from kresling_sim.sensors import positions_array, sensor_grid
from kresling_sim.windowing import WindowedSolver, select_window


def _grid(side: int = 7, pitch: float = 16.0):
    sensors = sensor_grid(side, side, pitch, pitch)
    return sensors, positions_array(sensors)


def check_block_shape_and_contents() -> None:
    """block is n x n, indices distinct, and contiguous in the grid"""
    _, pos = _grid()
    for n in (2, 3, 4, 7):
        idx = select_window(pos, np.array([5.0, -12.0]), n)
        assert len(idx) == n * n and len(set(idx)) == n * n
        assert len(set(pos[idx, 0])) == n and len(set(pos[idx, 1])) == n
    print("block shape ok")


def check_contains_nearest_sensor() -> None:
    """block holds the sensor nearest the guess (odd n for sure, any n when not at an edge)"""
    _, pos = _grid()
    rng = np.random.default_rng(0)
    for _ in range(200):
        c = rng.uniform(-40, 40, size=2)
        nearest = np.argmin(np.linalg.norm(pos[:, :2] - c, axis=1))
        for n in (2, 3, 4, 5):
            idx = select_window(pos, c, n)
            assert nearest in idx, (c, n)
    print("contains nearest sensor ok")


def check_centered_in_the_middle() -> None:
    """odd n on a sensor is centered on it, even n between four sensors"""
    _, pos = _grid(side=7, pitch=16.0)
    idx = select_window(pos, np.array([0.0, 0.0]), 3)
    assert np.allclose(pos[idx, :2].mean(axis=0), [0.0, 0.0])
    idx = select_window(pos, np.array([8.0, 8.0]), 2)
    assert np.allclose(pos[idx, :2].mean(axis=0), [8.0, 8.0])
    print("centering ok")


def check_shifts_inward_at_edges() -> None:
    """guess past the array edge or in a corner still gives n x n, pushed inside"""
    _, pos = _grid()
    for c in ([200.0, 0.0], [-200.0, 200.0], [60.0, -60.0]):
        idx = select_window(pos, np.array(c), 4)
        assert len(idx) == 16
        assert pos[idx, 0].min() >= pos[:, 0].min() and pos[idx, 0].max() <= pos[:, 0].max()
    idx = select_window(pos, np.array([-200.0, -200.0]), 3)
    assert np.allclose(pos[idx, :2].min(axis=0), pos[:, :2].min(axis=0))
    print("edge shift ok")


def check_full_window_is_all_sensors() -> None:
    _, pos = _grid(side=6, pitch=20.0)
    idx = select_window(pos, np.array([10.0, -30.0]), 6)
    assert sorted(idx) == list(range(36))
    print("n = side ok")


def check_symmetry() -> None:
    """mirroring the guess mirrors the window (first quadrant shortcut relies on this)"""
    _, pos = _grid(side=7, pitch=16.0)
    rng = np.random.default_rng(1)
    for n in (2, 3, 4):
        for _ in range(100):
            c = rng.uniform(0, 50, size=2)
            # half a pitch ties are measure zero for random draws
            a = pos[select_window(pos, c, n), :2]
            b = pos[select_window(pos, -c, n), :2]
            assert np.allclose(np.sort(a, axis=0), np.sort(-b, axis=0))
    print("mirror symmetry ok")


def check_solver_matches_full_when_n_is_side() -> None:
    sensors, _ = _grid(side=6, pitch=20.0)
    true = np.array([15.0, -10.0, 12.35])
    meas = simulate_measurement(true, sensors, DEFAULT_NOISE, cfg.MAGNET_KWARGS, np.random.default_rng(3))
    guess = true + np.array([2.0, -1.0, 1.0])
    full = LeastSquaresSolver().solve(sensors, meas, DEFAULT_NOISE, guess, cfg.MAGNET_KWARGS)
    win = WindowedSolver(6).solve(sensors, meas, DEFAULT_NOISE, guess, cfg.MAGNET_KWARGS)
    assert np.allclose(full.position_estimate, win.position_estimate, atol=1e-6)
    print("windowed (n = side) == full solve:", win.position_estimate)


def check_windowed_solve_recovers_position() -> None:
    sensors, _ = _grid(side=7, pitch=16.0)
    true = np.array([5.0, 20.0, 12.35])
    meas = simulate_measurement(true, sensors, DEFAULT_NOISE, cfg.MAGNET_KWARGS, np.random.default_rng(4))
    r = WindowedSolver(4).solve(sensors, meas, DEFAULT_NOISE, true + np.array([2.0, 2.0, 0.0]), cfg.MAGNET_KWARGS)
    err = np.linalg.norm(r.position_estimate - true)
    print(f"windowed n=4 error: {err:.4f} mm")
    assert err < 0.5


def check_windowed_crb_equals_full_when_n_is_side() -> None:
    sensors, pos = _grid(side=6, pitch=20.0)
    true = np.array([15.0, -10.0, 12.35])
    full = fisher.crlb(fisher.fisher_information_matrix(
        fisher.position_jacobian(true, sensors, magnet_kwargs=cfg.MAGNET_KWARGS), DEFAULT_NOISE))
    idx = select_window(pos, true[:2], 6)
    win = fisher.crlb(fisher.fisher_information_matrix(
        fisher.position_jacobian(true, [sensors[i] for i in idx], magnet_kwargs=cfg.MAGNET_KWARGS), DEFAULT_NOISE))
    assert np.allclose(full, win)
    print("windowed CRB (n = side) == full CRB:", win)


if __name__ == "__main__":
    check_block_shape_and_contents()
    check_contains_nearest_sensor()
    check_centered_in_the_middle()
    check_shifts_inward_at_edges()
    check_full_window_is_all_sensors()
    check_symmetry()
    check_solver_matches_full_when_n_is_side()
    check_windowed_solve_recovers_position()
    check_windowed_crb_equals_full_when_n_is_side()
    print("all sanity checks passed")
