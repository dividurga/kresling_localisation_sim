"""sanity checks for tracking.py, asserts + printed values, not pytest

uv run python tests/sanity_check_tracking.py
"""
from __future__ import annotations

import numpy as np

from kresling_sim import forward, kresling_2026_supp_config as cfg
from kresling_sim.config import SENSOR_PCB_GAP_MM
from kresling_sim.inverse import LeastSquaresSolver, PositionSolver, SolverResult, SolverReturn
from kresling_sim.kresling import Kresling
from kresling_sim.montecarlo import DEFAULT_NOISE, crb_position_rmse
from kresling_sim.noise import GaussianNoise
from kresling_sim.sensors import sensor_grid
from kresling_sim.tracking import (
    FACE_ANGLE, HEX_SIDE_MM, CycleTiming, RollingTrajectory, measure_sample, readout_time_s, rolling_position, track,
)


class HoldSolver(PositionSolver):
    """returns the warm start, no work"""

    def solve(self, sensors, sensor_measurements, noise, initial_guess, magnet_kwargs=None, rng=None):
        return SolverReturn(status=SolverResult.SUCCESS, position_estimate=np.asarray(initial_guess, dtype=float))


def check_geometry() -> None:
    """z between gap + apothem and gap + side, flat on every face boundary, x advances one side per face"""
    phi = np.linspace(0, 6 * FACE_ANGLE, 6001)
    p = rolling_position(phi)
    apothem = cfg.HEX_PLATE_INSCRIBED_RADIUS_MM
    assert abs(p[:, 2].min() - (SENSOR_PCB_GAP_MM + apothem)) < 1e-6, p[:, 2].min()
    assert abs(p[:, 2].max() - (SENSOR_PCB_GAP_MM + HEX_SIDE_MM)) < 1e-6, p[:, 2].max()
    assert abs(HEX_SIDE_MM - 6.93) < 0.01
    faces = rolling_position(np.arange(7) * FACE_ANGLE)
    assert np.allclose(np.diff(faces[:, 0]), HEX_SIDE_MM)
    assert np.allclose(faces[:, 2], SENSOR_PCB_GAP_MM + apothem)
    assert p[:, 2].max() < cfg.Z_MAX_MM and p[:, 2].min() > cfg.Z_MIN_MM
    print(f"geometry ok, z {p[:, 2].min():.3f} to {p[:, 2].max():.3f}, side {HEX_SIDE_MM:.3f}")


def check_trajectory() -> None:
    """mean x speed matches, ends rest flat inside the span, comes back to the start"""
    traj = RollingTrajectory(speed_mm_s=20.0)
    t = np.linspace(0, traj.pass_time_s, 2001)
    x = traj.position(t)[:, 0]
    assert abs((x[-1] - x[0]) / traj.pass_time_s - 20.0) < 1e-6
    assert np.all(np.diff(x) >= -1e-9)  # no backwards x going forward
    assert abs(x[0] + x[-1]) < 1e-6 and x[-1] - x[0] > 90
    assert np.allclose(traj.position(2 * traj.pass_time_s), traj.position(0.0))
    mid = traj.position(1.5 * traj.pass_time_s)
    assert np.allclose(mid, traj.position(0.5 * traj.pass_time_s))  # reverse retraces the same path
    print(f"trajectory ok, {traj.n_faces} faces, x {x[0]:.2f} to {x[-1]:.2f}, pass {traj.pass_time_s:.2f}s")


def check_timing() -> None:
    """cycle and i2c numbers from the datasheet table, sample timestamps"""
    t = CycleTiming()
    assert abs(t.period_s - 49.5e-3) < 1e-9, t.period_s  # 10 + 24.8 + 14 * 1.05
    assert abs(readout_time_s(36, 3, 400e3) - 2.74e-3) < 0.01e-3
    assert abs(readout_time_s(36, 1, 400e3) - 8.21e-3) < 0.02e-3
    assert abs(readout_time_s(36, 1, 100e3) - 33.0e-3) < 0.1e-3
    assert abs(readout_time_s(36, 3, 400e3, n_bytes=12) - 3.29e-3) < 0.02e-3  # with temperature
    assert abs(t.t_sample(0) - (10e-3 + 24.8e-3 + 7e-3)) < 1e-12  # conversion midpoint
    assert abs(t.t_avail(0) - (t.period_s + t.readout_s)) < 1e-12  # next cycle's coils on + readout
    assert abs(t.t_sample(3) - t.t_sample(0) - 3 * t.period_s) < 1e-12
    print(f"timing ok, period {t.period_s * 1e3:.1f} ms, readout {t.readout_s * 1e3:.2f} ms")


def check_axis_slots() -> None:
    """x, y, z each come from the robot at their own third of the conversion"""
    sensors = sensor_grid(6, 6, 18, 18)
    traj = RollingTrajectory(speed_mm_s=160.0)
    timing = CycleTiming()
    t = timing.t_sample(5)
    d = timing.conv_s / 3
    quiet = GaussianNoise(np.zeros(3))
    rng = np.random.default_rng(0)
    skew = measure_sample(traj, t, timing, sensors, quiet, cfg.MAGNET_KWARGS, rng, per_axis=True)
    flat = measure_sample(traj, t, timing, sensors, quiet, cfg.MAGNET_KWARGS, rng, per_axis=False)
    at = lambda s: forward.stacked_forward_prediction(Kresling(position=traj.position(s), **cfg.MAGNET_KWARGS), sensors)
    assert np.allclose(skew[:, 0], at(t - d)[:, 0]) and np.allclose(skew[:, 1], at(t)[:, 1])
    assert np.allclose(skew[:, 2], at(t + d)[:, 2])
    assert np.allclose(flat, at(t))
    assert np.allclose(skew[:, 1], flat[:, 1])  # y slot is centered on t
    assert np.abs(skew[:, 0] - flat[:, 0]).max() > 1e-3  # x really sees a different position, mT
    print(f"axis slots ok, x differs from the snapshot by up to {np.abs(skew[:, 0] - flat[:, 0]).max():.3f} mT")


def check_clock() -> None:
    """fixed latency: no drops when faster than the period, dropped samples and no early starts when slower"""
    sensors = sensor_grid(6, 6, 18, 18)
    traj = RollingTrajectory(speed_mm_s=10.0)
    timing = CycleTiming()
    p = timing.period_s
    fast = track(traj, sensors, HoldSolver(), duration_s=40 * p, solve_time_s=0.5 * p, rng=np.random.default_rng(0))
    assert fast.dropped.sum() == 0 and len(fast.ok) == 40
    assert np.allclose(fast.t_ready - fast.t_start, 0.5 * p)
    assert np.allclose(fast.t_avail - fast.t_sample, timing.conv_s * (0.5 + timing.trg_margin) + timing.readout_s)
    assert np.all(fast.t_start >= fast.t_avail - 1e-12)

    slow = track(traj, sensors, HoldSolver(), duration_s=40 * p, solve_time_s=2.5 * p, rng=np.random.default_rng(0))
    assert np.all(slow.t_start >= slow.t_avail - 1e-12)  # never solve before the readout is done
    assert np.allclose(slow.t_ready - slow.t_start, 2.5 * p)
    assert np.all(slow.t_start[1:] >= slow.t_ready[:-1] - 1e-12)  # one solve at a time
    assert (1 + slow.dropped).sum() >= slow.n_samples  # every sample is solved or dropped
    assert slow.dropped[1:].min() >= 1  # 2.5 periods per solve skips at least one sample
    assert np.all(slow.t_avail[1:] <= slow.t_start[1:] + 1e-12)
    print(f"clock ok, fast solves {len(fast.ok)}, slow solves {len(slow.ok)} dropping {slow.dropped.sum()}")


def check_static_accuracy() -> None:
    """crawl speed + zero latency, error should sit near the CRB"""
    sensors = sensor_grid(6, 6, 18, 18)
    traj = RollingTrajectory(speed_mm_s=1.0)
    res = track(traj, sensors, LeastSquaresSolver(), duration_s=40 * CycleTiming().period_s, solve_time_s=0.0,
                rng=np.random.default_rng(1))
    s = res.summary()
    crb = np.sqrt(np.mean([crb_position_rmse(t, sensors, DEFAULT_NOISE, cfg.MAGNET_KWARGS) ** 2 for t in res.truth]))
    ratio = s["rmse_mm"] / crb
    assert s["lost_frac"] == 0.0, s
    assert 0.6 < ratio < 1.6, (s["rmse_mm"], crb)
    print(f"static accuracy ok, rmse {s['rmse_mm']:.4f} mm, crb {crb:.4f} mm, ratio {ratio:.2f}")


if __name__ == "__main__":
    check_geometry()
    check_trajectory()
    check_timing()
    check_axis_slots()
    check_clock()
    check_static_accuracy()
    print("all tracking checks passed")
