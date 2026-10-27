"""closed loop tracking of the rolling hex prism (kresling_2026_supp).
virtual clock, one sense cycle per sample (CycleTiming). solver latency is real wall clock by default
so a slow solver means a stale estimate. when the solver frees up it takes the latest sample whose
readout has finished, older ones are dropped. warm start is the last finished estimate (optionally
extrapolated by its velocity)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from kresling_sim import forward, kresling_2026_supp_config as cfg
from kresling_sim.config import SENSOR_PCB_GAP_MM
from kresling_sim.inverse import PositionSolver, SolverResult
from kresling_sim.kresling import Kresling
from kresling_sim.montecarlo import DEFAULT_NOISE
from kresling_sim.noise import NoiseModel
from kresling_sim.sensors import Sensor

HEX_SIDE_MM = 2 * cfg.HEX_PLATE_INSCRIBED_RADIUS_MM / math.sqrt(3)  # regular hex, circumradius = side
FACE_ANGLE = math.pi / 3

# tldr.md 14ms row (5ms fall, 20A): coils on 10ms, then wait until the coil field is < sigma/3, then convert
ACT_WINDOW_S = 10e-3
SETTLE_WAIT_S = 24.8e-3
CONV_S = 14e-3
TRG_MARGIN = 0.05  # shared TRG pin means no DRDY, mcu waits a fixed time. +5% is the datasheet oscillator trim

# i2c RM read, MLX90393 rev 012 sec 15.4. addr+W, cmd, addr+R, status, 6 data bytes (xyz), 9 clocks a byte
I2C_BYTES_XYZ = 10
# start hold + repeated start setup + stop setup + bus free, table 15 (fast mode / standard mode)
I2C_OVERHEAD_S = {"fast": (0.6 + 0.6 + 0.6 + 1.3) * 1e-6, "standard": (4.0 + 4.7 + 4.0 + 4.7) * 1e-6}


def readout_time_s(n_sensors: int = 36, n_buses: int = 3, clk_hz: float = 400e3, n_bytes: int = I2C_BYTES_XYZ) -> float:
    """time to read every sensor over i2c. buses run in parallel, sensors on one bus one after another.
    no mcu or driver overhead, no mux time, so a lower bound"""
    per_bus = math.ceil(n_sensors / n_buses)
    overhead = I2C_OVERHEAD_S["fast" if clk_hz > 100e3 else "standard"]
    return per_bus * (n_bytes * 9 / clk_hz + overhead)


@dataclass(frozen=True)
class CycleTiming:
    """one sense cycle k starts at k * period_s:
    coils on (act_s) -> coils off, settle (wait_s) -> shared TRG, conversion (conv_s, x then y then z, one third each)
    -> mcu waits conv_s * (1 + trg_margin) -> coils back on, i2c readout (readout_s) overlaps the coils on window.
    sample k is timestamped at the middle of its conversion"""

    act_s: float = ACT_WINDOW_S
    wait_s: float = SETTLE_WAIT_S
    conv_s: float = CONV_S
    trg_margin: float = TRG_MARGIN
    readout_s: float = readout_time_s()

    @property
    def period_s(self) -> float:
        return self.act_s + self.wait_s + self.conv_s * (1 + self.trg_margin)

    def t_off(self, k: int) -> float:
        """coils commanded off"""
        return k * self.period_s + self.act_s

    def t_conv_start(self, k: int) -> float:
        return self.t_off(k) + self.wait_s

    def t_sample(self, k: int) -> float:
        """timestamp of sample k, middle of the conversion"""
        return self.t_conv_start(k) + self.conv_s / 2

    def t_avail(self, k: int) -> float:
        """sample k is readable by the solver, mcu wait over and readout done"""
        return self.t_conv_start(k) + self.conv_s * (1 + self.trg_margin) + self.readout_s


def rolling_position(phi, x_start_mm: float = 0.0, y_mm: float = 0.0) -> np.ndarray:
    """magnet center (..., 3) mm for roll angle phi (rad, 0 = flat on a face, array ok).
    rolls about the lower edge so x advances one side per 60deg and z goes apothem -> circumradius -> apothem"""
    phi = np.asarray(phi, dtype=float)
    face = np.floor(phi / FACE_ANGLE)
    alpha = phi - face * FACE_ANGLE - FACE_ANGLE / 2  # center angle off vertical, measured from the pivot edge
    x = x_start_mm + face * HEX_SIDE_MM + HEX_SIDE_MM / 2 + HEX_SIDE_MM * np.sin(alpha)
    z = SENSOR_PCB_GAP_MM + HEX_SIDE_MM * np.cos(alpha)
    return np.stack([x, np.full_like(x, y_mm), z], axis=-1)


class RollingTrajectory:
    """back and forth roll along x, axis stays +y. speed_mm_s is the mean x speed, starts flat at -x end"""

    def __init__(self, speed_mm_s: float, span_mm: float = 100.0, y_mm: float = 0.0):
        if speed_mm_s <= 0:
            raise ValueError(f"speed must be > 0, got {speed_mm_s}")
        self.speed_mm_s = speed_mm_s
        self.y_mm = y_mm
        self.n_faces = int(span_mm // HEX_SIDE_MM)  # whole faces so both ends rest flat
        self.x_start_mm = -self.n_faces * HEX_SIDE_MM / 2
        self.phi_max = self.n_faces * FACE_ANGLE
        self.omega = speed_mm_s / HEX_SIDE_MM * FACE_ANGLE  # rad/s
        self.pass_time_s = self.phi_max / self.omega

    def phi(self, t) -> np.ndarray:
        """roll angle (rad) at time t (s, array ok), clockwise seen from -y while rolling toward +x"""
        u = (self.omega * np.asarray(t, dtype=float)) % (2 * self.phi_max)
        return np.where(u <= self.phi_max, u, 2 * self.phi_max - u)

    def position(self, t) -> np.ndarray:
        """magnet center (..., 3) mm at time t (s, array ok)"""
        return rolling_position(self.phi(t), self.x_start_mm, self.y_mm)


def measure_sample(
    trajectory: RollingTrajectory,
    t_sample: float,
    timing: CycleTiming,
    sensors: list[Sensor],
    noise: NoiseModel,
    magnet_kwargs: dict,
    rng: np.random.Generator,
    per_axis: bool = True,
) -> np.ndarray:
    """one noisy snapshot (N,3) mT of the moving robot. per_axis: each axis is converted in its own third of
    the conversion (x, y, z) so it sees the robot at that slot's midpoint (t - conv/3, t, t + conv/3).
    otherwise every axis sees the robot at t_sample. slot midpoint stands in for the average over the slot"""
    offsets = (-1, 0, 1) if per_axis else (0, 0, 0)
    clean = np.empty((len(sensors), 3))
    for axis, off in enumerate(offsets):
        position = trajectory.position(t_sample + off * timing.conv_s / 3)
        clean[:, axis] = forward.stacked_forward_prediction(Kresling(position=position, **magnet_kwargs), sensors)[:, axis]
    return clean + noise.sample(n_sensors=len(sensors), rng=rng)


@dataclass
class TrackResult:
    """one row per solve. all (n,) unless noted, times in s, errors in mm"""

    timing: CycleTiming
    n_samples: int  # samples that arrived, solved or not
    t_sample: np.ndarray  # conversion midpoint
    t_avail: np.ndarray  # readout done
    t_start: np.ndarray  # solver starts
    t_ready: np.ndarray  # estimate done
    solve_ms: np.ndarray
    nfev: np.ndarray
    cost_ratio: np.ndarray  # nan on a start failure
    ok: np.ndarray
    dropped: np.ndarray  # samples skipped just before this solve
    est: np.ndarray  # (n,3) nan when not ok
    truth: np.ndarray  # (n,3) at t_sample
    truth_ready: np.ndarray  # (n,3) at t_ready
    x0_err: np.ndarray  # warm start vs truth at t_sample
    err: np.ndarray  # est vs truth at t_sample
    err_ready: np.ndarray  # est vs truth at t_ready, what a controller would see
    lost: np.ndarray  # not ok or err > lock_mm

    @property
    def period_s(self) -> float:
        return self.timing.period_s

    def summary(self) -> dict:
        good = ~self.lost
        rms = lambda e: float(np.sqrt(np.mean(e[good] ** 2))) if good.any() else float("nan")
        lost_idx = np.flatnonzero(self.lost)
        return dict(
            n_samples=self.n_samples,
            n_solves=len(self.ok),
            drop_frac=float(self.dropped.sum() / self.n_samples),
            lost_frac=float(self.lost.mean()),
            first_lost_s=float(self.t_sample[lost_idx[0]]) if len(lost_idx) else float("nan"),
            rmse_mm=rms(self.err),
            rmse_ready_mm=rms(self.err_ready),
            x0_err_mm=float(np.sqrt(np.mean(self.x0_err**2))),
            age_ms=float(np.mean(self.t_ready - self.t_sample) * 1e3),
            solve_ms_median=float(np.median(self.solve_ms)),
            solve_ms_p95=float(np.percentile(self.solve_ms, 95)),
            solve_ms_max=float(self.solve_ms.max()),
            nfev_mean=float(np.mean(self.nfev)),
        )


def track(
    trajectory: RollingTrajectory,
    sensors: list[Sensor],
    solver: PositionSolver,
    duration_s: float,
    noise: NoiseModel = DEFAULT_NOISE,
    timing: CycleTiming | None = None,
    magnet_kwargs: dict | None = None,
    predict: str = "last",
    solve_time_s: float | None = None,
    per_axis: bool = True,
    lock_mm: float = 2.0,
    init_guess: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> TrackResult:
    """track the trajectory for duration_s.
    predict: "last" warm starts at the last estimate, "velocity" extrapolates it with its finite difference velocity.
    solve_time_s: fixed solver latency, None = measured wall clock.
    per_axis: x, y, z converted one after another (see measure_sample).
    lock_mm: a solve counts as lost if it fails or lands further than this from the truth at t_sample.
    init_guess: first warm start, default the true start (acquisition assumed done)"""
    if predict not in ("last", "velocity"):
        raise ValueError(f"predict must be 'last' or 'velocity', got {predict!r}")
    timing = timing if timing is not None else CycleTiming()
    magnet_kwargs = cfg.MAGNET_KWARGS if magnet_kwargs is None else magnet_kwargs
    rng = rng if rng is not None else np.random.default_rng()

    n_samples = int(duration_s / timing.period_s)
    est = np.asarray(trajectory.position(0.0) if init_guess is None else init_guess, dtype=float)
    est_t = 0.0  # timestamp of est
    vel = np.zeros(3)  # mm/s
    t_free = 0.0  # solver idle from here
    k = 0
    rows = []
    while k < n_samples:
        t_k = timing.t_sample(k)
        t_avail = timing.t_avail(k)
        truth = trajectory.position(t_k)
        measurement = measure_sample(trajectory, t_k, timing, sensors, noise, magnet_kwargs, rng, per_axis)
        x0 = est + vel * (t_k - est_t) if predict == "velocity" else est

        t_start = max(t_free, t_avail)
        wall = time.perf_counter()
        result = solver.solve(
            sensors=sensors,
            sensor_measurements=measurement,
            noise=noise,
            initial_guess=x0,
            magnet_kwargs=magnet_kwargs,
            rng=rng,
        )
        solve_s = time.perf_counter() - wall if solve_time_s is None else solve_time_s
        t_ready = t_start + solve_s
        ok = result.status == SolverResult.SUCCESS

        if ok:
            new = np.asarray(result.position_estimate, dtype=float)
            vel = (new - est) / (t_k - est_t) if t_k > est_t else vel
            est, est_t = new, t_k
        truth_ready = trajectory.position(t_ready)
        err = float(np.linalg.norm(result.position_estimate - truth)) if ok else float("nan")
        err_ready = float(np.linalg.norm(result.position_estimate - truth_ready)) if ok else float("nan")

        # latest sample whose readout is done by t_ready, t_avail(j) = (j + 1) * period + readout
        k_next = max(k + 1, int((t_ready - timing.readout_s) // timing.period_s) - 1)
        rows.append((t_k, t_avail, t_start, t_ready, solve_s * 1e3, result.nfev or 0,
                     result.cost_ratio if result.cost_ratio is not None else float("nan"), ok, k_next - k - 1,
                     result.position_estimate if ok else np.full(3, np.nan), truth, truth_ready,
                     float(np.linalg.norm(x0 - truth)), err, err_ready, (not ok) or err > lock_mm))
        t_free = t_ready
        k = k_next

    cols = list(zip(*rows))
    arr = lambda i, dtype=float: np.array(cols[i], dtype=dtype)
    return TrackResult(
        timing=timing, n_samples=n_samples,
        t_sample=arr(0), t_avail=arr(1), t_start=arr(2), t_ready=arr(3), solve_ms=arr(4), nfev=arr(5, int),
        cost_ratio=arr(6), ok=arr(7, bool), dropped=arr(8, int),
        est=np.array(cols[9]), truth=np.array(cols[10]), truth_ready=np.array(cols[11]),
        x0_err=arr(12), err=arr(13), err_ready=arr(14), lost=arr(15, bool),
    )
