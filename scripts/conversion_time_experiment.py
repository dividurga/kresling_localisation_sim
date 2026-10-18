"""pick the MLX90393 conversion time and work out the actuate/sense cycle it allows.
per conversion time: settle wait after coils off until the residual coil field is < sigma/3 (simulated with Actuation, 5ms rise),
CRB position error for that sigma, then total off time and the cycle frequency for a few actuation windows.
sigma per row is the midpoint of the datasheet range in DATASHEET_TABLE

uv run python scripts/conversion_time_experiment.py [--current 20] [--count 6 --pitch 18]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kresling_sim import fisher, kresling_2026_supp_config as cfg
from kresling_sim.actuation import Actuation
from kresling_sim.coils import Coils
from kresling_sim.config import WORKSPACE_X_MM, WORKSPACE_Y_MM
from kresling_sim.noise import GaussianNoise
from kresling_sim.sensors import sensor_grid

# MLX90393 noise vs conversion time, GAIN_SEL=0x7, datasheet rev 012 figs 2 and 3 (read off the chart, ~+-0.1uT)
# (conversion ms, xy_lo, xy_hi, z_lo, z_hi) in uT, same numbers as hardware_notes.md
DATASHEET_TABLE = [
    (1.3,   5.1,  5.1,  7.6,  7.6),
    (2.0,   3.1,  3.5,  4.6,  5.2),
    (3.0,   2.3,  2.6,  3.5,  4.5),
    (5.0,   1.4,  1.8,  2.4,  3.0),
    (8.0,   1.1,  1.2,  1.7,  2.3),
    (14.0,  0.8,  0.8,  1.2,  1.7),
    (25.0,  0.6,  0.6,  0.9,  1.2),
    (50.0,  0.45, 0.45, 0.7,  0.9),
    (100.0, 0.3,  0.3,  0.5,  0.6),
    (175.0, 0.2,  0.2,  0.4,  0.4),   # chart says ~150-200ms
]
# sigma used per row is the midpoint of the range
DATASHEET = [(c, (xl + xh) / 2, (zl + zh) / 2) for c, xl, xh, zl, zh in DATASHEET_TABLE]
SETTLE_FRAC = 1 / 3         # residual must be < sigma * this
T_ACT_MS = [10.0, 20.0, 50.0]   # actuation windows to report frequencies for
AXIS_CH = {"x": ("i_xf", "i_xb"), "y": ("i_yl", "i_yr"), "z": ("i_zt", "i_zb")}


def residual_curves(current: float, rise_time: float, points: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """max over points of |B_axis| (mT) after coils go off at t=0, driving each axis alone. returns (3, len(ts))"""
    out = np.zeros((3, len(ts)))
    t_off = 0.1  # long enough to be fully settled before switching off
    for a, axis in enumerate("xyz"):
        act = Actuation(None, rise_time_s=rise_time)
        act.command({ch: current for ch in AXIS_CH[axis]}, t=0.0)
        act.command({ch: 0.0 for ch in AXIS_CH[axis]}, t=t_off)
        for k, t in enumerate(ts):
            b = np.asarray(Coils(**act.currents_at(t_off + t)).getB(points))
            out[a, k] = np.abs(b[:, a]).max()
    return out


def settle_time(curves: np.ndarray, ts: np.ndarray, sigma_mt: np.ndarray, conv_s: float, averaged: bool) -> float:
    """first wait (s) where every axis residual < sigma*SETTLE_FRAC. averaged=True uses the mean over the conversion window"""
    dt = ts[1] - ts[0]
    n_conv = max(1, int(round(conv_s / dt)))
    for k in range(len(ts)):
        ok = True
        for a in range(3):
            seg = curves[a, k:k + n_conv] if averaged else curves[a, k:k + 1]
            if len(seg) < n_conv and averaged:
                ok = False
                break
            if seg.mean() >= sigma_mt[a] * SETTLE_FRAC:
                ok = False
                break
        if ok:
            return ts[k]
    return np.nan


def mean_crb(sensors, noise, positions) -> float:
    r = []
    for p in positions:
        jac = fisher.position_jacobian(p, sensors, magnet_kwargs=cfg.MAGNET_KWARGS)
        bound = fisher.crlb(fisher.fisher_information_matrix(jac, noise))
        r.append(float(np.sqrt(np.sum(bound**2))))
    return float(np.mean(r))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--current", type=float, default=20.0, help="coil current per side, A (default 20)")
    p.add_argument("--rise-time", type=float, default=5e-3, help="10-90 rise time, s (default 5e-3)")
    p.add_argument("--count", type=int, default=6, help="sensor grid side for the CRB (default 6)")
    p.add_argument("--pitch", type=float, default=18.0, help="sensor pitch mm for the CRB (default 18)")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    # coil field at sensor plane points over the workspace, plus the top of the z band
    xs = np.linspace(-WORKSPACE_X_MM / 2, WORKSPACE_X_MM / 2, 5)
    ys = np.linspace(-WORKSPACE_Y_MM / 2, WORKSPACE_Y_MM / 2, 5)
    points = np.array([[x, y, z] for z in (0.0, cfg.Z_MAX_MM) for x in xs for y in ys])
    ts = np.arange(0.0, 0.3, 2e-4)
    curves = residual_curves(args.current, args.rise_time, points, ts)
    print("datasheet noise (uT)  conv(ms)  xy range    z range")
    for c, xl, xh, zl, zh in DATASHEET_TABLE:
        print(f"                      {c:7.1f}  {xl:4.2f}-{xh:4.2f}   {zl:4.2f}-{zh:4.2f}")
    print(f"peak coil field at sensors ({args.current}A): x={curves[0,0]:.2f} y={curves[1,0]:.2f} z={curves[2,0]:.2f} mT")

    sensors = sensor_grid(args.count, args.count, args.pitch, args.pitch)
    zs = np.linspace(cfg.Z_MIN_MM, cfg.Z_MAX_MM, 4)
    qs = np.linspace(0, WORKSPACE_X_MM / 2, 3)
    positions = np.array([[x, y, z] for z in zs for x in qs for y in qs])

    rows = []
    for conv_ms, sxy, sz in DATASHEET:
        sigma = np.array([sxy, sxy, sz]) * 1e-3  # mT
        crb = mean_crb(sensors, GaussianNoise(sigma), positions)
        wait_end = settle_time(curves, ts, sigma, conv_ms * 1e-3, averaged=False)
        wait_avg = settle_time(curves, ts, sigma, conv_ms * 1e-3, averaged=True)
        rows.append((conv_ms, sxy, sz, crb, wait_end * 1e3, wait_avg * 1e3))

    print(f"\nCRB: {args.count}x{args.count} @ {args.pitch:g}mm, kresling_2026_supp, mean over {len(positions)} positions")
    print("conv(ms) sxy(uT) sz(uT) CRB(mm) | wait_end(ms) off_end(ms) | wait_avg(ms) off_avg(ms) | f_sense(Hz) for T_act = " + ", ".join(f"{t:g}ms" for t in T_ACT_MS))
    for conv, sxy, sz, crb, w_end, w_avg in rows:
        off_end, off_avg = w_end + conv, w_avg + conv
        freqs = "  ".join(f"{1000 / (t + off_end):6.1f}" for t in T_ACT_MS)
        print(f"{conv:7.1f} {sxy:7.2f} {sz:6.2f} {crb:8.4f} | {w_end:11.1f} {off_end:11.1f} | {w_avg:11.1f} {off_avg:11.1f} | {freqs}")

    fig, ax = plt.subplots(figsize=(7, 5))
    off = [r[4] + r[0] for r in rows]
    ax.plot(off, [r[3] for r in rows], "o-")
    for r, o in zip(rows, off):
        ax.annotate(f"{r[0]:g}ms", (o, r[3]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("total coils-off time = settle wait + conversion (ms)")
    ax.set_ylabel("CRB position error (mm)")
    ax.set_title(f"conversion time tradeoff ({args.current:g}A, t_r={args.rise_time * 1e3:g}ms, {args.count}x{args.count} @ {args.pitch:g}mm)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = args.output or os.path.join(os.path.dirname(__file__), "conversion_time_experiment.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved plot to {out}")


if __name__ == "__main__":
    main()
