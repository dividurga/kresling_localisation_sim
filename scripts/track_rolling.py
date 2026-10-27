"""track the rolling hex prism (kresling_2026_supp) with the real solver and wall clock latency.
sweeps robot speed, prints a table and saves track_sweep.png, track_detail.png, track_summary.csv

uv run python scripts/track_rolling.py [--speeds 5 10 20 40 80 160] [--duration 10] [--predict last velocity]
uv run python scripts/track_rolling.py --solver windowed --window 3 --n-starts 1
uv run python scripts/track_rolling.py --solve-ms 5     # fixed solver latency instead of wall clock
"""
from __future__ import annotations

import argparse
import csv
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kresling_sim.inverse import LeastSquaresSolver
from kresling_sim.sensors import sensor_grid
from kresling_sim.tracking import CycleTiming, RollingTrajectory, readout_time_s, track
from kresling_sim.windowing import WindowedSolver

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "tracking")
INK, MUTED, GRID = "#1f2933", "#6b7785", "#e3e7eb"
BLUE, ORANGE = "#184f95", "#c2410c"
COLORS = {"last": BLUE, "velocity": ORANGE}


def make_solver(kind: str, window: int, n_starts: int):
    inner = LeastSquaresSolver(n_starts=n_starts)
    return WindowedSolver(window, inner=inner) if kind == "windowed" else inner


def _style(ax) -> None:
    ax.grid(True, color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)


def plot_sweep(rows: list[dict], predicts: list[str], period_s: float, path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for pred in predicts:
        r = [x for x in rows if x["predict"] == pred]
        v = [x["speed"] for x in r]
        axes[0].plot(v, [100 * x["lost_frac"] for x in r], "o-", color=COLORS[pred], label=pred)
        axes[1].plot(v, [x["rmse_ready_mm"] for x in r], "o-", color=COLORS[pred], label=pred)
        axes[2].plot(v, [x["solve_ms_median"] for x in r], "o-", color=COLORS[pred], label=pred)
    axes[0].set_ylabel("lost solves (%)")
    axes[1].set_ylabel("rmse vs truth at estimate ready (mm), not lost")
    axes[1].set_yscale("log")
    axes[2].set_ylabel("median solve time (ms)")
    axes[2].axhline(period_s * 1e3, color=MUTED, ls="--", lw=1)
    axes[2].annotate("sample period", (rows[0]["speed"], period_s * 1e3), xytext=(0, 4),
                     textcoords="offset points", fontsize=8, color=MUTED)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("mean roll speed (mm/s)", color=MUTED)
        ax.legend(title="warm start", frameon=False, fontsize=8)
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_detail(res, title: str, lock_mm: float, path: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    t = res.t_sample
    axes[0].plot(t, res.truth[:, 0], color=MUTED, lw=1, label="truth x")
    axes[0].plot(t, res.est[:, 0], ".", color=BLUE, ms=3, label="estimate x")
    axes[0].plot(t[res.lost], res.truth[res.lost, 0], "x", color=ORANGE, ms=5, label="lost")
    axes[0].set_ylabel("x (mm)")
    axes[1].semilogy(t, np.where(res.ok, res.err, np.nan), ".-", color=BLUE, lw=0.6, ms=3, label="vs truth at sample")
    axes[1].semilogy(t, np.where(res.ok, res.err_ready, np.nan), ".-", color=ORANGE, lw=0.6, ms=3,
                     label="vs truth at ready")
    axes[1].axhline(lock_mm, color=MUTED, ls="--", lw=1)
    axes[1].set_ylabel("position error (mm)")
    axes[2].plot(t, res.solve_ms, ".-", color=BLUE, lw=0.6, ms=3)
    axes[2].axhline(res.period_s * 1e3, color=MUTED, ls="--", lw=1)
    axes[2].set_ylabel("solve time (ms)")
    axes[2].set_xlabel("time (s)", color=MUTED)
    for ax in axes:
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8, ncol=3)
    axes[1].legend(frameon=False, fontsize=8)
    axes[0].set_title(title, loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speeds", type=float, nargs="+", default=[5, 10, 20, 40, 80, 160], help="mean roll speed mm/s")
    p.add_argument("--duration", type=float, default=10.0, help="seconds per run (default 10)")
    p.add_argument("--solver", choices=["full", "windowed"], default="full")
    p.add_argument("--window", type=int, default=3, help="n x n window for --solver windowed")
    p.add_argument("--n-starts", type=int, default=1)
    p.add_argument("--predict", nargs="+", choices=["last", "velocity"], default=["last", "velocity"])
    p.add_argument("--solve-ms", type=float, default=None, help="fixed solver latency in ms, default wall clock")
    p.add_argument("--lock-mm", type=float, default=2.0, help="error above this counts as lost (default 2)")
    p.add_argument("--count", type=int, default=6)
    p.add_argument("--pitch", type=float, default=18.0)
    p.add_argument("--buses", type=int, default=3, help="i2c buses read in parallel (default 3)")
    p.add_argument("--i2c-khz", type=float, default=400.0, help="i2c clock (default 400, datasheet max)")
    p.add_argument("--readout-ms", type=float, default=None, help="override the i2c readout time, ms")
    p.add_argument("--margin", type=float, default=0.05, help="extra wait after conversion, fraction (default 0.05)")
    p.add_argument("--instant", action="store_true", help="every axis sampled at the conversion midpoint, no x/y/z slots")
    p.add_argument("--detail-speed", type=float, default=None, help="speed for track_detail.png, default the fastest")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default=DEFAULT_OUT)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sensors = sensor_grid(args.count, args.count, args.pitch, args.pitch)
    solver = make_solver(args.solver, args.window, args.n_starts)
    fixed = None if args.solve_ms is None else args.solve_ms * 1e-3
    lat = "wall clock" if fixed is None else f"{args.solve_ms:g} ms fixed"
    readout = (readout_time_s(args.count**2, args.buses, args.i2c_khz * 1e3) if args.readout_ms is None
               else args.readout_ms * 1e-3)
    timing = CycleTiming(trg_margin=args.margin, readout_s=readout)
    desc = (f"{args.count}x{args.count} @ {args.pitch:g}mm, {args.solver}"
            f"{f' n={args.window}' if args.solver == 'windowed' else ''}, n_starts={args.n_starts}, latency {lat}")
    print(f"{desc}\nperiod {timing.period_s * 1e3:.1f} ms, readout {readout * 1e3:.2f} ms, "
          f"{'instant snapshot' if args.instant else 'x/y/z slots'}, {args.duration:g}s per run, lost > {args.lock_mm:g} mm\n")

    detail_speed = args.detail_speed if args.detail_speed is not None else max(args.speeds)
    rows, details = [], {}
    head = f"{'speed':>6} {'warm':>8} {'solves':>6} {'drop%':>6} {'lost%':>6} {'first lost':>10} {'rmse':>7} {'rmse@rdy':>8} {'age ms':>7} {'x0 err':>7} {'ms med':>7} {'ms p95':>7} {'ms max':>7} {'nfev':>5}"
    print(head)
    for speed in args.speeds:
        for pred in args.predict:
            res = track(RollingTrajectory(speed), sensors, solver, args.duration, timing=timing, predict=pred,
                        solve_time_s=fixed, per_axis=not args.instant, lock_mm=args.lock_mm,
                        rng=np.random.default_rng(args.seed))
            s = res.summary()
            rows.append(dict(speed=speed, predict=pred, **s))
            details[(speed, pred)] = res
            print(f"{speed:6g} {pred:>8} {s['n_solves']:6d} {100 * s['drop_frac']:6.1f} {100 * s['lost_frac']:6.1f} "
                  f"{s['first_lost_s']:10.2f} {s['rmse_mm']:7.3f} {s['rmse_ready_mm']:8.3f} {s['age_ms']:7.1f} {s['x0_err_mm']:7.3f} "
                  f"{s['solve_ms_median']:7.1f} {s['solve_ms_p95']:7.1f} {s['solve_ms_max']:7.1f} {s['nfev_mean']:5.1f}",
                  flush=True)

    with open(os.path.join(args.outdir, "track_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    plot_sweep(rows, args.predict, timing.period_s, os.path.join(args.outdir, "track_sweep.png"))
    pred = args.predict[0]
    plot_detail(details[(detail_speed, pred)], f"{desc}\nroll {detail_speed:g} mm/s, warm start = {pred}",
                args.lock_mm, os.path.join(args.outdir, "track_detail.png"))
    print(f"\nplots + csv in {args.outdir}")


if __name__ == "__main__":
    main()
