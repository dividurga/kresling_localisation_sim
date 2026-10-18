"""what the 5ms fall time implies for noise and position error, as a plot and a markdown table.
per conversion time: datasheet sigma, settle wait + off time + sense rate at the fall time (default 5ms, the worst case),
and CRB position error (per axis, mean and worst over the test positions) for one array (default 6x6 @ 18mm).
CRB only, MC tracked it within a few % on the shortlist

uv run python scripts/noise_implications.py [--fall 5e-3] [--current 20] [--count 6 --pitch 18]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from conversion_time_experiment import DATASHEET, DATASHEET_TABLE, residual_curves, settle_time
from kresling_sim import fisher, kresling_2026_supp_config as cfg
from kresling_sim.config import WORKSPACE_X_MM, WORKSPACE_Y_MM
from kresling_sim.noise import GaussianNoise
from kresling_sim.sensors import sensor_grid

INK, MUTED, GRID = "#1f2933", "#6b7785", "#e3e7eb"
BLUE, BLUE_LIGHT, ORANGE = "#184f95", "#6da7ec", "#c2410c"
T_ACT_MS = 10.0   # actuation window for the sense rate column


def crb_axes(sensors, noise, positions) -> np.ndarray:
    """(n_positions, 3) per axis CRB stdev in mm"""
    out = []
    for p in positions:
        jac = fisher.position_jacobian(p, sensors, magnet_kwargs=cfg.MAGNET_KWARGS)
        out.append(fisher.crlb(fisher.fisher_information_matrix(jac, noise)))
    return np.array(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fall", type=float, default=5e-3, help="coil fall time 90-10, s (default 5e-3)")
    p.add_argument("--current", type=float, default=20.0)
    p.add_argument("--count", type=int, default=6)
    p.add_argument("--pitch", type=float, default=18.0)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    xs = np.linspace(-WORKSPACE_X_MM / 2, WORKSPACE_X_MM / 2, 5)
    ys = np.linspace(-WORKSPACE_Y_MM / 2, WORKSPACE_Y_MM / 2, 5)
    pts = np.array([[x, y, z] for z in (0.0, cfg.Z_MAX_MM) for x in xs for y in ys])
    ts = np.arange(0.0, 0.2, 2e-4)
    curves = residual_curves(args.current, args.fall, pts, ts)

    sensors = sensor_grid(args.count, args.count, args.pitch, args.pitch)
    zs = np.linspace(cfg.Z_MIN_MM, cfg.Z_MAX_MM, 4)
    qs = np.linspace(0, WORKSPACE_X_MM / 2, 3)
    positions = np.array([[x, y, z] for z in zs for x in qs for y in qs])

    rows = []
    for (conv, sxy, sz), (_, xl, xh, zl, zh) in zip(DATASHEET, DATASHEET_TABLE):
        sigma = np.array([sxy, sxy, sz]) * 1e-3
        b = crb_axes(sensors, GaussianNoise(sigma), positions)
        wait = settle_time(curves, ts, sigma, conv * 1e-3, averaged=False) * 1e3
        off = wait + conv
        rms = np.sqrt((b**2).sum(axis=1))
        rows.append(dict(conv=conv, sxy=sxy, sz=sz, xl=xl, xh=xh, zl=zl, zh=zh, wait=wait, off=off,
                         rate=1000 / (T_ACT_MS + off), cx=b[:, 0].mean(), cy=b[:, 1].mean(), cz=b[:, 2].mean(),
                         mean=rms.mean(), worst=rms.max()))

    print(f"fall {args.fall * 1e3:g} ms, {args.current:g}A, {args.count}x{args.count} @ {args.pitch:g}mm, kresling_2026_supp, CRB over {len(positions)} positions\n")
    print("| conv (ms) | sigma_xy (uT) | sigma_z (uT) | settle wait (ms) | off time (ms) | rate @10ms act (Hz) | CRB x / y / z (mm) | CRB mean (mm) | CRB worst (mm) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['conv']:g} | {r['sxy']:.2f} | {r['sz']:.2f} | {r['wait']:.1f} | {r['off']:.1f} | {r['rate']:.1f} | "
              f"{r['cx']:.3f} / {r['cy']:.3f} / {r['cz']:.3f} | {r['mean']:.3f} | {r['worst']:.3f} |")

    c = np.array([r["conv"] for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 9), sharex=True)
    ax = axes[0]
    ax.fill_between(c, [r["xl"] for r in rows], [r["xh"] for r in rows], color=BLUE_LIGHT, alpha=0.35, lw=0)
    ax.fill_between(c, [r["zl"] for r in rows], [r["zh"] for r in rows], color=ORANGE, alpha=0.2, lw=0)
    ax.plot(c, [r["sxy"] for r in rows], "o-", color=BLUE, label="sigma xy")
    ax.plot(c, [r["sz"] for r in rows], "o-", color=ORANGE, label="sigma z")
    ax.set_ylabel("noise rms (uT)")
    ax.set_title(f"noise at a {args.fall * 1e3:g} ms coil fall ({args.count}x{args.count} @ {args.pitch:g}mm)  band = datasheet range", loc="left", fontsize=10, color=INK)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(c, [r["worst"] for r in rows], "o--", color=BLUE_LIGHT, label="worst position")
    ax.plot(c, [r["mean"] for r in rows], "o-", color=BLUE, label="mean")
    ax.set_ylabel("CRB position error (mm)")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot(c, [r["off"] for r in rows], "o-", color=BLUE, label="coils-off time (wait + conversion)")
    ax.plot(c, [r["wait"] for r in rows], "o--", color=BLUE_LIGHT, label="settle wait")
    for r in rows:
        ax.annotate(f"{r['rate']:.0f} Hz", (r["conv"], r["off"]), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7, color=ORANGE)
    ax.set_ylabel("ms (labels = sense rate, 10 ms actuation)")
    ax.set_xlabel("MLX90393 conversion time (ms)")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for a, ticks in zip(axes, ([0.2, 0.5, 1, 2, 5], [0.01, 0.02, 0.05, 0.1, 0.2, 0.5], [20, 30, 50, 100, 200])):
        a.set_yticks(ticks)
        a.set_xscale("log")
        a.set_xticks([c_ for c_ in (1.3, 2, 3, 5, 8, 14, 25, 50, 100, 175)])
        a.set_xticklabels(["1.3", "2", "3", "5", "8", "14", "25", "50", "100", "175"])
        a.minorticks_off()
        a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        a.grid(True, color=GRID)
        a.tick_params(colors=MUTED)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    out = args.output or os.path.join(os.path.dirname(__file__), "noise_implications.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
