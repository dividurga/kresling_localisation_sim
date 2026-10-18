"""fall time analysis, how the coil fall time sets the settle wait, cycle time and sensing rate.
reuses conversion_time_experiment.py (same residual < sigma/3 criterion, same datasheet noise table).
fall time is the 90->10% time of the coil current after switch off, tau = t_fall/ln9, simulated with Actuation.
CRB is for one sensor array (default 6x6 @ 18mm) and only depends on conversion time, not fall time

uv run python scripts/fall_time_analysis.py [--current 20] [--count 6 --pitch 18]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from conversion_time_experiment import DATASHEET, T_ACT_MS, mean_crb, residual_curves, settle_time
from kresling_sim import kresling_2026_supp_config as cfg
from kresling_sim.config import WORKSPACE_X_MM, WORKSPACE_Y_MM
from kresling_sim.noise import GaussianNoise
from kresling_sim.sensors import sensor_grid

FALL_MS = [1.5, 2.5, 5.0, 7.5, 10.0, 15.0]   # 1.5 = IEEE APEC measured step, 5 = controller design target
SHOW_CONV = [2.0, 5.0, 14.0, 50.0]            # conversion times to plot lines for


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--current", type=float, default=20.0)
    p.add_argument("--count", type=int, default=6)
    p.add_argument("--pitch", type=float, default=18.0)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    xs = np.linspace(-WORKSPACE_X_MM / 2, WORKSPACE_X_MM / 2, 5)
    ys = np.linspace(-WORKSPACE_Y_MM / 2, WORKSPACE_Y_MM / 2, 5)
    points = np.array([[x, y, z] for z in (0.0, cfg.Z_MAX_MM) for x in xs for y in ys])
    ts = np.arange(0.0, 0.2, 2e-4)

    sensors = sensor_grid(args.count, args.count, args.pitch, args.pitch)
    zs = np.linspace(cfg.Z_MIN_MM, cfg.Z_MAX_MM, 4)
    qs = np.linspace(0, WORKSPACE_X_MM / 2, 3)
    positions = np.array([[x, y, z] for z in zs for x in qs for y in qs])
    crb = {c: mean_crb(sensors, GaussianNoise(np.array([sxy, sxy, sz]) * 1e-3), positions) for c, sxy, sz in DATASHEET}

    wait = {}   # (conv, fall) -> ms
    for tf in FALL_MS:
        curves = residual_curves(args.current, tf * 1e-3, points, ts)
        for conv, sxy, sz in DATASHEET:
            w = settle_time(curves, ts, np.array([sxy, sxy, sz]) * 1e-3, conv * 1e-3, averaged=False)
            wait[(conv, tf)] = w * 1e3
        print(f"fall {tf} ms done", flush=True)

    print(f"\n{args.count}x{args.count} @ {args.pitch:g}mm, {args.current:g}A, residual < sigma/3, kresling_2026_supp")
    print("\nsettle wait (ms), rows conversion time, cols fall time (90-10%)")
    print("conv(ms) CRB(mm) | " + " ".join(f"{tf:6.1f}" for tf in FALL_MS))
    for conv, _, _ in DATASHEET:
        print(f"{conv:7.1f} {crb[conv]:8.4f} | " + " ".join(f"{wait[(conv, tf)]:6.1f}" for tf in FALL_MS))

    for t_act in T_ACT_MS[:2]:
        print(f"\nsense rate (Hz) with a {t_act:g} ms actuation window, 1/(t_act + wait + conv)")
        print("conv(ms) | " + " ".join(f"{tf:6.1f}" for tf in FALL_MS))
        for conv, _, _ in DATASHEET:
            print(f"{conv:7.1f} | " + " ".join(f"{1000 / (t_act + wait[(conv, tf)] + conv):6.1f}" for tf in FALL_MS))

    ratio = np.mean([wait[(c, tf)] / tf for c, _, _ in DATASHEET for tf in FALL_MS])
    print(f"\nwait is ~{ratio:.1f} x fall time (mean over the grid), so the fall time is the lever")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ramp = plt.cm.Blues(np.linspace(0.45, 0.95, len(SHOW_CONV)))
    for conv, color in zip(SHOW_CONV, ramp):
        ax.plot(FALL_MS, [wait[(conv, tf)] + conv for tf in FALL_MS], "o-", color=color, label=f"conv {conv:g} ms (CRB {crb[conv]:.3f} mm)")
    ax.set_xlabel("coil fall time, 90-10% (ms)", color="#6b7785")
    ax.set_ylabel("total coils-off time = wait + conversion (ms)", color="#6b7785")
    ax.set_title(f"fall time sets the sensing window ({args.current:g}A, {args.count}x{args.count} @ {args.pitch:g}mm)", loc="left", fontsize=11)
    ax.axvline(5.0, color="#6b7785", ls=":", lw=1)
    ax.text(5.1, ax.get_ylim()[0] + 2, "5 ms design", fontsize=8, color="#6b7785", va="bottom")
    ax.grid(True, color="#e3e7eb")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    out = args.output or os.path.join(os.path.dirname(__file__), "fall_time_analysis.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
