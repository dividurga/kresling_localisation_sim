"""plot on-axis |Bz| vs distance (log-log) against a 1/d^3 line. far field should be dipole like,
if not parallel something is off in the forward model or units.
defaults are the workspace z range, sensor gap to deployed height

Examples:
  uv run python scripts/field_falloff_check.py
  uv run python scripts/field_falloff_check.py --d-min 5 --d-max 100 --n-points 20
  uv run python scripts/field_falloff_check.py --diameter 4 --height 1 --remanence 1480
  uv run python scripts/field_falloff_check.py --output /tmp/falloff.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kresling_sim.config import MAGNET_BR_MT, MAGNET_DIAMETER_MM, MAGNET_HEIGHT_MM, SENSOR_PCB_GAP_MM, Z_DEPLOYED_MM
from kresling_sim.kresling import Kresling

BR_N50_MT = MAGNET_BR_MT  # alias for the argparse default below


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d-min", type=float, default=SENSOR_PCB_GAP_MM, help=f"nearest on-axis distance to plot, mm (default: {SENSOR_PCB_GAP_MM}, sensor-plane gap)")
    p.add_argument("--d-max", type=float, default=Z_DEPLOYED_MM, help=f"farthest on-axis distance to plot, mm (default: {Z_DEPLOYED_MM}, h1 deployed cell height)")
    p.add_argument("--n-points", type=int, default=12, help="number of log-spaced distances to sample (default: 12)")
    p.add_argument("--diameter", type=float, default=MAGNET_DIAMETER_MM, help=f"magnet diameter, mm (default: {MAGNET_DIAMETER_MM}, N50 Supermagnetman D1002-10)")
    p.add_argument("--height", type=float, default=MAGNET_HEIGHT_MM, help=f"magnet height/thickness, mm (default: {MAGNET_HEIGHT_MM})")
    p.add_argument("--remanence", type=float, default=MAGNET_BR_MT, help=f"magnet remanence Br, mT (default: {MAGNET_BR_MT}, N50 datasheet midpoint)")
    p.add_argument("--mark", type=float, nargs="*", default=[SENSOR_PCB_GAP_MM, Z_DEPLOYED_MM], help=f"distances (mm) to mark with vertical reference lines (default: {SENSOR_PCB_GAP_MM} {Z_DEPLOYED_MM})")
    p.add_argument("--output", type=str, default=None, help="output PNG path (default: alongside this script)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    distances_mm = np.geomspace(args.d_min, args.d_max, args.n_points)

    k = Kresling(position=np.array([0, 0, 0]), diameter=args.diameter, height=args.height, remanence=args.remanence)
    points = np.column_stack([np.zeros_like(distances_mm), np.zeros_like(distances_mm), distances_mm])
    bz_mt = k.getB(points)[:, 2]

    print(f"magnet: diameter={args.diameter}mm, height={args.height}mm, remanence={args.remanence}mT")
    print(f"{'d (mm)':>8} {'Bz (mT)':>14} {'Bz (uT)':>14}")
    for d, b in zip(distances_mm, bz_mt):
        print(f"{d:8.2f} {b:14.6f} {b * 1000:14.3f}")

    # 1/d^3 reference line, anchored to match at the first distance point
    ref = bz_mt[0] * (distances_mm[0] / distances_mm) ** 3

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(distances_mm, np.abs(bz_mt), "o-", label="Bz (Kresling magnet, on-axis)")
    ax.loglog(distances_mm, ref, "k--", alpha=0.6, label="1/d^3 reference")
    for m in args.mark:
        if args.d_min <= m <= args.d_max:
            ax.axvline(m, color="gray", linestyle=":", alpha=0.5, label=f"{m:g}mm")
    ax.set_xlabel("distance from magnet center (mm)")
    ax.set_ylabel("|Bz| (mT)")
    ax.set_title("On-axis field falloff")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out_path = args.output or __file__.replace(".py", ".png")
    fig.savefig(out_path, dpi=150)
    print(f"\nsaved plot to {out_path}")


if __name__ == "__main__":
    main()
