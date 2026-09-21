"""simulate an actuate -> sense -> actuate cycle and plot the field a
sensor would actually see over time.

Defaults (20A, 1.5ms rise time) are real numbers from the IEEE APEC 2025
paper: tr=1.5ms is the demonstrated step response (Fig. 6), ~20A is what
drove the cube robot's rolling demo (Fig. 9). Drive electronics are rated
to 100A/submodule but that's never run continuously -- pass --current 100
to test that worst case separately.

Examples:
  uv run python scripts/actuation_cycle_check.py
  uv run python scripts/actuation_cycle_check.py --axis x --current 100 --magnet-z 5
  uv run python scripts/actuation_cycle_check.py --rise-time 5e-3 --t-sense 0.05 --t-resume 0.09
  uv run python scripts/actuation_cycle_check.py --sat-threshold 25 --output /tmp/cycle.png
"""
from __future__ import annotations

import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from kresling_sim.kresling import Kresling
from kresling_sim.actuation import Actuation

_AXIS_CHANNELS = {
    "x": ("i_xf", "i_xb"),
    "y": ("i_yl", "i_yr"),
    "z": ("i_zt", "i_zb"),
}

# magnetization is always along local z, so to probe x/y we rotate that z
# onto the axis being driven -- same convention coils.py uses for the coils
_AXIS_ORIENTATION = {
    "x": Rotation.from_euler("y", 90, degrees=True),
    "y": Rotation.from_euler("x", -90, degrees=True),
    "z": None,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--axis", choices=["x", "y", "z"], default="z", help="which coil axis to drive in uniform mode (default: z)")
    p.add_argument("--current", type=float, default=20.0, help="current per side, amps (default: 20.0, matches the paper's actual cube-rolling demo; try --current 100 for the drive's rated max)")
    p.add_argument("--magnet-z", type=float, default=15.0, help="magnet position on the drive axis, mm from the sensor point (default: 15.0, mid-workspace)")
    p.add_argument("--rise-time", type=float, default=1.5e-3, help="10-90%% current rise time, seconds (default: 1.5e-3, matches the paper)")
    p.add_argument("--sat-threshold", type=float, default=50.0, help="sensor saturation onset to plot as a reference line, mT (default: 50.0, standard-IMC MLX90393)")
    p.add_argument("--t-actuate", type=float, default=0.0, help="time actuation starts, s (default: 0.0)")
    p.add_argument("--t-sense", type=float, default=0.020, help="time the sense window starts (coils commanded off), s (default: 0.020)")
    p.add_argument("--t-resume", type=float, default=0.030, help="time actuation resumes, s (default: 0.030)")
    p.add_argument("--t-end", type=float, default=0.050, help="end of the simulated window, s (default: 0.050)")
    p.add_argument("--n-samples", type=int, default=2000, help="number of time samples to plot (default: 2000)")
    p.add_argument("--output", type=str, default=None, help="output PNG path (default: alongside this script)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ch_pos, ch_neg = _AXIS_CHANNELS[args.axis]

    magnet_position = {"x": (args.magnet_z, 0, 0), "y": (0, args.magnet_z, 0), "z": (0, 0, args.magnet_z)}[args.axis]
    sensor_point = (0.0, 0.0, 0.0)
    axis_idx = {"x": 0, "y": 1, "z": 2}[args.axis]

    magnet = Kresling(position=magnet_position, orientation=_AXIS_ORIENTATION[args.axis])
    act = Actuation(magnet, rise_time_s=args.rise_time)

    act.command({ch_pos: args.current, ch_neg: args.current}, t=args.t_actuate)
    act.command({ch_pos: 0.0, ch_neg: 0.0}, t=args.t_sense)
    act.command({ch_pos: args.current, ch_neg: args.current}, t=args.t_resume)

    ts = np.linspace(0, args.t_end, args.n_samples)
    i_pos = np.array([act.currents_at(t)[ch_pos] for t in ts])
    b_total = np.array([act.field_at(sensor_point, t)[axis_idx] for t in ts])
    b_magnet_only = magnet.getB(sensor_point)[axis_idx]  # constant -- magnet doesn't move here

    tau = args.rise_time / math.log(9)
    t_settle_1pct = tau * math.log(100)
    t_settle_0p1pct = tau * math.log(1000)
    sense_mask = (ts >= args.t_sense) & (ts < args.t_resume)

    print(f"axis={args.axis} ({ch_pos}={ch_neg}={args.current}A), magnet at {magnet_position}mm, rise_time={args.rise_time * 1000:.3f}ms")
    print(f"magnet-alone B[{args.axis}] at sensor: {b_magnet_only:.6f} mT")
    print(f"peak actuation B[{args.axis}] at sensor: {b_total.max():.4f} mT")
    print(f"  -> {b_total.max() / args.sat_threshold * 100:.1f}% of the {args.sat_threshold}mT saturation threshold")
    if abs(b_magnet_only) > 0:
        print(f"  -> magnet signal is {abs(b_total.max() / b_magnet_only):.0f}x smaller than the actuation field while coils are on")
    print(f"current tau = {tau * 1000:.3f} ms; settles to 1% in ~{t_settle_1pct * 1000:.2f} ms, to 0.1% in ~{t_settle_0p1pct * 1000:.2f} ms")
    if sense_mask.any():
        print(f"minimum residual B[{args.axis}] during the sense window: {b_total[sense_mask].min():.6f} mT")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)

    ax1.plot(ts * 1000, i_pos, color="tab:blue")
    ax1.set_ylabel(f"{ch_pos} = {ch_neg} (A)")
    ax1.set_title(f"Commanded coil current ({args.axis}-axis, uniform mode)")
    ax1.grid(alpha=0.3)

    ax2.plot(ts * 1000, b_total, color="tab:blue", label=f"B{args.axis} at sensor (coils + magnet)")
    ax2.axhline(b_magnet_only, color="tab:green", linewidth=1.5, label=f"magnet alone ({b_magnet_only:.4f} mT)")
    ax2.axhline(args.sat_threshold, color="tab:red", linestyle="--", alpha=0.7, label=f"sensor saturation ({args.sat_threshold} mT)")
    ax2.axvspan(args.t_sense * 1000, args.t_resume * 1000, color="gray", alpha=0.12, label="sense window")
    ax2.set_ylabel(f"B{args.axis} at sensor (mT)")
    ax2.set_xlabel("time (ms)")
    ax2.set_title("Field at sensor: actuate -> sense -> actuate")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_path = args.output or __file__.replace(".py", ".png")
    fig.savefig(out_path, dpi=150)
    print(f"\nsaved plot to {out_path}")


if __name__ == "__main__":
    main()
