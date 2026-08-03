"""Sweep sensor-grid configurations over TWO variables -- sensor count and
pitch (spacing between consecutive sensors) -- rank by CRB, validate the
top candidates with Monte Carlo, and plot achievable position error vs.
sensor count (one line per pitch) -- the "optimum number and placement of
sensors" deliverable everything else in this repo has been building toward.

Test positions (where the magnet is assumed to actually be, for scoring
each candidate layout): xy sampled within the Kresling cell's own polygon
radius (r=16.8mm, Nature SI S10) around the plate center -- the only
concrete xy figure either paper gives; there's no stated number for how far
the assembly roams during sustained locomotion, so this uses the single
cell's resting footprint rather than inventing a roaming range. z sampled
across the confirmed v1 workspace range (~3mm sensor-plane gap to ~28.3mm,
h1 deployed cell height).

Approach: for each (count, pitch) combination, compute the CRB position-RMS
averaged over every (xy, z) test position -- that's the fast ranking
metric, no solver runs. Then re-run the best few configs through actual
Monte Carlo (noisy simulate + LeastSquaresSolver) to confirm the CRB
ranking is trustworthy, not just theoretically convenient.

Run with: uv run python scripts/sweep.py
Writes: scripts/sweep.png
"""
from __future__ import annotations

import argparse
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kresling_sim import fisher
from kresling_sim.montecarlo import DEFAULT_NOISE, run_monte_carlo
from kresling_sim.sensors import positions_array, sensor_grid

# xy: sampled within the Kresling cell's own polygon radius (Nature SI S10),
# the only concrete xy figure the papers give -- center plus 6 points toward
# the edge (cardinal + two diagonal directions).
CELL_POLYGON_RADIUS_MM = 16.8
_XY_UNIT_OFFSETS = np.array(
    [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [0.7071, 0.7071], [-0.7071, -0.7071]]
)
XY_TEST_OFFSETS_MM = _XY_UNIT_OFFSETS * CELL_POLYGON_RADIUS_MM

# z: the confirmed v1 workspace range -- ~3mm sensor-plane gap to ~28.3mm
# (h1, deployed cell height).
Z_TEST_VALUES_MM = np.linspace(3.0, 28.3, 4)

TEST_POSITIONS_MM = np.array(
    [[x, y, z] for z in Z_TEST_VALUES_MM for x, y in XY_TEST_OFFSETS_MM]
)

# A smaller subset for the (slow) Monte Carlo validation step -- center and
# the two most extreme corners (nearest/farthest z, plus an off-axis xy),
# not the full 28-point grid, so validation stays fast.
MC_VALIDATION_POSITIONS_MM = np.array(
    [
        [0.0, 0.0, Z_TEST_VALUES_MM[0]],
        [0.0, 0.0, Z_TEST_VALUES_MM[-1]],
        [CELL_POLYGON_RADIUS_MM, 0.0, Z_TEST_VALUES_MM[-1]],
        [-CELL_POLYGON_RADIUS_MM * 0.7071, -CELL_POLYGON_RADIUS_MM * 0.7071, Z_TEST_VALUES_MM[0]],
    ]
)


def crb_rms_mm(points: np.ndarray, noise, test_positions: np.ndarray = TEST_POSITIONS_MM) -> float:
    """Average (over test_positions) CRB position-RMS (mm) for one sensor
    layout -- the fast ranking metric, no solver runs involved."""
    rms_values = []
    for position in test_positions:
        jacobian = fisher.position_jacobian(position, points)
        fim = fisher.fisher_information_matrix(jacobian, noise)
        bound = fisher.crlb(fim)  # (3,) per-axis std dev, mm
        rms_values.append(float(np.sqrt(np.sum(bound**2))))
    return float(np.mean(rms_values))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--counts", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7], help="grid side lengths to sweep (num_across == num_below for each), default: 2..7")
    p.add_argument("--pitches", type=float, nargs="+", default=[5.0, 8.0, 10.0, 15.0, 20.0], help="sensor pitch values to sweep, mm, both directions (default: 5 8 10 15 20)")
    p.add_argument("--validate-top", type=int, default=3, help="how many top-ranked (by CRB) configs to re-check with real Monte Carlo (default: 3)")
    p.add_argument("--n-trials", type=int, default=150, help="Monte Carlo trials per validated config per validation position (default: 150)")
    p.add_argument("--output", type=str, default=None, help="output PNG path (default: alongside this script)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    noise = DEFAULT_NOISE

    print(f"test positions: xy within r={CELL_POLYGON_RADIUS_MM}mm of center, z in {Z_TEST_VALUES_MM.round(1).tolist()}mm ({len(TEST_POSITIONS_MM)} total)")
    print(f"sweeping counts={args.counts} x pitches={args.pitches}mm, noise sigma={noise.sigma} mT\n")

    configs = []
    for side, pitch in itertools.product(args.counts, args.pitches):
        sensors = sensor_grid(num_across=side, num_below=side, sep_horizontal=pitch, sep_vertical=pitch)
        points = positions_array(sensors)
        n_sensors = len(sensors)
        crb = crb_rms_mm(points, noise)
        span = (side - 1) * pitch
        configs.append(
            {"side": side, "pitch": pitch, "n_sensors": n_sensors, "span_mm": span, "points": points, "crb_rms_mm": crb}
        )
        print(f"  {side}x{side} @ {pitch:5.1f}mm pitch ({n_sensors:3d} sensors, {span:5.1f}mm span): CRB position-RMS = {crb:.4f} mm")

    configs.sort(key=lambda c: c["crb_rms_mm"])
    print("\ntop 10 ranked best -> worst by CRB:")
    for c in configs[:10]:
        print(f"  {c['side']}x{c['side']} @ {c['pitch']:5.1f}mm ({c['n_sensors']:3d} sensors, {c['span_mm']:5.1f}mm span): {c['crb_rms_mm']:.4f} mm")

    # validate the top-ranked configs with real (slow) Monte Carlo
    top = configs[: args.validate_top]
    print(f"\nvalidating top {len(top)} config(s) with Monte Carlo ({args.n_trials} trials x {len(MC_VALIDATION_POSITIONS_MM)} positions each)...")
    for c in top:
        empirical_rmses = []
        crb_rmses = []
        for true_position in MC_VALIDATION_POSITIONS_MM:
            result = run_monte_carlo(true_position, c["points"], noise=noise, n_trials=args.n_trials)
            empirical_rmses.append(result.rmse_mm)
            crb_rmses.append(result.crb_rmse_mm)
        c["mc_rmse_mm"] = float(np.mean(empirical_rmses))
        c["mc_crb_rmse_mm"] = float(np.mean(crb_rmses))
        print(
            f"  {c['side']}x{c['side']} @ {c['pitch']:5.1f}mm ({c['n_sensors']:3d} sensors): "
            f"empirical={c['mc_rmse_mm']:.4f} mm, CRB={c['mc_crb_rmse_mm']:.4f} mm "
            f"(agreement: {100 * c['mc_rmse_mm'] / c['mc_crb_rmse_mm']:.1f}%)"
        )

    best = configs[0]
    print(
        f"\nbest config by CRB: {best['side']}x{best['side']} @ {best['pitch']}mm pitch "
        f"({best['n_sensors']} sensors, {best['span_mm']:.1f}mm span), CRB position-RMS = {best['crb_rms_mm']:.4f} mm"
    )

    # plot: CRB-RMS vs sensor count, one line per pitch, MC-validated points overlaid
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(args.pitches)))
    for pitch, color in zip(args.pitches, colors):
        subset = sorted([c for c in configs if c["pitch"] == pitch], key=lambda c: c["n_sensors"])
        ax.plot(
            [c["n_sensors"] for c in subset],
            [c["crb_rms_mm"] for c in subset],
            "o-",
            color=color,
            label=f"pitch={pitch:g}mm",
        )

    validated = [c for c in configs if "mc_rmse_mm" in c]
    if validated:
        ax.plot(
            [c["n_sensors"] for c in validated],
            [c["mc_rmse_mm"] for c in validated],
            "s",
            color="tab:red",
            markersize=10,
            markeredgecolor="black",
            label="empirical Monte Carlo RMSE (validated configs)",
            zorder=5,
        )

    ax.set_xlabel("number of sensors")
    ax.set_ylabel("position RMS error (mm)")
    ax.set_title("Sensor count & pitch vs. achievable localization accuracy")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out_path = args.output or __file__.replace(".py", ".png")
    fig.savefig(out_path, dpi=150)
    print(f"\nsaved plot to {out_path}")


if __name__ == "__main__":
    main()
