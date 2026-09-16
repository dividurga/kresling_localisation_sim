"""Sweep sensor-grid configurations over TWO variables -- sensor count and
pitch (spacing between consecutive sensors) -- rank by CRB, validate the
top candidates with Monte Carlo, and plot achievable position error vs.
sensor count (one line per pitch) -- the "optimum number and placement of
sensors" deliverable everything else in this repo has been building toward.

Test positions: the sensor array is fixed on the floor; the Kresling cell
(and its magnet) can occupy any location in the full workspace volume --
WORKSPACE_X_MM x WORKSPACE_Y_MM in xy, SENSOR_PCB_GAP_MM to Z_DEPLOYED_MM
in z. xy is sampled on a 5x5 grid across the workspace; z at 4 evenly
spaced heights = 100 test positions total for CRB ranking.

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
from kresling_sim.config import SENSOR_PCB_GAP_MM, WORKSPACE_X_MM, WORKSPACE_Y_MM, Z_DEPLOYED_MM
from kresling_sim.montecarlo import DEFAULT_NOISE, run_monte_carlo
from kresling_sim.sensors import positions_array, sensor_grid

# xy: 5x5 grid across the full workspace (sensor array is fixed, magnet can
# be anywhere in the WORKSPACE_X_MM x WORKSPACE_Y_MM footprint).
_XY_GRID_1D_X = np.linspace(-WORKSPACE_X_MM / 2, WORKSPACE_X_MM / 2, 5)
_XY_GRID_1D_Y = np.linspace(-WORKSPACE_Y_MM / 2, WORKSPACE_Y_MM / 2, 5)
_gx, _gy = np.meshgrid(_XY_GRID_1D_X, _XY_GRID_1D_Y)
XY_TEST_POSITIONS_MM = np.column_stack([_gx.ravel(), _gy.ravel()])  # (25, 2)

# z: sensor-plane gap (SENSOR_PCB_GAP_MM) to fully deployed height (Z_DEPLOYED_MM).
Z_TEST_VALUES_MM = np.linspace(SENSOR_PCB_GAP_MM, Z_DEPLOYED_MM, 4)

TEST_POSITIONS_MM = np.array(
    [[x, y, z] for z in Z_TEST_VALUES_MM for x, y in XY_TEST_POSITIONS_MM]
)  # 100 total positions

# MC validation: center + 4 workspace corners, at both extreme z values -- 10 positions.
# Covers the hardest cases (corners at min z = closest to sensor plane, worst SNR at range).
_HALF_X, _HALF_Y = WORKSPACE_X_MM / 2, WORKSPACE_Y_MM / 2
MC_VALIDATION_POSITIONS_MM = np.array(
    [
        [0.0,     0.0,    Z_TEST_VALUES_MM[0]],
        [0.0,     0.0,    Z_TEST_VALUES_MM[-1]],
        [ _HALF_X,  _HALF_Y, Z_TEST_VALUES_MM[0]],
        [-_HALF_X,  _HALF_Y, Z_TEST_VALUES_MM[-1]],
        [ _HALF_X, -_HALF_Y, Z_TEST_VALUES_MM[-1]],
        [-_HALF_X, -_HALF_Y, Z_TEST_VALUES_MM[0]],
    ]
)


def crb_rms_mm(points: np.ndarray, noise, test_positions: np.ndarray = None) -> float:
    """Average CRB position-RMS (mm) over test_positions (analytical, no solver)."""
    positions = test_positions if test_positions is not None else TEST_POSITIONS_MM
    rmses = []
    for position in positions:
        jacobian = fisher.position_jacobian(position, points)
        fim = fisher.fisher_information_matrix(jacobian, noise)
        bound = fisher.crlb(fim)
        rmses.append(float(np.sqrt(np.sum(bound**2))))
    return float(np.mean(rmses))


def mc_rms_mm(
    points: np.ndarray,
    noise,
    test_positions: np.ndarray,
    n_trials: int,
    warm_start_sigma: float,
    rng: np.random.Generator,
) -> float:
    """Average warm-start MC position-RMSE (mm) over test_positions."""
    rmses = []
    for position in test_positions:
        result = run_monte_carlo(
            position, points, noise=noise, n_trials=n_trials,
            warm_start_sigma=warm_start_sigma, rng=rng,
        )
        rmses.append(result.rmse_mm)
    return float(np.mean(rmses))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--counts", type=int, nargs="+", default=[4, 5, 6, 7, 8], help="grid side lengths to sweep (default: 4..8)")
    p.add_argument("--pitches", type=float, nargs="+", default=[10.0, 12.0, 15.0, 17.0, 20.0], help="sensor pitch values to sweep, mm (default: 10 12 15 17 20)")
    p.add_argument("--validate-top", type=int, default=5, help="how many top CRB-ranked configs to validate with warm-start MC (default: 5)")
    p.add_argument("--n-trials", type=int, default=100, help="MC trials per position for validated configs (default: 100)")
    p.add_argument("--warm-start-sigma", type=float, default=3.0, help="std dev (mm) of warm-start perturbation (default: 3.0)")
    p.add_argument("--output", type=str, default=None, help="output PNG path (default: alongside this script)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    noise = DEFAULT_NOISE
    rng = np.random.default_rng(42)

    print(f"test positions: {len(TEST_POSITIONS_MM)} total ({len(XY_TEST_POSITIONS_MM)} xy × {len(Z_TEST_VALUES_MM)} z) across {WORKSPACE_X_MM}x{WORKSPACE_Y_MM}mm workspace")
    print(f"sweeping counts={args.counts} x pitches={args.pitches}mm")
    print(f"CRB ranks all configs; warm-start MC (sigma={args.warm_start_sigma}mm, {args.n_trials} trials) validates top {args.validate_top}\n")

    # --- Phase 1: CRB ranking (fast, analytical, all configs) ---
    configs = []
    for side, pitch in itertools.product(args.counts, args.pitches):
        sensors = sensor_grid(num_across=side, num_below=side, sep_horizontal=pitch, sep_vertical=pitch)
        points = positions_array(sensors)
        n_sensors = len(sensors)
        span = (side - 1) * pitch
        crb = crb_rms_mm(points, noise)
        configs.append({"side": side, "pitch": pitch, "n_sensors": n_sensors, "span_mm": span, "points": points, "crb_rms_mm": crb})
        print(f"  {side}x{side} @ {pitch:5.1f}mm ({n_sensors:3d} sensors, {span:5.1f}mm span): CRB = {crb:.4f} mm")

    configs.sort(key=lambda c: c["crb_rms_mm"])
    print(f"\ntop 10 by CRB:")
    for c in configs[:10]:
        print(f"  {c['side']}x{c['side']} @ {c['pitch']:5.1f}mm ({c['n_sensors']:3d} sensors, {c['span_mm']:5.1f}mm span): {c['crb_rms_mm']:.4f} mm")

    # --- Phase 2: warm-start MC validation (all 100 positions, top N configs) ---
    top = configs[: args.validate_top]
    print(f"\nvalidating top {len(top)} configs with warm-start MC ({args.n_trials} trials × {len(TEST_POSITIONS_MM)} positions)...")
    for c in top:
        mc = mc_rms_mm(c["points"], noise, TEST_POSITIONS_MM, args.n_trials, args.warm_start_sigma, rng)
        c["mc_rmse_mm"] = mc
        agreement = 100 * mc / c["crb_rms_mm"]
        print(f"  {c['side']}x{c['side']} @ {c['pitch']:5.1f}mm ({c['n_sensors']:3d} sensors): "
              f"CRB={c['crb_rms_mm']:.4f} mm  MC={mc:.4f} mm  (MC/CRB={agreement:.0f}%)")

    best_crb = configs[0]
    best_mc  = min((c for c in configs if "mc_rmse_mm" in c), key=lambda c: c["mc_rmse_mm"])
    print(f"\nbest by CRB: {best_crb['side']}x{best_crb['side']} @ {best_crb['pitch']}mm — CRB {best_crb['crb_rms_mm']:.4f} mm")
    print(f"best by MC:  {best_mc['side']}x{best_mc['side']} @ {best_mc['pitch']}mm — MC  {best_mc['mc_rmse_mm']:.4f} mm")

    # --- Plot: CRB curves + MC validated points overlaid ---
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(args.pitches)))
    for pitch, color in zip(args.pitches, colors):
        subset = sorted([c for c in configs if c["pitch"] == pitch], key=lambda c: c["n_sensors"])
        ax.plot([c["n_sensors"] for c in subset], [c["crb_rms_mm"] for c in subset],
                "o-", color=color, label=f"CRB pitch={pitch:g}mm")

    validated = [c for c in configs if "mc_rmse_mm" in c]
    if validated:
        ax.scatter([c["n_sensors"] for c in validated], [c["mc_rmse_mm"] for c in validated],
                   s=120, color="tab:red", marker="s", edgecolors="black", zorder=5,
                   label=f"warm-start MC (σ={args.warm_start_sigma}mm)")

    ax.set_xlabel("number of sensors")
    ax.set_ylabel("position RMS error (mm)")
    ax.set_title("CRB ranking + warm-start MC validation across full workspace")
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out_path = args.output or __file__.replace(".py", ".png")
    fig.savefig(out_path, dpi=150)
    print(f"\nsaved plot to {out_path}")


if __name__ == "__main__":
    main()
