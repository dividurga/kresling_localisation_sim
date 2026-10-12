"""sweep sensor-grid configs over count and pitch, rank by CRB, validate the
top candidates with Monte Carlo, and plot error vs sensor count (one line
per pitch) -- the "optimal sensor placement" deliverable everything else
here has been building toward.

--target picks kresling or cube_robot (see cube_robot_config.py for where
its numbers come from).

Test positions: sensor array is fixed, magnet can be anywhere in the
workspace footprint -- sampled on a 5x5 xy grid x 4 z heights, 100
positions total for CRB ranking.

CRB position-RMS averaged over all 100 positions ranks every config for
free (no solver). Top few then get validated with real Monte Carlo (noisy
simulate + solve) to check the ranking isn't just theoretical.

Run with: uv run python scripts/sweep.py [--target kresling|cube_robot]
Writes: scripts/sweep_<target>.png
"""
from __future__ import annotations

import argparse
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kresling_sim import cube_robot_config, fisher, kresling_2026_supp_config
from kresling_sim.config import SENSOR_PCB_GAP_MM, WORKSPACE_X_MM, WORKSPACE_Y_MM, Z_DEPLOYED_MM
from kresling_sim.montecarlo import DEFAULT_NOISE, run_monte_carlo
from kresling_sim.sensors import sensor_grid

TARGETS = {
    "kresling": {"z_min": SENSOR_PCB_GAP_MM, "z_max": Z_DEPLOYED_MM, "magnet_kwargs": None},
    "cube_robot": {
        "z_min": cube_robot_config.Z_MIN_MM,
        "z_max": cube_robot_config.Z_MAX_MM,
        "magnet_kwargs": cube_robot_config.MAGNET_KWARGS,
    },
    "kresling_2026_supp": {
        "z_min": kresling_2026_supp_config.Z_MIN_MM,
        "z_max": kresling_2026_supp_config.Z_MAX_MM,
        "magnet_kwargs": kresling_2026_supp_config.MAGNET_KWARGS,
    },
}

# xy test grid: 5x5 across the full workspace, shared by both targets
# (only z-range differs)
_XY_GRID_1D_X = np.linspace(-WORKSPACE_X_MM / 2, WORKSPACE_X_MM / 2, 5)
_XY_GRID_1D_Y = np.linspace(-WORKSPACE_Y_MM / 2, WORKSPACE_Y_MM / 2, 5)
_gx, _gy = np.meshgrid(_XY_GRID_1D_X, _XY_GRID_1D_Y)
XY_TEST_POSITIONS_MM = np.column_stack([_gx.ravel(), _gy.ravel()])  # (25, 2)
_HALF_X, _HALF_Y = WORKSPACE_X_MM / 2, WORKSPACE_Y_MM / 2


def build_test_positions(z_min: float, z_max: float) -> tuple[np.ndarray, np.ndarray]:
    """(TEST_POSITIONS_MM, MC_VALIDATION_POSITIONS_MM) for the given
    z-range -- see module docstring for the xy/z sampling scheme."""
    z_values = np.linspace(z_min, z_max, 4)
    test_positions = np.array([[x, y, z] for z in z_values for x, y in XY_TEST_POSITIONS_MM])  # 100 total

    # MC validation set: center + 4 corners at both extreme z -- 10 positions,
    # covers the hardest cases (corner at min z = worst SNR)
    mc_validation_positions = np.array(
        [
            [0.0,     0.0,    z_values[0]],
            [0.0,     0.0,    z_values[-1]],
            [ _HALF_X,  _HALF_Y, z_values[0]],
            [-_HALF_X,  _HALF_Y, z_values[-1]],
            [ _HALF_X, -_HALF_Y, z_values[-1]],
            [-_HALF_X, -_HALF_Y, z_values[0]],
        ]
    )
    return test_positions, mc_validation_positions


def crb_rms_mm(sensors, noise, test_positions: np.ndarray, magnet_kwargs: dict | None = None) -> float:
    """average CRB position-RMS (mm) over test_positions (analytical, no solver)."""
    rmses = []
    for position in test_positions:
        jacobian = fisher.position_jacobian(position, sensors, magnet_kwargs=magnet_kwargs)
        fim = fisher.fisher_information_matrix(jacobian, noise)
        bound = fisher.crlb(fim)
        rmses.append(float(np.sqrt(np.sum(bound**2))))
    return float(np.mean(rmses))


def mc_rms_mm(
    sensors,
    noise,
    test_positions: np.ndarray,
    n_trials: int,
    warm_start_sigma: float,
    rng: np.random.Generator,
    magnet_kwargs: dict | None = None,
) -> float:
    """average warm-start MC position-RMSE (mm) over test_positions."""
    rmses = []
    for position in test_positions:
        result = run_monte_carlo(
            position, sensors, noise=noise, n_trials=n_trials,
            warm_start_sigma=warm_start_sigma, rng=rng, magnet_kwargs=magnet_kwargs,
        )
        rmses.append(result.rmse_mm)
    return float(np.mean(rmses))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", choices=sorted(TARGETS), default="kresling", help="magnet/geometry target to sweep (default: kresling)")
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

    target = TARGETS[args.target]
    magnet_kwargs = target["magnet_kwargs"]
    test_positions, mc_validation_positions = build_test_positions(target["z_min"], target["z_max"])

    print(f"target: {args.target}  (z-range {target['z_min']:.2f}-{target['z_max']:.2f}mm)")
    print(f"test positions: {len(test_positions)} total ({len(XY_TEST_POSITIONS_MM)} xy × 4 z) across {WORKSPACE_X_MM}x{WORKSPACE_Y_MM}mm workspace")
    print(f"sweeping counts={args.counts} x pitches={args.pitches}mm")
    print(f"CRB ranks all configs; warm-start MC (sigma={args.warm_start_sigma}mm, {args.n_trials} trials) validates top {args.validate_top}\n")

    # phase 1: CRB ranking (fast, analytical, all configs)
    configs = []
    for side, pitch in itertools.product(args.counts, args.pitches):
        sensors = sensor_grid(num_across=side, num_below=side, sep_horizontal=pitch, sep_vertical=pitch)
        n_sensors = len(sensors)
        span = (side - 1) * pitch
        crb = crb_rms_mm(sensors, noise, test_positions, magnet_kwargs=magnet_kwargs)
        configs.append({"side": side, "pitch": pitch, "n_sensors": n_sensors, "span_mm": span, "sensors": sensors, "crb_rms_mm": crb})
        print(f"  {side}x{side} @ {pitch:5.1f}mm ({n_sensors:3d} sensors, {span:5.1f}mm span): CRB = {crb:.4f} mm")

    configs.sort(key=lambda c: c["crb_rms_mm"])
    print(f"\ntop 10 by CRB:")
    for c in configs[:10]:
        print(f"  {c['side']}x{c['side']} @ {c['pitch']:5.1f}mm ({c['n_sensors']:3d} sensors, {c['span_mm']:5.1f}mm span): {c['crb_rms_mm']:.4f} mm")

    # phase 2: warm-start MC validation (top N configs)
    top = configs[: args.validate_top]
    print(f"\nvalidating top {len(top)} configs with warm-start MC ({args.n_trials} trials × {len(test_positions)} positions)...")
    for c in top:
        mc = mc_rms_mm(c["sensors"], noise, test_positions, args.n_trials, args.warm_start_sigma, rng, magnet_kwargs=magnet_kwargs)
        c["mc_rmse_mm"] = mc
        agreement = 100 * mc / c["crb_rms_mm"]
        print(f"  {c['side']}x{c['side']} @ {c['pitch']:5.1f}mm ({c['n_sensors']:3d} sensors): "
              f"CRB={c['crb_rms_mm']:.4f} mm  MC={mc:.4f} mm  (MC/CRB={agreement:.0f}%)")

    best_crb = configs[0]
    best_mc  = min((c for c in configs if "mc_rmse_mm" in c), key=lambda c: c["mc_rmse_mm"])
    print(f"\nbest by CRB: {best_crb['side']}x{best_crb['side']} @ {best_crb['pitch']}mm — CRB {best_crb['crb_rms_mm']:.4f} mm")
    print(f"best by MC:  {best_mc['side']}x{best_mc['side']} @ {best_mc['pitch']}mm — MC  {best_mc['mc_rmse_mm']:.4f} mm")

    # plot: CRB curves + MC validated points overlaid
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
    ax.set_title(f"CRB ranking + warm-start MC validation across full workspace ({args.target})")
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out_path = args.output or __file__.replace(".py", f"_{args.target}.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nsaved plot to {out_path}")


if __name__ == "__main__":
    main()
