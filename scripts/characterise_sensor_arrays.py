"""Spatially characterise localisation accuracy for each sensor array config.

For every (count, pitch) pair -- counts 5-10, pitches 12-24mm in 2mm steps,
grid centred at (0, 0) -- this script:
  1. saves the sensor positions
  2. runs warm-start Monte Carlo at a regular grid of magnet positions
     spanning the full workspace volume
  3. plots RMSE vs x (one line per z level, averaged over y)
  4. plots RMSE vs z (one line per xy region: centre / edge / corner)
  5. plots a 2-D RMSE heatmap averaged over z
  6. saves raw RMSE data as .npz for later analysis

All outputs land in scripts/sensor_array/<count>x<count>_<pitch>mm/.
A cross-config summary plot is written to scripts/sensor_array/summary.png.

Run with: uv run python scripts/characterise_sensor_arrays.py
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kresling_sim.config import (
    SENSOR_PCB_GAP_MM,
    WORKSPACE_X_MM,
    WORKSPACE_Y_MM,
    Z_DEPLOYED_MM,
)
from kresling_sim import fisher
from kresling_sim.montecarlo import DEFAULT_NOISE, run_monte_carlo
from kresling_sim.sensors import positions_array, save_sensors, sensor_grid

# ---------------------------------------------------------------------------
# Test grid: 5x5 xy × 4 z = 100 magnet positions across the full workspace
# ---------------------------------------------------------------------------
N_XY = 5
N_Z = 4
TEST_XS = np.linspace(-WORKSPACE_X_MM / 2, WORKSPACE_X_MM / 2, N_XY)
TEST_YS = np.linspace(-WORKSPACE_Y_MM / 2, WORKSPACE_Y_MM / 2, N_XY)
TEST_ZS = np.linspace(SENSOR_PCB_GAP_MM, Z_DEPLOYED_MM, N_Z)

# (N_XY*N_XY, 2) -- row-major so outer loop is y, inner is x (matches meshgrid)
_gx, _gy = np.meshgrid(TEST_XS, TEST_YS)
XY_GRID = np.column_stack([_gx.ravel(), _gy.ravel()])  # (25, 2)


# ---------------------------------------------------------------------------
# Config space
# ---------------------------------------------------------------------------
COUNTS = list(range(5, 11))           # 5, 6, 7, 8, 9, 10
PITCHES = list(range(12, 25, 2))      # 12, 14, 16, 18, 20, 22, 24 mm
WARM_START_SIGMA = 3.0                # mm -- represents tracking-mode uncertainty


def run_config(
    side: int,
    pitch: float,
    n_trials: int,
    rng: np.random.Generator,
    outdir: str,
) -> tuple[float, float]:
    """Run one (side x side @ pitch mm) config. Returns (agg_mc_rmse, agg_crb) in mm."""
    os.makedirs(outdir, exist_ok=True)

    # --- build and save sensor grid ---
    sensors = sensor_grid(num_across=side, num_below=side,
                          sep_horizontal=pitch, sep_vertical=pitch)
    points = positions_array(sensors)
    save_sensors(sensors, os.path.join(outdir, "sensor_positions"))
    np.savetxt(os.path.join(outdir, "sensor_positions.txt"), points,
               header="x_mm  y_mm  z_mm", fmt="%.3f")

    # --- CRB at every test position (analytical, fast) ---
    crb = np.full((len(XY_GRID), N_Z), np.nan)
    for i_z, z in enumerate(TEST_ZS):
        for i_xy, (x, y) in enumerate(XY_GRID):
            true_pos = np.array([x, y, z])
            jac = fisher.position_jacobian(true_pos, points)
            fim = fisher.fisher_information_matrix(jac, DEFAULT_NOISE)
            bound = fisher.crlb(fim)
            crb[i_xy, i_z] = float(np.sqrt(np.sum(bound**2)))

    # --- warm-start MC at every test position ---
    # rmse[i_xy, i_z]: empirical RMSE at XY_GRID[i_xy] and TEST_ZS[i_z]
    rmse = np.full((len(XY_GRID), N_Z), np.nan)
    total = len(XY_GRID) * N_Z
    done = 0
    for i_z, z in enumerate(TEST_ZS):
        for i_xy, (x, y) in enumerate(XY_GRID):
            true_pos = np.array([x, y, z])
            result = run_monte_carlo(
                true_pos, points,
                noise=DEFAULT_NOISE,
                n_trials=n_trials,
                warm_start_sigma=WARM_START_SIGMA,
                rng=rng,
            )
            rmse[i_xy, i_z] = result.rmse_mm
            done += 1
            if done % 10 == 0:
                print(f"    {done}/{total} positions done", end="\r")

    print()
    np.savez(os.path.join(outdir, "rmse_data.npz"),
             rmse=rmse, crb=crb, xy_grid=XY_GRID,
             test_xs=TEST_XS, test_ys=TEST_YS, test_zs=TEST_ZS)

    _plot_rmse_vs_x(rmse, crb, outdir, side, pitch)
    _plot_rmse_vs_z(rmse, crb, outdir, side, pitch)
    _plot_heatmap(rmse, crb, outdir, side, pitch)

    return float(np.nanmean(rmse)), float(np.nanmean(crb))


def _plot_rmse_vs_x(rmse: np.ndarray, crb: np.ndarray, outdir: str, side: int, pitch: float) -> None:
    """MC RMSE and CRB vs x, one line per z level, averaged over y."""
    rmse_grid = rmse.reshape(N_XY, N_XY, N_Z)   # (y, x, z)
    crb_grid  = crb.reshape(N_XY, N_XY, N_Z)
    rmse_vs_x = rmse_grid.mean(axis=0)           # (x, z) averaged over y
    crb_vs_x  = crb_grid.mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, N_Z))
    for i_z, (z, color) in enumerate(zip(TEST_ZS, colors)):
        ax.plot(TEST_XS, rmse_vs_x[:, i_z], "o-",  color=color, label=f"MC z={z:.1f}mm")
        ax.plot(TEST_XS, crb_vs_x[:, i_z],  "--",  color=color, alpha=0.5, label=f"CRB z={z:.1f}mm")

    ax.set_xlabel("x position (mm)")
    ax.set_ylabel("error (mm)")
    ax.set_title(f"{side}x{side} @ {pitch:.0f}mm — RMSE vs x  (solid=MC, dashed=CRB, mean over y)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "rmse_vs_x.png"), dpi=130)
    plt.close(fig)


def _plot_rmse_vs_z(rmse: np.ndarray, crb: np.ndarray, outdir: str, side: int, pitch: float) -> None:
    """MC RMSE and CRB vs z for centre, edge midpoints, and corners."""
    rmse_grid = rmse.reshape(N_XY, N_XY, N_Z)
    crb_grid  = crb.reshape(N_XY, N_XY, N_Z)
    cx = cy = N_XY // 2

    def _region(g):
        centre = g[cy, cx, :]
        edge   = np.mean([g[cy, 0, :], g[cy, -1, :], g[0, cx, :], g[-1, cx, :]], axis=0)
        corner = np.mean([g[0, 0, :], g[0, -1, :], g[-1, 0, :], g[-1, -1, :]], axis=0)
        return centre, edge, corner

    mc_c, mc_e, mc_corner = _region(rmse_grid)
    crb_c, crb_e, crb_corner = _region(crb_grid)

    fig, ax = plt.subplots(figsize=(7, 5))
    for mc, crb_vals, label, marker in [
        (mc_c, crb_c, "centre (0,0)", "o"),
        (mc_e, crb_e, "edge midpoints", "s"),
        (mc_corner, crb_corner, "corners (±50,±50)", "^"),
    ]:
        (line,) = ax.plot(TEST_ZS, mc, f"{marker}-", label=f"MC {label}")
        ax.plot(TEST_ZS, crb_vals, "--", color=line.get_color(), alpha=0.5, label=f"CRB {label}")

    ax.set_xlabel("z — magnet height (mm)")
    ax.set_ylabel("error (mm)")
    ax.set_title(f"{side}x{side} @ {pitch:.0f}mm — RMSE vs z  (solid=MC, dashed=CRB)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "rmse_vs_z.png"), dpi=130)
    plt.close(fig)


def _plot_heatmap(rmse: np.ndarray, crb: np.ndarray, outdir: str, side: int, pitch: float) -> None:
    """Side-by-side heatmaps: MC RMSE and CRB, averaged over z."""
    rmse_xy = rmse.reshape(N_XY, N_XY, N_Z).mean(axis=2)
    crb_xy  = crb.reshape(N_XY, N_XY, N_Z).mean(axis=2)
    vmax = max(rmse_xy.max(), crb_xy.max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    extent = [TEST_XS[0], TEST_XS[-1], TEST_YS[0], TEST_YS[-1]]
    for ax, data, title in zip(axes, [rmse_xy, crb_xy], ["MC RMSE", "CRB"]):
        im = ax.imshow(data, origin="lower", extent=extent,
                       aspect="equal", cmap="hot_r", vmin=0, vmax=vmax)
        fig.colorbar(im, ax=ax, label="error (mm)")
        ax.set_xlabel("x (mm)")
        ax.set_title(f"{side}x{side} @ {pitch:.0f}mm — {title} (mean over z)")
    axes[0].set_ylabel("y (mm)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "heatmap.png"), dpi=130)
    plt.close(fig)


def plot_summary(results: list[dict], outdir: str) -> None:
    """Aggregate MC RMSE and CRB vs sensor count, one line per pitch."""
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(PITCHES)))

    for pitch, color in zip(PITCHES, colors):
        subset = sorted(
            [r for r in results if r["pitch"] == pitch],
            key=lambda r: r["n_sensors"],
        )
        ns = [r["n_sensors"] for r in subset]
        ax.plot(ns, [r["agg_rmse_mm"] for r in subset],
                "o-", color=color, label=f"MC pitch={pitch}mm")
        ax.plot(ns, [r["agg_crb_mm"] for r in subset],
                "--", color=color, alpha=0.5, label=f"CRB pitch={pitch}mm")

    ax.set_xlabel("number of sensors")
    ax.set_ylabel("aggregate error (mm)")
    ax.set_title(f"MC RMSE (solid) vs CRB (dashed) across workspace\n(warm-start σ={WARM_START_SIGMA}mm)")
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "summary.png"), dpi=150)
    plt.close(fig)
    print(f"\nsummary plot saved to {outdir}/summary.png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-trials", type=int, default=50, help="MC trials per test position (default: 50)")
    p.add_argument("--outdir", type=str, default=None, help="root output directory (default: scripts/sensor_array/)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.outdir or os.path.join(os.path.dirname(__file__), "sensor_array")
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(42)

    n_configs = len(COUNTS) * len(PITCHES)
    print(f"characterising {n_configs} sensor array configs")
    print(f"  counts: {COUNTS}")
    print(f"  pitches: {PITCHES} mm")
    print(f"  test grid: {N_XY}x{N_XY} xy × {N_Z} z = {N_XY*N_XY*N_Z} positions per config")
    print(f"  warm-start sigma: {WARM_START_SIGMA} mm, {args.n_trials} trials/position")
    print(f"  output root: {root}\n")

    results = []
    for i, side in enumerate(COUNTS):
        for j, pitch in enumerate(PITCHES):
            span = (side - 1) * pitch
            config_name = f"{side}x{side}_{pitch:.0f}mm"
            outdir = os.path.join(root, config_name)
            print(f"[{i*len(PITCHES)+j+1}/{n_configs}] {side}x{side} @ {pitch}mm pitch "
                  f"({side*side} sensors, {span:.0f}mm span) → {config_name}/")
            agg_rmse, agg_crb = run_config(side, pitch, args.n_trials, rng, outdir)
            print(f"    aggregate MC RMSE: {agg_rmse:.4f} mm  CRB: {agg_crb:.4f} mm  (MC/CRB={100*agg_rmse/agg_crb:.0f}%)")
            results.append({"side": side, "pitch": pitch,
                            "n_sensors": side * side, "span_mm": span,
                            "agg_rmse_mm": agg_rmse, "agg_crb_mm": agg_crb})

    results.sort(key=lambda r: r["agg_rmse_mm"])
    print("\nfinal ranking (best → worst by aggregate RMSE):")
    for r in results:
        print(f"  {r['side']}x{r['side']} @ {r['pitch']:5.1f}mm "
              f"({r['n_sensors']:3d} sensors, {r['span_mm']:5.0f}mm span): "
              f"{r['agg_rmse_mm']:.4f} mm")

    plot_summary(results, root)


if __name__ == "__main__":
    main()
