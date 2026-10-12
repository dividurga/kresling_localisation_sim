"""sweep sensor array configs (count x count, pitch 12-24mm) and check localisation
accuracy at a grid of magnet positions across the workspace. Per config: build the
grid, run CRB (analytical) + warm-start MC at every test position, dump plots
(rmse vs x, vs z, heatmap) and raw data, plus one summary plot across all configs.

--target picks the magnet/geometry: kresling (default) or cube_robot (IEEE APEC demo).

Run with: uv run python scripts/characterise_sensor_arrays.py [--target kresling|cube_robot]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kresling_sim import cube_robot_config, fisher, kresling_2026_supp_config
from kresling_sim.config import SENSOR_PCB_GAP_MM, WORKSPACE_X_MM, WORKSPACE_Y_MM, Z_DEPLOYED_MM
from kresling_sim.montecarlo import DEFAULT_NOISE, run_monte_carlo
from kresling_sim.sensors import positions_array, save_sensors, sensor_grid

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

# test grid: 5x5 xy x 4 z = 100 magnet positions across the workspace
N_XY = 5
N_Z = 4
TEST_XS = np.linspace(-WORKSPACE_X_MM / 2, WORKSPACE_X_MM / 2, N_XY)
TEST_YS = np.linspace(-WORKSPACE_Y_MM / 2, WORKSPACE_Y_MM / 2, N_XY)

# row-major (y outer, x inner) to match meshgrid
_gx, _gy = np.meshgrid(TEST_XS, TEST_YS)
XY_GRID = np.column_stack([_gx.ravel(), _gy.ravel()])  # (25, 2)

# config space to sweep
COUNTS = list(range(5, 11))       # 5-10
PITCHES = list(range(12, 25, 2))  # 12-24mm
WARM_START_SIGMA = 3.0            # mm, rough tracking-mode uncertainty


def run_config(
    side: int,
    pitch: float,
    n_trials: int,
    rng: np.random.Generator,
    outdir: str,
    target_name: str,
    test_zs: np.ndarray,
    magnet_kwargs: dict | None,
) -> tuple[float, float]:
    """run one (side x side @ pitch mm) config. returns (agg_mc_rmse, agg_crb) in mm."""
    os.makedirs(outdir, exist_ok=True)

    # build + save sensor grid
    sensors = sensor_grid(num_across=side, num_below=side,
                          sep_horizontal=pitch, sep_vertical=pitch)
    points = positions_array(sensors)
    save_sensors(sensors, os.path.join(outdir, "sensor_positions"))
    np.savetxt(os.path.join(outdir, "sensor_positions.txt"), points,
               header="x_mm  y_mm  z_mm", fmt="%.3f")

    # CRB at every test position (analytical, fast)
    crb = np.full((len(XY_GRID), N_Z), np.nan)
    for i_z, z in enumerate(test_zs):
        for i_xy, (x, y) in enumerate(XY_GRID):
            true_pos = np.array([x, y, z])
            jac = fisher.position_jacobian(true_pos, sensors, magnet_kwargs=magnet_kwargs)
            fim = fisher.fisher_information_matrix(jac, DEFAULT_NOISE)
            bound = fisher.crlb(fim)
            crb[i_xy, i_z] = float(np.sqrt(np.sum(bound**2)))

    # warm-start MC at every test position, rmse[i_xy, i_z] is the result at
    # XY_GRID[i_xy] / test_zs[i_z]
    rmse = np.full((len(XY_GRID), N_Z), np.nan)
    total = len(XY_GRID) * N_Z
    done = 0
    for i_z, z in enumerate(test_zs):
        for i_xy, (x, y) in enumerate(XY_GRID):
            true_pos = np.array([x, y, z])
            result = run_monte_carlo(
                true_pos, sensors,
                noise=DEFAULT_NOISE,
                n_trials=n_trials,
                warm_start_sigma=WARM_START_SIGMA,
                rng=rng,
                magnet_kwargs=magnet_kwargs,
            )
            rmse[i_xy, i_z] = result.rmse_mm
            done += 1
            if done % 10 == 0:
                print(f"    {done}/{total} positions done", end="\r")

    print()
    np.savez(os.path.join(outdir, "rmse_data.npz"),
             rmse=rmse, crb=crb, xy_grid=XY_GRID,
             test_xs=TEST_XS, test_ys=TEST_YS, test_zs=test_zs)

    _plot_rmse_vs_x(rmse, crb, outdir, side, pitch, target_name, test_zs)
    _plot_rmse_vs_z(rmse, crb, outdir, side, pitch, target_name, test_zs)
    _plot_heatmap(rmse, crb, outdir, side, pitch, target_name)

    return float(np.nanmean(rmse)), float(np.nanmean(crb))


def _plot_rmse_vs_x(rmse: np.ndarray, crb: np.ndarray, outdir: str, side: int, pitch: float,
                     target_name: str, test_zs: np.ndarray) -> None:
    """MC RMSE and CRB vs x, one line per z level, averaged over y."""
    rmse_grid = rmse.reshape(N_XY, N_XY, N_Z)   # (y, x, z)
    crb_grid  = crb.reshape(N_XY, N_XY, N_Z)
    rmse_vs_x = rmse_grid.mean(axis=0)           # (x, z) averaged over y
    crb_vs_x  = crb_grid.mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, N_Z))
    for i_z, (z, color) in enumerate(zip(test_zs, colors)):
        ax.plot(TEST_XS, rmse_vs_x[:, i_z], "o-",  color=color, label=f"MC z={z:.1f}mm")
        ax.plot(TEST_XS, crb_vs_x[:, i_z],  "--",  color=color, alpha=0.5, label=f"CRB z={z:.1f}mm")

    ax.set_xlabel("x position (mm)")
    ax.set_ylabel("error (mm)")
    ax.set_title(f"{side}x{side} @ {pitch:.0f}mm ({target_name}) — RMSE vs x  (solid=MC, dashed=CRB, mean over y)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "rmse_vs_x.png"), dpi=130)
    plt.close(fig)


def _plot_rmse_vs_z(rmse: np.ndarray, crb: np.ndarray, outdir: str, side: int, pitch: float,
                     target_name: str, test_zs: np.ndarray) -> None:
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
        (line,) = ax.plot(test_zs, mc, f"{marker}-", label=f"MC {label}")
        ax.plot(test_zs, crb_vals, "--", color=line.get_color(), alpha=0.5, label=f"CRB {label}")

    ax.set_xlabel("z — magnet height (mm)")
    ax.set_ylabel("error (mm)")
    ax.set_title(f"{side}x{side} @ {pitch:.0f}mm ({target_name}) — RMSE vs z  (solid=MC, dashed=CRB)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "rmse_vs_z.png"), dpi=130)
    plt.close(fig)


def _plot_heatmap(rmse: np.ndarray, crb: np.ndarray, outdir: str, side: int, pitch: float, target_name: str) -> None:
    """side-by-side heatmaps: MC RMSE and CRB, averaged over z."""
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
        ax.set_title(f"{side}x{side} @ {pitch:.0f}mm ({target_name}) — {title} (mean over z)")
    axes[0].set_ylabel("y (mm)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "heatmap.png"), dpi=130)
    plt.close(fig)


def plot_summary(results: list[dict], outdir: str, target_name: str, suffix: str) -> None:
    """aggregate MC RMSE and CRB vs sensor count, one line per pitch."""
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
    ax.set_title(f"MC RMSE (solid) vs CRB (dashed) across workspace ({target_name})\n(warm-start σ={WARM_START_SIGMA}mm)")
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(outdir, f"summary{suffix}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nsummary plot saved to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", choices=sorted(TARGETS), default="kresling", help="magnet/geometry target (default: kresling)")
    p.add_argument("--n-trials", type=int, default=50, help="MC trials per test position (default: 50)")
    p.add_argument("--outdir", type=str, default=None, help="root output directory (default: scripts/sensor_array/)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.outdir or os.path.join(os.path.dirname(__file__), "sensor_array")
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(42)

    target = TARGETS[args.target]
    z_min, z_max, magnet_kwargs = target["z_min"], target["z_max"], target["magnet_kwargs"]
    test_zs = np.linspace(z_min, z_max, N_Z)
    suffix = "" if args.target == "kresling" else f"_{args.target}"

    n_configs = len(COUNTS) * len(PITCHES)
    print(f"target: {args.target}  (z-range {z_min:.2f}-{z_max:.2f}mm)")
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
            config_name = f"{side}x{side}_{pitch:.0f}mm{suffix}"
            outdir = os.path.join(root, config_name)
            print(f"[{i*len(PITCHES)+j+1}/{n_configs}] {side}x{side} @ {pitch}mm pitch "
                  f"({side*side} sensors, {span:.0f}mm span) → {config_name}/")
            agg_rmse, agg_crb = run_config(side, pitch, args.n_trials, rng, outdir, args.target, test_zs, magnet_kwargs)
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

    plot_summary(results, root, args.target, suffix)


if __name__ == "__main__":
    main()
