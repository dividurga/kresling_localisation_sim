"""Per-config spatial Monte Carlo RMSE heatmaps across the first-quadrant workspace.

For every (count, pitch) config, runs warm-start Monte Carlo at a dense
first-quadrant grid (x>=0, y>=0) and plots a 2-D heatmap of RMSE(x, y)
averaged over z.  The grid is symmetric under x->-x and y->-y for a centred
square array, so the first quadrant fully characterises the workspace.

Runs against either target magnet/geometry via --target: "kresling" (the
Nature 2025 single-cell magnet, config.py) or "cube_robot" (the IEEE APEC
2025 rolling-cube demo magnet, cube_robot_config.py).

Outputs per config (in scripts/sensor_array/<config>[_<target>]/):
  mc_spatial_heatmap.png   -- 2-D MC RMSE(x,y) averaged over z

One combined figure with all configs in a count x pitch grid:
  scripts/sensor_array/montecarlo_spatial_all[_<target>].png

Runtime: ~10-30 min for 42 configs x 324 positions x 30 trials each.
Use --n-trials to trade speed for accuracy.

Run with: uv run python scripts/mc_spatial_heatmaps.py [--target kresling|cube_robot] [--n-trials 30]
"""
from __future__ import annotations

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kresling_sim import cube_robot_config
from kresling_sim.config import SENSOR_PCB_GAP_MM, WORKSPACE_X_MM, WORKSPACE_Y_MM, Z_DEPLOYED_MM
from kresling_sim.montecarlo import DEFAULT_NOISE, run_monte_carlo
from kresling_sim.sensors import positions_array, sensor_grid

TARGETS = {
    "kresling": {"z_min": SENSOR_PCB_GAP_MM, "z_max": Z_DEPLOYED_MM, "magnet_kwargs": None},
    "cube_robot": {
        "z_min": cube_robot_config.Z_MIN_MM,
        "z_max": cube_robot_config.Z_MAX_MM,
        "magnet_kwargs": cube_robot_config.MAGNET_KWARGS,
    },
}

COUNTS  = list(range(5, 11))       # 5, 6, 7, 8, 9, 10
PITCHES = list(range(12, 25, 2))   # 12, 14, 16, 18, 20, 22, 24 mm

WARM_START_SIGMA = 3.0  # mm

N_XY = 9   # points per axis in the first quadrant -- 9x9 = 81 xy points
N_Z  = 4

TEST_XS = np.linspace(0, WORKSPACE_X_MM / 2, N_XY)
TEST_YS = np.linspace(0, WORKSPACE_Y_MM / 2, N_XY)

_gx, _gy = np.meshgrid(TEST_XS, TEST_YS)
XY_GRID = np.column_stack([_gx.ravel(), _gy.ravel()])   # (81, 2)


def compute_mc_spatial(points: np.ndarray, noise, n_trials: int,
                       rng: np.random.Generator, test_zs: np.ndarray, magnet_kwargs: dict | None) -> np.ndarray:
    """Warm-start MC RMSE at every (xy, z) test point. Returns (N_XY*N_XY, N_Z)."""
    rmse = np.zeros((len(XY_GRID), N_Z))
    total = len(XY_GRID) * N_Z
    done = 0
    for i_z, z in enumerate(test_zs):
        for i_xy, (x, y) in enumerate(XY_GRID):
            result = run_monte_carlo(
                np.array([x, y, z]), points,
                noise=noise,
                n_trials=n_trials,
                warm_start_sigma=WARM_START_SIGMA,
                rng=rng,
                magnet_kwargs=magnet_kwargs,
            )
            rmse[i_xy, i_z] = result.rmse_mm
            done += 1
            if done % 20 == 0:
                print(f"    {done}/{total} positions", end="\r")
    print()
    return rmse


def _rmse_xy_mean(rmse: np.ndarray) -> np.ndarray:
    """(N_XY, N_XY) RMSE averaged over z. Row=y, col=x."""
    return rmse.reshape(N_XY, N_XY, N_Z).mean(axis=2)


def plot_single(rmse_xy: np.ndarray, ax: plt.Axes, side: int, pitch: int,
                vmin: float, vmax: float) -> None:
    extent = [TEST_XS[0], TEST_XS[-1], TEST_YS[0], TEST_YS[-1]]
    ax.imshow(rmse_xy, origin="lower", extent=extent, aspect="equal",
              cmap="hot_r", vmin=vmin, vmax=vmax)
    ax.set_title(f"{side}×{side} @{pitch}mm", fontsize=7)
    ax.set_xticks([0, 25, 50])
    ax.set_yticks([0, 25, 50])
    ax.tick_params(labelsize=6)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", choices=sorted(TARGETS), default="kresling", help="magnet/geometry target (default: kresling)")
    p.add_argument("--n-trials", type=int, default=30,
                   help="MC trials per position (default: 30)")
    p.add_argument("--n-xy", type=int, default=N_XY,
                   help=f"grid points per axis in first quadrant (default: {N_XY})")
    p.add_argument("--outdir", type=str, default=None,
                   help="root output dir (default: scripts/sensor_array/)")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed (default: 42)")
    return p.parse_args()


def main() -> None:
    global N_XY, TEST_XS, TEST_YS, XY_GRID

    args = parse_args()
    N_XY = args.n_xy
    TEST_XS = np.linspace(0, WORKSPACE_X_MM / 2, N_XY)
    TEST_YS = np.linspace(0, WORKSPACE_Y_MM / 2, N_XY)
    _gx, _gy = np.meshgrid(TEST_XS, TEST_YS)
    XY_GRID = np.column_stack([_gx.ravel(), _gy.ravel()])

    target = TARGETS[args.target]
    z_min, z_max, magnet_kwargs = target["z_min"], target["z_max"], target["magnet_kwargs"]
    test_zs = np.linspace(z_min, z_max, N_Z)
    suffix = "" if args.target == "kresling" else f"_{args.target}"

    root = args.outdir or os.path.join(os.path.dirname(__file__), "sensor_array")
    os.makedirs(root, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    n_configs = len(COUNTS) * len(PITCHES)
    total_solves = n_configs * N_XY * N_XY * N_Z * args.n_trials
    print(f"target: {args.target}  (z-range {z_min:.2f}-{z_max:.2f}mm)")
    print(f"MC spatial heatmaps: {n_configs} configs")
    print(f"grid: {N_XY}×{N_XY} first-quadrant xy × {N_Z} z = {N_XY*N_XY*N_Z} positions per config")
    print(f"trials per position: {args.n_trials}  |  total solver runs: {total_solves:,}")
    print(f"warm-start sigma: {WARM_START_SIGMA}mm\n")

    noise = DEFAULT_NOISE
    t0 = time.time()

    all_rmse_xy = {}
    for i, side in enumerate(COUNTS):
        for j, pitch in enumerate(PITCHES):
            cfg_t0 = time.time()
            sensors = sensor_grid(num_across=side, num_below=side,
                                  sep_horizontal=pitch, sep_vertical=pitch)
            points = positions_array(sensors)
            print(f"  [{i*len(PITCHES)+j+1}/{n_configs}] {side}×{side} @{pitch}mm ...")
            rmse = compute_mc_spatial(points, noise, args.n_trials, rng, test_zs, magnet_kwargs)
            rmse_xy = _rmse_xy_mean(rmse)
            all_rmse_xy[(side, pitch)] = rmse_xy

            config_dir = os.path.join(root, f"{side}x{side}_{pitch}mm{suffix}")
            os.makedirs(config_dir, exist_ok=True)
            np.savez(os.path.join(config_dir, "mc_spatial.npz"),
                     rmse=rmse, rmse_xy=rmse_xy,
                     test_xs=TEST_XS, test_ys=TEST_YS, test_zs=test_zs,
                     n_trials=args.n_trials, warm_start_sigma=WARM_START_SIGMA)

            elapsed = time.time() - cfg_t0
            print(f"         mean={rmse_xy.mean():.3f}mm  max={rmse_xy.max():.3f}mm  ({elapsed:.1f}s)")

    elapsed_total = time.time() - t0
    print(f"\nall configs done in {elapsed_total/60:.1f} min")

    vmin = min(v.min() for v in all_rmse_xy.values())
    vmax = np.percentile(np.concatenate([v.ravel() for v in all_rmse_xy.values()]), 95)

    # --- individual heatmaps (per-config colour scale) ---
    for (side, pitch), rmse_xy in all_rmse_xy.items():
        config_dir = os.path.join(root, f"{side}x{side}_{pitch}mm{suffix}")
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(rmse_xy, origin="lower",
                       extent=[TEST_XS[0], TEST_XS[-1], TEST_YS[0], TEST_YS[-1]],
                       aspect="equal", cmap="hot_r")
        fig.colorbar(im, ax=ax, label="MC RMSE (mm)")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(f"{side}×{side} @ {pitch}mm pitch ({args.target}) — MC RMSE(x,y) mean over z\n"
                     f"({N_XY}×{N_XY} first-quadrant grid, {args.n_trials} trials, σ={WARM_START_SIGMA}mm)")
        fig.tight_layout()
        fig.savefig(os.path.join(config_dir, "mc_spatial_heatmap.png"), dpi=130)
        plt.close(fig)

    # --- combined figure: all 42 configs, shared colour scale ---
    fig, axes = plt.subplots(len(COUNTS), len(PITCHES),
                             figsize=(len(PITCHES) * 2.1, len(COUNTS) * 2.1))
    for i, side in enumerate(COUNTS):
        for j, pitch in enumerate(PITCHES):
            plot_single(all_rmse_xy[(side, pitch)], axes[i, j],
                        side, pitch, vmin, vmax)
            if i == len(COUNTS) - 1:
                axes[i, j].set_xlabel(f"{pitch}mm", fontsize=7)
            if j == 0:
                axes[i, j].set_ylabel(f"{side}×{side}", fontsize=7)

    sm = plt.cm.ScalarMappable(cmap="hot_r",
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, ax=axes, label="MC RMSE (mm)", shrink=0.6)
    fig.suptitle(
        f"MC RMSE(x,y) — first-quadrant workspace ({args.target})  |  rows=count, cols=pitch\n"
        f"({N_XY}×{N_XY} grid, x∈[0,{WORKSPACE_X_MM/2:.0f}]mm, "
        f"z∈[{z_min:.2f},{z_max:.2f}]mm, averaged over z, "
        f"{args.n_trials} trials, warm-start σ={WARM_START_SIGMA}mm)",
        fontsize=9,
    )
    combined_path = os.path.join(root, f"montecarlo_spatial_all{suffix}.png")
    fig.savefig(combined_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"combined heatmap saved to {combined_path}")
    print(f"individual heatmaps saved in each config subfolder")


if __name__ == "__main__":
    main()
