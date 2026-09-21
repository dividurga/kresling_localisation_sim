"""per-config spatial CRB heatmaps across the first-quadrant workspace.

For each (count, pitch) config, computes CRB(x,y) averaged over z on a
dense first-quadrant grid (symmetric for a centred square array, so one
quadrant is enough) and plots it. Saves one heatmap per config plus a
combined grid of all configs. Analytical, so it's fast. --target picks
kresling or cube_robot.

Outputs: scripts/sensor_array/<config>[_<target>]/crb_spatial_heatmap.png
         scripts/sensor_array/crb_spatial_all[_<target>].png

Run with: uv run python scripts/crb_spatial_heatmaps.py [--target kresling|cube_robot]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kresling_sim import cube_robot_config, fisher
from kresling_sim.config import SENSOR_PCB_GAP_MM, WORKSPACE_X_MM, WORKSPACE_Y_MM, Z_DEPLOYED_MM
from kresling_sim.montecarlo import DEFAULT_NOISE
from kresling_sim.sensors import sensor_grid

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

N_XY = 9   # points per axis in the first quadrant -- 9x9 = 81 xy points
N_Z  = 4

TEST_XS = np.linspace(0, WORKSPACE_X_MM / 2, N_XY)
TEST_YS = np.linspace(0, WORKSPACE_Y_MM / 2, N_XY)

_gx, _gy = np.meshgrid(TEST_XS, TEST_YS)
XY_GRID = np.column_stack([_gx.ravel(), _gy.ravel()])   # (81, 2)


def compute_crb_spatial(sensors, noise, test_zs: np.ndarray, magnet_kwargs: dict | None) -> np.ndarray:
    """CRB RMSE at every (xy, z) test point. Returns (N_XY*N_XY, N_Z)."""
    crb = np.zeros((len(XY_GRID), N_Z))
    for i_z, z in enumerate(test_zs):
        for i_xy, (x, y) in enumerate(XY_GRID):
            jac = fisher.position_jacobian(np.array([x, y, z]), sensors, magnet_kwargs=magnet_kwargs)
            fim = fisher.fisher_information_matrix(jac, noise)
            bound = fisher.crlb(fim)
            crb[i_xy, i_z] = float(np.sqrt(np.sum(bound**2)))
    return crb


def _crb_xy_mean(crb: np.ndarray) -> np.ndarray:
    """(N_XY, N_XY) CRB averaged over z. Row=y, col=x."""
    return crb.reshape(N_XY, N_XY, N_Z).mean(axis=2)


def plot_single(crb_xy: np.ndarray, ax: plt.Axes, side: int, pitch: int,
                vmin: float, vmax: float) -> None:
    extent = [TEST_XS[0], TEST_XS[-1], TEST_YS[0], TEST_YS[-1]]
    ax.imshow(crb_xy, origin="lower", extent=extent, aspect="equal",
              cmap="viridis_r", vmin=vmin, vmax=vmax)
    ax.set_title(f"{side}×{side} @{pitch}mm", fontsize=7)
    ax.set_xticks([0, 25, 50])
    ax.set_yticks([0, 25, 50])
    ax.tick_params(labelsize=6)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", choices=sorted(TARGETS), default="kresling", help="magnet/geometry target (default: kresling)")
    p.add_argument("--n-xy", type=int, default=N_XY,
                   help=f"grid points per axis in first quadrant (default: {N_XY})")
    p.add_argument("--outdir", type=str, default=None,
                   help="root output dir (default: scripts/sensor_array/)")
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

    n_configs = len(COUNTS) * len(PITCHES)
    print(f"target: {args.target}  (z-range {z_min:.2f}-{z_max:.2f}mm)")
    print(f"spatial CRB heatmaps: {n_configs} configs")
    print(f"grid: {N_XY}×{N_XY} first-quadrant xy × {N_Z} z = {N_XY*N_XY*N_Z} positions per config\n")

    noise = DEFAULT_NOISE

    # collect all crb_xy arrays and find global vmin/vmax for shared colorscale
    all_crb_xy = {}
    for i, side in enumerate(COUNTS):
        for j, pitch in enumerate(PITCHES):
            sensors = sensor_grid(num_across=side, num_below=side,
                                  sep_horizontal=pitch, sep_vertical=pitch)
            crb = compute_crb_spatial(sensors, noise, test_zs, magnet_kwargs)
            crb_xy = _crb_xy_mean(crb)
            all_crb_xy[(side, pitch)] = crb_xy

            # save individual plot and data
            config_dir = os.path.join(root, f"{side}x{side}_{pitch}mm{suffix}")
            os.makedirs(config_dir, exist_ok=True)
            np.savez(os.path.join(config_dir, "crb_spatial.npz"),
                     crb=crb, crb_xy=crb_xy,
                     test_xs=TEST_XS, test_ys=TEST_YS, test_zs=test_zs)
            print(f"  [{i*len(PITCHES)+j+1}/{n_configs}] {side}×{side} @{pitch}mm  "
                  f"mean={crb_xy.mean():.3f}mm  max={crb_xy.max():.3f}mm")

    # global color scale across all configs
    vmin = min(v.min() for v in all_crb_xy.values())
    vmax = np.percentile(np.concatenate([v.ravel() for v in all_crb_xy.values()]), 95)

    # individual heatmaps, per-config colour scale
    for (side, pitch), crb_xy in all_crb_xy.items():
        config_dir = os.path.join(root, f"{side}x{side}_{pitch}mm{suffix}")
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(crb_xy, origin="lower",
                       extent=[TEST_XS[0], TEST_XS[-1], TEST_YS[0], TEST_YS[-1]],
                       aspect="equal", cmap="viridis_r")
        fig.colorbar(im, ax=ax, label="CRB position-RMS (mm)")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(f"{side}×{side} @ {pitch}mm pitch ({args.target}) — CRB(x,y) mean over z\n"
                     f"({N_XY}×{N_XY} first-quadrant grid)")
        fig.tight_layout()
        fig.savefig(os.path.join(config_dir, "crb_spatial_heatmap.png"), dpi=130)
        plt.close(fig)

    # combined figure: all 42 configs, shared colour scale
    fig, axes = plt.subplots(len(COUNTS), len(PITCHES),
                             figsize=(len(PITCHES) * 2.1, len(COUNTS) * 2.1))
    for i, side in enumerate(COUNTS):
        for j, pitch in enumerate(PITCHES):
            plot_single(all_crb_xy[(side, pitch)], axes[i, j],
                        side, pitch, vmin, vmax)
            if i == len(COUNTS) - 1:
                axes[i, j].set_xlabel(f"{pitch}mm", fontsize=7)
            if j == 0:
                axes[i, j].set_ylabel(f"{side}×{side}", fontsize=7)

    # shared colourbar
    sm = plt.cm.ScalarMappable(cmap="viridis_r",
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, ax=axes, label="CRB position-RMS (mm)", shrink=0.6)
    fig.suptitle(
        f"CRB(x,y) — first-quadrant workspace ({args.target})  |  rows=count, cols=pitch\n"
        f"({N_XY}×{N_XY} grid, x∈[0,{WORKSPACE_X_MM/2:.0f}]mm, "
        f"z∈[{z_min:.2f},{z_max:.2f}]mm, averaged over z)",
        fontsize=9,
    )
    combined_path = os.path.join(root, f"crb_spatial_all{suffix}.png")
    fig.savefig(combined_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\ncombined heatmap saved to {combined_path}")
    print(f"individual heatmaps saved in each config subfolder")


if __name__ == "__main__":
    main()
