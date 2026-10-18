"""CRB sweep over all (count, pitch) configs -> heatmap.
centred square grid is symmetric so only the first quadrant is sampled.
36 test positions (3x3 xy x 4 z), 42 configs. analytical so its fast.

writes scripts/sensor_array/<target>/crb_heatmap[_<target>].png
uv run python scripts/crb_heatmap.py [--target ...]
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
from kresling_sim.montecarlo import DEFAULT_NOISE
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

# first quadrant only (x,y >= 0)
N_XY = 3   # points per axis: [0, 25, 50] mm
N_Z  = 4

TEST_XS = np.linspace(0, WORKSPACE_X_MM / 2, N_XY)
TEST_YS = np.linspace(0, WORKSPACE_Y_MM / 2, N_XY)

_gx, _gy = np.meshgrid(TEST_XS, TEST_YS)
XY_GRID = np.column_stack([_gx.ravel(), _gy.ravel()])        # (9, 2)

COUNTS  = list(range(5, 11))       # 5, 6, 7, 8, 9, 10
PITCHES = list(range(12, 25, 2))   # 12, 14, 16, 18, 20, 22, 24 mm


def compute_crb_matrix(noise, test_positions: np.ndarray, magnet_kwargs: dict | None) -> np.ndarray:
    """(len(COUNTS), len(PITCHES)) array of aggregate CRB RMSE in mm."""
    crb = np.zeros((len(COUNTS), len(PITCHES)))
    for i, side in enumerate(COUNTS):
        for j, pitch in enumerate(PITCHES):
            sensors = sensor_grid(num_across=side, num_below=side,
                                  sep_horizontal=pitch, sep_vertical=pitch)
            rmses = []
            for pos in test_positions:
                jac = fisher.position_jacobian(pos, sensors, magnet_kwargs=magnet_kwargs)
                fim = fisher.fisher_information_matrix(jac, noise)
                bound = fisher.crlb(fim)
                rmses.append(float(np.sqrt(np.sum(bound**2))))
            crb[i, j] = float(np.mean(rmses))
        print(f"  {side}x{side}: {crb[i].round(3).tolist()}")
    return crb


def plot_heatmap(crb: np.ndarray, outpath: str, target_name: str, z_min: float, z_max: float, n_positions: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(crb, aspect="auto", cmap="viridis_r", origin="lower")
    fig.colorbar(im, ax=ax, label="aggregate CRB position-RMS (mm)")

    ax.set_xticks(range(len(PITCHES)))
    ax.set_xticklabels([f"{p}mm" for p in PITCHES])
    ax.set_yticks(range(len(COUNTS)))
    ax.set_yticklabels([f"{c}×{c}" for c in COUNTS])
    ax.set_xlabel("pitch")
    ax.set_ylabel("grid count")
    ax.set_title(
        f"CRB aggregate RMSE — count × pitch ({target_name})\n"
        f"(first-quadrant symmetry, {n_positions} test positions, "
        f"x∈[0,{WORKSPACE_X_MM/2:.0f}]mm y∈[0,{WORKSPACE_Y_MM/2:.0f}]mm "
        f"z∈[{z_min:.2f},{z_max:.2f}]mm)"
    )

    mean_val = crb.mean()
    for i in range(len(COUNTS)):
        for j in range(len(PITCHES)):
            ax.text(j, i, f"{crb[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if crb[i, j] < mean_val else "black")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", choices=sorted(TARGETS), default="kresling", help="magnet/geometry target (default: kresling)")
    p.add_argument("--output", type=str, default=None,
                   help="output PNG path (default: scripts/sensor_array/<target>/crb_heatmap[_<target>].png)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    target = TARGETS[args.target]
    z_min, z_max, magnet_kwargs = target["z_min"], target["z_max"], target["magnet_kwargs"]
    test_zs = np.linspace(z_min, z_max, N_Z)
    test_positions = np.array([[x, y, z] for z in test_zs for x, y in XY_GRID])  # 36 positions total

    suffix = "" if args.target == "kresling" else f"_{args.target}"
    outpath = args.output or os.path.join(
        os.path.dirname(__file__), "sensor_array", args.target, f"crb_heatmap{suffix}.png"
    )
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    print(f"target: {args.target}  (z-range {z_min:.2f}-{z_max:.2f}mm)")
    print(f"CRB sweep: {len(COUNTS)} counts × {len(PITCHES)} pitches = {len(COUNTS)*len(PITCHES)} configs")
    print(f"test positions: {N_XY}x{N_XY} first-quadrant xy × {N_Z} z = {len(test_positions)} positions\n")

    crb = compute_crb_matrix(DEFAULT_NOISE, test_positions, magnet_kwargs)

    best_i, best_j = np.unravel_index(crb.argmin(), crb.shape)
    print(f"\nbest config: {COUNTS[best_i]}x{COUNTS[best_i]} @ {PITCHES[best_j]}mm — CRB={crb.min():.4f}mm")
    print(f"worst config: {COUNTS[np.unravel_index(crb.argmax(), crb.shape)[0]]}x{COUNTS[np.unravel_index(crb.argmax(), crb.shape)[0]]} @ {PITCHES[np.unravel_index(crb.argmax(), crb.shape)[1]]}mm — CRB={crb.max():.4f}mm")

    plot_heatmap(crb, outpath, args.target, z_min, z_max, len(test_positions))
    print(f"\nsaved to {outpath}")

    # also save raw data
    np.savez(outpath.replace(".png", ".npz"),
             crb=crb, counts=COUNTS, pitches=PITCHES,
             test_xs=TEST_XS, test_ys=TEST_YS, test_zs=test_zs)


if __name__ == "__main__":
    main()
