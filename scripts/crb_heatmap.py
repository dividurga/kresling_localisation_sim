"""CRB sweep across all (count, pitch) configs — generates a heatmap.

Exploits the symmetry of a centred square sensor grid: CRB at (-x, y, z)
equals CRB at (x, y, z), so only the first quadrant (x>=0, y>=0) needs to
be sampled. The first-quadrant average equals the full-workspace average.

Test positions: 3x3 first-quadrant xy grid × 4 z levels = 36 positions.
Config space: counts 5-10, pitches 12-24mm in 2mm steps (42 configs total).
CRB is analytical — no solver runs — so this completes in seconds.

Writes: scripts/sensor_array/crb_heatmap.png

Run with: uv run python scripts/crb_heatmap.py
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kresling_sim import fisher
from kresling_sim.config import (
    SENSOR_PCB_GAP_MM,
    WORKSPACE_X_MM,
    WORKSPACE_Y_MM,
    Z_DEPLOYED_MM,
)
from kresling_sim.montecarlo import DEFAULT_NOISE
from kresling_sim.sensors import positions_array, sensor_grid

# --- symmetry-exploited test grid ---
# First quadrant only: x in [0, WORKSPACE/2], y in [0, WORKSPACE/2].
# A centred square sensor grid is symmetric under x->-x and y->-y, so
# the first-quadrant CRB average equals the full-workspace average.
N_XY = 3   # points per axis: [0, 25, 50] mm
N_Z  = 4

TEST_XS = np.linspace(0, WORKSPACE_X_MM / 2, N_XY)
TEST_YS = np.linspace(0, WORKSPACE_Y_MM / 2, N_XY)
TEST_ZS = np.linspace(SENSOR_PCB_GAP_MM, Z_DEPLOYED_MM, N_Z)

_gx, _gy = np.meshgrid(TEST_XS, TEST_YS)
XY_GRID = np.column_stack([_gx.ravel(), _gy.ravel()])        # (9, 2)
TEST_POSITIONS = np.array(
    [[x, y, z] for z in TEST_ZS for x, y in XY_GRID]
)  # 36 positions total

COUNTS  = list(range(5, 11))       # 5, 6, 7, 8, 9, 10
PITCHES = list(range(12, 25, 2))   # 12, 14, 16, 18, 20, 22, 24 mm


def compute_crb_matrix(noise) -> np.ndarray:
    """(len(COUNTS), len(PITCHES)) array of aggregate CRB RMSE in mm."""
    crb = np.zeros((len(COUNTS), len(PITCHES)))
    for i, side in enumerate(COUNTS):
        for j, pitch in enumerate(PITCHES):
            sensors = sensor_grid(num_across=side, num_below=side,
                                  sep_horizontal=pitch, sep_vertical=pitch)
            points = positions_array(sensors)
            rmses = []
            for pos in TEST_POSITIONS:
                jac = fisher.position_jacobian(pos, points)
                fim = fisher.fisher_information_matrix(jac, noise)
                bound = fisher.crlb(fim)
                rmses.append(float(np.sqrt(np.sum(bound**2))))
            crb[i, j] = float(np.mean(rmses))
        print(f"  {side}x{side}: {crb[i].round(3).tolist()}")
    return crb


def plot_heatmap(crb: np.ndarray, outpath: str) -> None:
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
        "CRB aggregate RMSE — count × pitch\n"
        f"(first-quadrant symmetry, {len(TEST_POSITIONS)} test positions, "
        f"x∈[0,{WORKSPACE_X_MM/2:.0f}]mm y∈[0,{WORKSPACE_Y_MM/2:.0f}]mm "
        f"z∈[{SENSOR_PCB_GAP_MM:.1f},{Z_DEPLOYED_MM:.1f}]mm)"
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
    p.add_argument("--output", type=str, default=None,
                   help="output PNG path (default: scripts/sensor_array/crb_heatmap.png)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outpath = args.output or os.path.join(
        os.path.dirname(__file__), "sensor_array", "crb_heatmap.png"
    )
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    print(f"CRB sweep: {len(COUNTS)} counts × {len(PITCHES)} pitches = {len(COUNTS)*len(PITCHES)} configs")
    print(f"test positions: {N_XY}x{N_XY} first-quadrant xy × {N_Z} z = {len(TEST_POSITIONS)} positions\n")

    crb = compute_crb_matrix(DEFAULT_NOISE)

    best_i, best_j = np.unravel_index(crb.argmin(), crb.shape)
    print(f"\nbest config: {COUNTS[best_i]}x{COUNTS[best_i]} @ {PITCHES[best_j]}mm — CRB={crb.min():.4f}mm")
    print(f"worst config: {COUNTS[np.unravel_index(crb.argmax(), crb.shape)[0]]}x{COUNTS[np.unravel_index(crb.argmax(), crb.shape)[0]]} @ {PITCHES[np.unravel_index(crb.argmax(), crb.shape)[1]]}mm — CRB={crb.max():.4f}mm")

    plot_heatmap(crb, outpath)
    print(f"\nsaved to {outpath}")

    # also save raw data
    np.savez(outpath.replace(".png", ".npz"),
             crb=crb, counts=COUNTS, pitches=PITCHES,
             test_xs=TEST_XS, test_ys=TEST_YS, test_zs=TEST_ZS)


if __name__ == "__main__":
    main()
