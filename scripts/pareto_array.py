"""pareto curve of sensor count vs position error for kresling_2026_supp.
all 42 configs from the CRB sweep (crb_heatmap npz), MC shortlist overlaid if the json exists.
pareto front = no other config has fewer-or-equal sensors and lower error

uv run python scripts/pareto_array.py [--dir scripts/sensor_array/kresling_sweep_multi_solve]
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRID = "#1f2933", "#6b7785", "#e3e7eb"
ACCENT = "#0d366b"      # front + labels
RAMP = plt.cm.Blues     # pitch, one hue light -> dark


def pareto_front(points: list[dict]) -> list[dict]:
    """configs not beaten on both sensor count and error"""
    front = []
    best = np.inf
    for c in sorted(points, key=lambda c: (c["n"], c["crb"])):
        if c["crb"] < best:
            front.append(c)
            best = c["crb"]
    return front


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=str, default=os.path.join(os.path.dirname(__file__), "sensor_array", "kresling_sweep_multi_solve"))
    args = p.parse_args()

    d = np.load(os.path.join(args.dir, "crb_heatmap_kresling_2026_supp.npz"))
    counts, pitches, crb = list(d["counts"]), list(d["pitches"]), d["crb"]
    configs = [{"side": int(s), "pitch": int(pt), "n": int(s) ** 2, "crb": float(crb[i, j])}
               for i, s in enumerate(counts) for j, pt in enumerate(pitches)]
    front = pareto_front(configs)

    mc_path = os.path.join(args.dir, "mc_shortlist.json")
    mc = json.load(open(mc_path)) if os.path.exists(mc_path) else []

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_facecolor("white")
    norm = plt.Normalize(min(pitches) - 6, max(pitches))
    sc = ax.scatter([c["n"] for c in configs], [c["crb"] for c in configs],
                    c=[c["pitch"] for c in configs], cmap=RAMP, norm=norm, s=46,
                    edgecolors="white", linewidths=1.2, zorder=3)
    ax.step([c["n"] for c in front], [c["crb"] for c in front], where="post", color=ACCENT, lw=2, zorder=2)
    ax.scatter([c["n"] for c in front], [c["crb"] for c in front], s=90, facecolors="none", edgecolors=ACCENT, linewidths=2, zorder=4)
    for c in front:
        ax.annotate(f'{c["side"]}x{c["side"]} @{c["pitch"]}', (c["n"], c["crb"]), textcoords="offset points",
                    xytext=(7, 7), fontsize=8, color=INK)
    if mc:
        ax.scatter([r["n"] for r in mc], [r["mc_mean"] for r in mc], marker="D", s=34, color="#c2410c",
                   edgecolors="white", linewidths=1, zorder=5, label="MC (n_starts=5, 20 trials)")
        ax.legend(loc="upper right", fontsize=8, frameon=False)

    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("pitch (mm)", color=MUTED)
    cb.set_ticks(pitches)
    ax.set_yscale("log")
    ax.set_yticks([0.02, 0.03, 0.05, 0.1, 0.2])
    ax.set_yticklabels(["0.02", "0.03", "0.05", "0.1", "0.2"])
    ax.minorticks_off()
    ax.set_xlim(20, 112)
    ax.set_xlabel("number of sensors", color=MUTED)
    ax.set_ylabel("mean position error (mm), CRB", color=MUTED)
    ax.set_title("kresling_2026_supp: sensors vs error, 42 configs (line = pareto front)", color=INK, fontsize=11, loc="left")
    ax.grid(True, which="both", color=GRID, lw=0.8, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    out = os.path.join(args.dir, "pareto_sensors_vs_error.png")
    fig.savefig(out, dpi=150)
    print("pareto front:")
    for c in front:
        print(f'  {c["side"]}x{c["side"]} @ {c["pitch"]}mm  n={c["n"]:3d}  CRB={c["crb"]:.4f}mm')
    print(f"saved {out}")


if __name__ == "__main__":
    main()
