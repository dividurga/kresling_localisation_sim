"""windowed solve sweep, only the n x n sensors around the initial guess are used (kresling_2026_supp).
sweeps window size n and pitch. full array per pitch spans >= workspace (side = ceil(100/pitch)+1).
per config: CRB (window around the true position) and warm start MC (window around the guess) on a
first quadrant xy grid x z. the solver always sees the whole array, the quadrant only picks test positions.

writes per config npz, then error_vs_n.png, error_vs_pitch.png, montecarlo_spatial_all.png,
crb_spatial_all.png (rows = pitch, cols = n) and summary.md / summary.csv

uv run python scripts/windowed_sweep.py [--workers 12] [--n-trials 20 --n-starts 5]
uv run python scripts/windowed_sweep.py --plot-only
"""
from __future__ import annotations

import argparse
import csv
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # one thread per worker
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from multiprocessing import Pool

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kresling_sim import fisher, kresling_2026_supp_config as cfg
from kresling_sim.config import WORKSPACE_X_MM, WORKSPACE_Y_MM
from kresling_sim.inverse import LeastSquaresSolver
from kresling_sim.montecarlo import DEFAULT_NOISE, run_monte_carlo
from kresling_sim.sensors import positions_array, sensor_grid
from kresling_sim.windowing import WindowedSolver, select_window

NS = [2, 3, 4, 5, 6]
PITCHES = list(range(12, 25, 2))
N_XY = 9
N_Z = 4
WARM_START_SIGMA = 3.0
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "sensor_array", "kresling_sweep_windowed")

INK, MUTED, GRID = "#1f2933", "#6b7785", "#e3e7eb"
BLUES = plt.cm.Blues


def side_for(pitch: float) -> int:
    """sensors per side so the array spans at least the workspace"""
    return math.ceil(WORKSPACE_X_MM / pitch - 1e-9) + 1


def test_grid(n_xy: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(0, WORKSPACE_X_MM / 2, n_xy)
    ys = np.linspace(0, WORKSPACE_Y_MM / 2, n_xy)
    gx, gy = np.meshgrid(xs, ys)
    return xs, ys, np.column_stack([gx.ravel(), gy.ravel()])


def config_dir(root: str, n: int, pitch: int) -> str:
    return os.path.join(root, f"n{n}_pitch{pitch}mm")


def run_config(job: tuple) -> tuple[int, int]:
    n, pitch, n_trials, n_starts, n_xy, seed, root = job
    side = side_for(pitch)
    sensors = sensor_grid(side, side, pitch, pitch)
    positions = positions_array(sensors)
    xs, ys, xy_grid = test_grid(n_xy)
    zs = np.linspace(cfg.Z_MIN_MM, cfg.Z_MAX_MM, N_Z)
    rng = np.random.default_rng(seed)
    solver = WindowedSolver(n, inner=LeastSquaresSolver(n_starts=n_starts))

    rmse = np.full((len(xy_grid), N_Z), np.nan)
    crb = np.full((len(xy_grid), N_Z), np.nan)
    conv = np.full((len(xy_grid), N_Z), np.nan)
    for i_z, z in enumerate(zs):
        for i_xy, (x, y) in enumerate(xy_grid):
            true_pos = np.array([x, y, z])
            # CRB: window around the true position
            idx = select_window(positions, true_pos[:2], n)
            jac = fisher.position_jacobian(true_pos, [sensors[i] for i in idx], magnet_kwargs=cfg.MAGNET_KWARGS)
            bound = fisher.crlb(fisher.fisher_information_matrix(jac, DEFAULT_NOISE))
            crb[i_xy, i_z] = float(np.sqrt(np.sum(bound**2)))
            # MC: window around each trial's guess
            try:
                r = run_monte_carlo(true_pos, sensors, noise=DEFAULT_NOISE, solver=solver, n_trials=n_trials,
                                    warm_start_sigma=WARM_START_SIGMA, rng=rng, magnet_kwargs=cfg.MAGNET_KWARGS)
                rmse[i_xy, i_z] = r.rmse_mm
                conv[i_xy, i_z] = r.converged.mean()
            except RuntimeError:   # every trial failed
                conv[i_xy, i_z] = 0.0

    out = config_dir(root, n, pitch)
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "windowed_data.npz"), rmse=rmse, crb=crb, converged=conv, xy_grid=xy_grid,
             test_xs=xs, test_ys=ys, test_zs=zs, n=n, pitch=pitch, side=side,
             n_trials=n_trials, n_starts=n_starts, warm_start_sigma=WARM_START_SIGMA)
    return n, pitch


def load_all(root: str) -> dict:
    data = {}
    for n in NS:
        for pitch in PITCHES:
            f = os.path.join(config_dir(root, n, pitch), "windowed_data.npz")
            if os.path.exists(f):
                data[(n, pitch)] = dict(np.load(f))
    return data


def _style(ax) -> None:
    ax.grid(True, color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)


def _shade(i: int, count: int) -> tuple:
    return BLUES(0.35 + 0.6 * i / max(count - 1, 1))


def plot_lines(data: dict, root: str) -> None:
    """error vs n (line per pitch) and error vs pitch (line per n), MC solid, CRB dashed"""
    def agg(key, field):
        d = data[key]
        return float(np.nanmean(d[field]))

    pitches = [p for p in PITCHES if any((n, p) in data for n in NS)]
    ns = [n for n in NS if any((n, p) in data for p in PITCHES)]
    for kind, lines, xs_, fmt in (("n", pitches, ns, lambda v: f"{v} mm"), ("pitch", ns, pitches, lambda v: f"{v}x{v}")):
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for i, line in enumerate(lines):
            color = _shade(i, len(lines))
            keys = [((x, line) if kind == "n" else (line, x)) for x in xs_]
            pts = [(x, k) for x, k in zip(xs_, keys) if k in data]
            if not pts:
                continue
            xv = [x for x, _ in pts]
            ax.plot(xv, [agg(k, "rmse") for _, k in pts], "o-", color=color, label=fmt(line))
            ax.plot(xv, [agg(k, "crb") for _, k in pts], "--", color=color, alpha=0.6)
            ax.annotate(fmt(line), (xv[-1], agg(pts[-1][1], "rmse")), textcoords="offset points", xytext=(6, 0),
                        fontsize=8, color=INK, va="center")
        ax.set_yscale("log")
        ax.minorticks_off()
        lo, hi = ax.get_ylim()
        nice = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1, 2, 5]
        ax.set_yticks([t for t in nice if lo <= t <= hi])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("window size n (n x n sensors)" if kind == "n" else "pitch (mm)", color=MUTED)
        ax.set_ylabel("mean position error (mm)", color=MUTED)
        ax.set_xticks(xs_)
        what = "pitch" if kind == "n" else "n"
        ax.set_title(f"windowed solve, error vs {'n' if kind == 'n' else 'pitch'}, one line per {what}  (solid = MC, dashed = CRB)",
                     loc="left", fontsize=10, color=INK)
        ax.legend(title=what, frameon=False, fontsize=8, loc="best")
        _style(ax)
        ax.margins(x=0.12)
        fig.tight_layout()
        fig.savefig(os.path.join(root, f"error_vs_{kind if kind == 'n' else 'pitch'}.png"), dpi=150)
        plt.close(fig)


def plot_spatial_all(data: dict, root: str, field: str, fname: str, label: str) -> None:
    """rows = pitch, cols = n, each cell RMSE(x,y) averaged over z, shared colour scale"""
    pitches = [p for p in PITCHES if any((n, p) in data for n in NS)]
    ns = [n for n in NS if any((n, p) in data for p in PITCHES)]
    maps = {}
    for key, d in data.items():
        n_xy = int(round(math.sqrt(d[field].shape[0])))
        maps[key] = np.nanmean(d[field].reshape(n_xy, n_xy, -1), axis=2)
    allv = np.concatenate([m.ravel() for m in maps.values()])
    vmin, vmax = float(np.nanmin(allv)), float(np.nanpercentile(allv, 95))
    any_d = next(iter(data.values()))
    ext = [any_d["test_xs"][0], any_d["test_xs"][-1], any_d["test_ys"][0], any_d["test_ys"][-1]]

    fig, axes = plt.subplots(len(pitches), len(ns), figsize=(len(ns) * 2.1, len(pitches) * 2.1), squeeze=False)
    for i, p in enumerate(pitches):
        for j, n in enumerate(ns):
            ax = axes[i, j]
            if (n, p) in maps:
                ax.imshow(maps[(n, p)], origin="lower", extent=ext, aspect="equal", cmap="hot_r", vmin=vmin, vmax=vmax)
            else:
                ax.axis("off")
                continue
            ax.set_xticks([0, 25, 50])
            ax.set_yticks([0, 25, 50])
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(f"n = {n}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{p}mm ({side_for(p)}x{side_for(p)})", fontsize=7)
    sm = plt.cm.ScalarMappable(cmap="hot_r", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, ax=axes, label=f"{label} (mm)", shrink=0.6)
    n_trials = int(any_d["n_trials"])
    fig.suptitle(f"{label}(x,y), first quadrant, windowed solve (kresling_2026_supp)  |  rows = pitch (full array), cols = window n\n"
                 f"(z in [{any_d['test_zs'][0]:.2f},{any_d['test_zs'][-1]:.2f}]mm averaged, "
                 f"{n_trials} trials, n_starts={int(any_d['n_starts'])}, warm start sigma={WARM_START_SIGMA}mm)", fontsize=9)
    fig.savefig(os.path.join(root, fname), dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_summary(data: dict, root: str) -> None:
    rows = []
    for (n, p), d in sorted(data.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        rows.append(dict(pitch_mm=p, n=n, full_array=f"{side_for(p)}x{side_for(p)}", window_sensors=n * n,
                         mc_mean_mm=np.nanmean(d["rmse"]), mc_worst_mm=np.nanmax(d["rmse"]),
                         crb_mean_mm=np.nanmean(d["crb"]), crb_worst_mm=np.nanmax(d["crb"]),
                         converged_mean=np.nanmean(d["converged"]), converged_min=np.nanmin(d["converged"])))
    with open(os.path.join(root, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    lines = ["| pitch (mm) | n | full array | MC mean | MC worst | CRB mean | CRB worst | converged mean / min |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['pitch_mm']} | {r['n']} | {r['full_array']} | {r['mc_mean_mm']:.4f} | {r['mc_worst_mm']:.4f} | "
                     f"{r['crb_mean_mm']:.4f} | {r['crb_worst_mm']:.4f} | {r['converged_mean']:.3f} / {r['converged_min']:.2f} |")
    open(os.path.join(root, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def make_plots(root: str) -> None:
    data = load_all(root)
    if not data:
        print("no data found in", root)
        return
    plot_lines(data, root)
    plot_spatial_all(data, root, "rmse", "montecarlo_spatial_all.png", "MC RMSE")
    plot_spatial_all(data, root, "crb", "crb_spatial_all.png", "CRB")
    write_summary(data, root)
    print(f"plots in {root}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-trials", type=int, default=20, help="MC trials per position (default 20)")
    p.add_argument("--n-starts", type=int, default=5, help="solver starts per solve (default 5, multi solve)")
    p.add_argument("--n-xy", type=int, default=N_XY, help=f"first quadrant points per axis (default {N_XY})")
    p.add_argument("--ns", type=int, nargs="+", default=NS, help="window sizes (default 2 3 4 5 6)")
    p.add_argument("--pitches", type=int, nargs="+", default=PITCHES, help="pitches in mm (default 12..24 step 2)")
    p.add_argument("--workers", type=int, default=1, help="parallel configs (default 1)")
    p.add_argument("--seed", type=int, default=42, help="base seed, config k uses seed + k")
    p.add_argument("--outdir", type=str, default=DEFAULT_OUT)
    p.add_argument("--plot-only", action="store_true", help="skip the sweep, just redo plots from saved npz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if not args.plot_only:
        configs = [(n, pitch) for pitch in args.pitches for n in args.ns if n <= side_for(pitch)]
        jobs = [(n, pitch, args.n_trials, args.n_starts, args.n_xy, args.seed + k, args.outdir)
                for k, (n, pitch) in enumerate(configs)]
        print(f"{len(jobs)} configs, {args.n_xy}x{args.n_xy} first quadrant xy x {N_Z} z = {args.n_xy**2 * N_Z} positions each, "
              f"{args.n_trials} trials, n_starts={args.n_starts}, workers={args.workers}", flush=True)
        if args.workers > 1:
            with Pool(args.workers) as pool:
                for done, (n, pitch) in enumerate(pool.imap_unordered(run_config, jobs), 1):
                    print(f"[{done}/{len(jobs)}] n={n} pitch={pitch}mm done", flush=True)
        else:
            for done, job in enumerate(jobs, 1):
                n, pitch = run_config(job)
                print(f"[{done}/{len(jobs)}] n={n} pitch={pitch}mm done", flush=True)
    make_plots(args.outdir)


if __name__ == "__main__":
    main()
