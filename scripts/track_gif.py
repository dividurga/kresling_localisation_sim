"""gif of the rolling robot tracking loop (kresling_2026_supp). small render of the setup, coil field at the sensors,
the sense cycle (coils, conversion, readout, solve) and true vs reconstructed position with timestamps.
sim time is slowed down, one frame = 1 / fps / slow seconds of sim time

uv run python scripts/track_gif.py [--speed 20] [--span 1.0] [--slow 10] [--fps 20] [--solve-ms 14]
uv run python scripts/track_gif.py --still 0.5     # one frame at t = 0.5 s as a png, to check the layout
"""
from __future__ import annotations

import argparse
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from conversion_time_experiment import residual_curves
from kresling_sim import kresling_2026_supp_config as cfg
from kresling_sim.actuation import Actuation
from kresling_sim.config import SENSOR_PCB_GAP_MM, WORKSPACE_X_MM, WORKSPACE_Y_MM
from kresling_sim.inverse import LeastSquaresSolver
from kresling_sim.montecarlo import DEFAULT_NOISE
from kresling_sim.sensors import positions_array, sensor_grid
from kresling_sim.tracking import FACE_ANGLE, HEX_SIDE_MM, CycleTiming, RollingTrajectory, readout_time_s, track
from kresling_sim.windowing import WindowedSolver

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "tracking", "track_rolling.gif")
INK, MUTED, GRID = "#1f2933", "#6b7785", "#e3e7eb"
AXIS = {"x": "#2a78d6", "y": "#eb6834", "z": "#1baf7a"}  # same axis, same color in every panel
EST = "#6250d6"  # latest estimate pin in the render
SLATE, SLATE_LIGHT = "#52606d", "#9aa5b1"
LANES = {"coils": 3, "convert": 2, "readout": 1, "solve": 0}
RISE_S = 5e-3
CURRENT_A = 20.0


def coil_current_curve(timing: CycleTiming, t_grid: np.ndarray, n_cycles: int) -> np.ndarray:
    """coil current / commanded current (0..1), first order, off at t_off(k), back on at the next cycle start"""
    act = Actuation(None, rise_time_s=RISE_S)
    act.command({"i_xf": 1.0}, t=-1.0)  # settled on before t = 0
    for k in range(n_cycles):
        act.command({"i_xf": 0.0}, t=timing.t_off(k))
        act.command({"i_xf": 1.0}, t=(k + 1) * timing.period_s)
    return np.array([act.currents_at(t)["i_xf"] for t in t_grid])


def prism_polys(center: np.ndarray, phi: float) -> list[np.ndarray]:
    """hex prism (axis along y, 12 mm tall) rolled by phi plus the two magnet discs, 2 caps + 6 sides + 2 discs"""
    cx, cy, cz = center
    ang = np.arange(6) * FACE_ANGLE
    loc = HEX_SIDE_MM * np.stack([np.cos(ang), np.sin(ang)], axis=1)  # (x, z) in the prism frame
    c, s = math.cos(phi), math.sin(phi)
    rx = loc[:, 0] * c + loc[:, 1] * s  # clockwise seen from -y, same as rolling toward +x
    rz = -loc[:, 0] * s + loc[:, 1] * c
    half = cfg.MAGNET_SEPARATION_MM / 2
    y_lo, y_hi = cy - half, cy + half
    ring = lambda y: np.column_stack([cx + rx, np.full(6, y), cz + rz])
    polys = [ring(y_lo), ring(y_hi)]
    for i in range(6):
        j = (i + 1) % 6
        polys.append(np.array([[cx + rx[i], y_lo, cz + rz[i]], [cx + rx[j], y_lo, cz + rz[j]],
                               [cx + rx[j], y_hi, cz + rz[j]], [cx + rx[i], y_hi, cz + rz[i]]]))
    t = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    r = cfg.MAGNET_DIAMETER_MM / 2
    for y in (y_lo - 0.1, y_hi + 0.1):  # a hair outside the cap so the disc shows
        polys.append(np.column_stack([cx + r * np.cos(t), np.full(20, y), cz + r * np.sin(t)]))
    return polys


def _style(ax) -> None:
    ax.grid(True, color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)


def build(args):
    """run the tracker, return everything the frames need"""
    timing = CycleTiming(trg_margin=args.margin,
                         readout_s=readout_time_s(args.count**2, args.buses, args.i2c_khz * 1e3))
    traj = RollingTrajectory(args.speed)
    sensors = sensor_grid(args.count, args.count, args.pitch, args.pitch)
    solver = WindowedSolver(args.window, inner=LeastSquaresSolver()) if args.solver == "windowed" else LeastSquaresSolver()
    res = track(traj, sensors, solver, args.span, timing=timing, solve_time_s=args.solve_ms * 1e-3,
                rng=np.random.default_rng(args.seed))
    t_end = max(args.span, float(res.t_ready[-1])) + 0.005
    return timing, traj, sensors, res, t_end


def make_figure(args, timing, traj, sensors, res, t_end):
    """static artists once, returns update(t) that moves the cursors and the moving artists"""
    ms = 1e3
    P = timing.period_s
    n_cyc = res.n_samples + 2
    fig = plt.figure(figsize=(10.5, 7.0), dpi=args.dpi, facecolor="white")
    gs = GridSpec(9, 2, figure=fig, width_ratios=[1, 1.15], left=0.07, right=0.985, top=0.87, bottom=0.07,
                  wspace=0.17, hspace=1.1)
    ax3 = fig.add_subplot(gs[0:4, 0], projection="3d", computed_zorder=False)  # draw order = zorder, not depth
    ax_f = fig.add_subplot(gs[4:6, 0])
    ax_g = fig.add_subplot(gs[6:9, 0], sharex=ax_f)
    ax_p = [fig.add_subplot(gs[0:3, 1]), fig.add_subplot(gs[3:6, 1]), fig.add_subplot(gs[6:9, 1])]

    # --- 3d render
    half_x, half_y = WORKSPACE_X_MM / 2, WORKSPACE_Y_MM / 2
    board = [np.array([[-half_x, -half_y, 0], [half_x, -half_y, 0], [half_x, half_y, 0], [-half_x, half_y, 0]])]
    ax3.add_collection3d(Poly3DCollection(board, facecolors="#eef1f4", edgecolors=GRID, linewidths=1, zorder=1))
    sp = positions_array(sensors)
    ax3.scatter(sp[:, 0], sp[:, 1], np.full(len(sp), 0.3), marker="s", s=9, color=SLATE_LIGHT, depthshade=False, zorder=2)
    surface = [b + np.array([0, 0, SENSOR_PCB_GAP_MM]) for b in board]  # what the prism rolls on
    ax3.add_collection3d(Poly3DCollection(surface, facecolors="#dbe4ee", edgecolors=GRID, linewidths=0.8, alpha=0.35,
                                          zorder=3))
    colors = ["#aeb9c6"] * 2 + ["#d3dbe4"] * 6 + [SLATE] * 2
    prism = Poly3DCollection(prism_polys(traj.position(0.0), float(traj.phi(0.0))), facecolors=colors,
                             edgecolors=SLATE, linewidths=0.6, zorder=5)
    ax3.add_collection3d(prism)
    pin, = ax3.plot([0, 0], [0, 0], [0, 0], color=EST, lw=1.6, zorder=6)
    ring, = ax3.plot([0], [0], [0], "o", mfc="none", mec=EST, mew=2, ms=9, zorder=7)
    ax3.set_xlim(-half_x - 5, half_x + 5)
    ax3.set_ylim(-half_y - 5, half_y + 5)
    ax3.set_zlim(0, 28)
    ax3.set_box_aspect((1, 1, 0.3), zoom=1.5)
    ax3.view_init(elev=30, azim=-68)
    ax3.set_axis_off()
    ax3.set_title("setup: 6x6 sensors @ 18 mm, hex prism rolling along x", loc="left", fontsize=8.5, color=INK)
    ax3.text2D(0.0, 0.93, "violet pin = latest estimate that has arrived", transform=ax3.transAxes, fontsize=7.5,
               color=EST, zorder=20)
    hud = ax3.text2D(0.0, 0.0, "", transform=ax3.transAxes, fontsize=7.5, color=INK, family="monospace", zorder=20,
                     bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5))

    # --- coil field at the sensors, worst axis, each axis driven alone at 20 A as in the settle wait analysis
    t_grid = np.arange(-0.02, t_end + 0.12, 0.25e-3)
    cur = coil_current_curve(timing, t_grid, n_cyc)
    b_on = residual_curves(CURRENT_A, RISE_S, positions_array(sensors), np.array([0.0]))[:, 0] * 1e3  # uT per axis
    field = np.maximum(b_on.max() * cur, 1e-2)
    ax_f.semilogy(t_grid * ms, field, color=INK, lw=1.6)
    sig = DEFAULT_NOISE.sigma * 1e3  # uT
    for val, lab, va in ((sig[0] / 3, "sigma/3 x,y", "top"), (sig[2] / 3, "sigma/3 z", "bottom")):
        ax_f.axhline(val, color=MUTED, ls="--", lw=1)
        ax_f.text(0.995, val, lab, transform=ax_f.get_yaxis_transform(), ha="right", va=va, fontsize=7, color=MUTED)
    ax_f.set_ylim(0.1, 6e4)
    ax_f.set_yticks([1, 100, 1e4])
    ax_f.set_yticklabels(["1", "100", "10^4"])
    ax_f.set_ylabel("uT", fontsize=8, color=MUTED)
    ax_f.set_title(f"actuation field at sensors, worst axis ({CURRENT_A:g} A, {RISE_S * 1e3:g} ms fall)",
                   loc="left", fontsize=8.5, color=INK)
    plt.setp(ax_f.get_xticklabels(), visible=False)

    # --- sense cycle lanes
    def bar(lane, t0, t1, fc, ec="white", hatch=None, label=None, lw=1.0):
        ax_g.add_patch(Rectangle((t0 * ms, LANES[lane] - 0.36), (t1 - t0) * ms, 0.72, facecolor=fc, edgecolor=ec,
                                 hatch=hatch, linewidth=lw))
        if label:
            ax_g.text((t0 + t1) / 2 * ms, LANES[lane], label, ha="center", va="center", fontsize=6.5, clip_on=True,
                      color="white" if fc not in ("white",) and hatch is None else INK)

    d = timing.conv_s / 3
    for k in range(n_cyc):
        bar("coils", k * P, timing.t_off(k), SLATE, label="on")
        bar("coils", timing.t_off(k), timing.t_conv_start(k), "white", ec=SLATE_LIGHT, hatch="////", label="settle", lw=0.0)
        for a, axis in enumerate("xyz"):
            bar("convert", timing.t_conv_start(k) + a * d, timing.t_conv_start(k) + (a + 1) * d, AXIS[axis],
                label=axis.upper())
        t_conv_end = timing.t_conv_start(k) + timing.conv_s
        bar("convert", t_conv_end, t_conv_end + timing.conv_s * timing.trg_margin, "white", ec=SLATE_LIGHT, hatch="////", lw=0.0)
        if k < res.n_samples + 1:
            bar("readout", (k + 1) * P, (k + 1) * P + timing.readout_s, SLATE_LIGHT)
    for t0, t1 in zip(res.t_start, res.t_ready):
        bar("solve", t0, t1, INK, label="solve")
    ax_g.set_yticks(list(LANES.values()))
    ax_g.set_yticklabels(list(LANES), fontsize=7.5, color=MUTED)
    ax_g.set_ylim(-0.6, 3.6)
    ax_g.set_xlabel("time (ms)", fontsize=8, color=MUTED)
    ax_g.set_title(f"sense cycle {P * ms:.1f} ms: conversion {timing.conv_s * ms:g} ms (+{timing.trg_margin:.0%} wait, "
                   f"hatched), i2c readout {timing.readout_s * ms:.2f} ms, solve {args.solve_ms:g} ms",
                   loc="left", fontsize=8.5, color=INK)
    _style(ax_f)
    _style(ax_g)
    ax_g.grid(False, axis="y")

    # --- true vs reconstructed position
    tg = np.arange(0, t_end, 1e-3)
    true_pos = traj.position(tg)
    limits = [(true_pos[:, 0].min() - 2, true_pos[:, 0].max() + 2), (-0.12, 0.12), (12.2, 13.4)]
    lines_true, dots_sample, dots_arrive, cursors = [], [], [], []
    ok = res.ok
    for a, axis in enumerate("xyz"):
        ax = ax_p[a]
        c = AXIS[axis]
        lt, = ax.plot([], [], color=INK, lw=1.4)
        ds, = ax.plot([], [], "o", color=c, ms=4.5, mew=0)
        da, = ax.plot([], [], "o", mfc="none", mec=c, ms=7, mew=1.4)
        lines_true.append(lt)
        dots_sample.append(ds)
        dots_arrive.append(da)
        cursors.append(ax.axvline(0, color=MUTED, ls="--", lw=1))
        ax.set_xlim(0, t_end * ms)
        ax.set_ylim(*limits[a])
        ax.set_ylabel(f"{axis} (mm)", fontsize=8, color=c)
        _style(ax)
        if a < 2:
            plt.setp(ax.get_xticklabels(), visible=False)
    ax_p[2].set_xlabel("time (ms)", fontsize=8, color=MUTED)
    ax_p[0].set_title("true vs reconstructed position, timestamped", loc="left", fontsize=8.5, color=INK)
    handles = [Line2D([], [], color=INK, lw=1.4, label="true"),
               Line2D([], [], marker="o", ls="", color=SLATE, ms=4.5, mew=0, label="estimate at its sample timestamp"),
               Line2D([], [], marker="o", ls="", mfc="none", mec=SLATE, ms=7, mew=1.4, label="estimate at its arrival time")]
    ax_p[0].legend(handles=handles, loc="upper left", fontsize=7, frameon=False)

    fig.text(0.035, 0.955, f"rolling hex prism {args.speed:g} mm/s, tracked with a windowed {args.window}x{args.window} "
             f"least squares solve ({args.slow:g}x slow motion)", fontsize=11, color=INK, weight="bold")
    fig.text(0.035, 0.925, "sample stamped at conversion midpoint, x/y/z converted one after another, "
             "estimate arrives after readout + solve", fontsize=8.5, color=MUTED)

    cursor_g = [ax_f.axvline(0, color=INK, lw=1), ax_g.axvline(0, color=INK, lw=1)]
    t_arrive = res.t_ready
    est_ok = np.where(ok)[0]

    def update(t: float) -> None:
        for cur_line in cursor_g:
            cur_line.set_xdata([t * ms, t * ms])
        ax_f.set_xlim((t - 0.04) * ms, (t + 0.11) * ms)
        n_arr = int(np.searchsorted(t_arrive, t, side="right"))  # rows whose estimate has arrived
        rows = est_ok[est_ok < n_arr]
        keep = tg <= t
        for a in range(3):
            lines_true[a].set_data(tg[keep] * ms, true_pos[keep, a])
            dots_sample[a].set_data(res.t_sample[rows] * ms, res.est[rows, a])
            dots_arrive[a].set_data(res.t_ready[rows] * ms, res.est[rows, a])
            cursors[a].set_xdata([t * ms, t * ms])
        truth_now = traj.position(t)
        prism.set_verts(prism_polys(truth_now, float(traj.phi(t))))
        if len(rows):
            r = rows[-1]
            e = res.est[r]
            pin.set_data_3d([e[0], e[0]], [e[1], e[1]], [e[2], e[2] + 16])
            ring.set_data_3d([e[0]], [e[1]], [e[2]])
            pin.set_visible(True)
            ring.set_visible(True)
            hud.set_text(f"t = {t * ms:6.1f} ms  latest estimate: sample {r}, age {(t - res.t_sample[r]) * ms:4.1f} ms,"
                         f"\nerror vs the robot now {np.linalg.norm(e - truth_now):.2f} mm")
        else:
            pin.set_visible(False)
            ring.set_visible(False)
            hud.set_text(f"t = {t * ms:6.1f} ms  waiting for the first estimate")

    return fig, update


def grab(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speed", type=float, default=20.0, help="mean roll speed mm/s")
    p.add_argument("--span", type=float, default=1.0, help="sim seconds to cover")
    p.add_argument("--slow", type=float, default=10.0, help="slow motion factor")
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--solver", choices=["full", "windowed"], default="windowed")
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--solve-ms", type=float, default=14.0, help="fixed solver latency in ms")
    p.add_argument("--margin", type=float, default=0.05, help="extra wait after conversion, fraction")
    p.add_argument("--buses", type=int, default=3)
    p.add_argument("--i2c-khz", type=float, default=400.0)
    p.add_argument("--count", type=int, default=6)
    p.add_argument("--pitch", type=float, default=18.0)
    p.add_argument("--dpi", type=int, default=90)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--still", type=float, default=None, help="save one frame at this sim time (s) as png, no gif")
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = p.parse_args()

    timing, traj, sensors, res, t_end = build(args)
    s = res.summary()
    print(f"period {timing.period_s * 1e3:.1f} ms, readout {timing.readout_s * 1e3:.2f} ms, {len(res.ok)} solves, "
          f"lost {100 * s['lost_frac']:.0f}%, rmse {s['rmse_mm']:.3f} mm, at arrival {s['rmse_ready_mm']:.3f} mm, "
          f"age {s['age_ms']:.1f} ms")
    fig, update = make_figure(args, timing, traj, sensors, res, t_end)

    if args.still is not None:
        update(args.still)
        path = os.path.splitext(args.out)[0] + "_still.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=args.dpi)
        print(f"saved {path}")
        return

    dt = 1.0 / args.fps / args.slow
    times = np.arange(0.0, t_end, dt)
    print(f"{len(times)} frames, {dt * 1e3:g} ms of sim time per frame, {len(times) / args.fps:.1f} s of gif")
    palette_src = []
    for t in (0.25 * t_end, 0.55 * t_end, 0.95 * t_end):
        update(t)
        palette_src.append(grab(fig))
    pal = Image.fromarray(np.concatenate(palette_src, axis=0)).quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                                                                       dither=Image.Dither.NONE)
    frames = []
    for i, t in enumerate(times):
        update(t)
        frames.append(Image.fromarray(grab(fig)).quantize(palette=pal, dither=Image.Dither.NONE))
        if i % 25 == 0:
            print(f"  frame {i}/{len(times)}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=int(round(1000 / args.fps)), loop=0,
                   optimize=False, disposal=1)
    print(f"saved {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
