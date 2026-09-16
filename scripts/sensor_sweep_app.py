"""Interactive sensor-array sweep GUI.

Pick a target (kresling or cube_robot -- see TARGETS below), a sensor grid
count (N x N) and pitch, see it overlaid on the 100x100mm workspace, then
run either a quick CRB heatmap (analytical, seconds) or a full Monte Carlo
sweep (nonlinear solver, warm-start, can take a while) to see position-RMSE
across the workspace.

Both heatmaps average over `n_z` z-levels spanning the selected target's
z-range, matching how crb_heatmap.py and mc_spatial_heatmaps.py already
summarize a config -- one RMSE number per (x,y) pixel.

Only the first quadrant (x>=0, y>=0) is actually computed -- a centered
square sensor grid is symmetric under x->-x and y->-y, so the rest carries
no new information (same symmetry crb_heatmap.py and mc_spatial_heatmaps.py
already exploit) -- but it's mirrored into all four quadrants for display,
since a heatmap that only fills a quarter of the visible workspace reads as
broken rather than intentional.

Run with: uv run streamlit run scripts/sensor_sweep_app.py
"""
from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from kresling_sim import cube_robot_config, fisher
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

st.set_page_config(page_title="Kresling sensor sweep", layout="wide")

# -- palette (references/palette.md sequential-blue ramp + categorical slot 2),
# switched between the light/dark columns to match the app's actual theme so
# the chart card doesn't sit as a mismatched box against the page. --
SEQUENTIAL_BLUE = [
    (0 / 6, "#cde2fb"), (1 / 6, "#9ec5f4"), (2 / 6, "#6da7ec"), (3 / 6, "#3987e5"),
    (4 / 6, "#256abf"), (5 / 6, "#184f95"), (6 / 6, "#0d366b"),
]
_DARK = (st.context.theme.type or "dark") == "dark"
SENSOR_COLOR = "#d95926" if _DARK else "#eb6834"   # categorical slot 2 (orange)
INK = "#ffffff" if _DARK else "#0b0b0b"
MUTED = "#898781"  # same muted axis/label tone in both modes
GRIDLINE = "#2c2c2a" if _DARK else "#e1e0d9"
SURFACE = "#1a1a19" if _DARK else "#fcfcfb"

SENSOR_SIZE_MM = 3.0  # drawn footprint of one Hall sensor, true-to-scale


def first_quadrant_grid(n_xy: int) -> tuple[np.ndarray, np.ndarray]:
    """(n_xy,) xs and ys spanning [0, workspace/2] mm."""
    xs = np.linspace(0.0, WORKSPACE_X_MM / 2, n_xy)
    ys = np.linspace(0.0, WORKSPACE_Y_MM / 2, n_xy)
    return xs, ys


def n_points_for_spacing(spacing_mm: float) -> int:
    """Number of grid points per quadrant axis so consecutive points are
    ~spacing_mm apart over [0, workspace/2] mm."""
    return max(2, round((WORKSPACE_X_MM / 2) / spacing_mm) + 1)


def mirror_to_full(quadrant: np.ndarray) -> np.ndarray:
    """(n,n) RMSE for x,y >= 0 -> (2n-1,2n-1) mirrored into all four
    quadrants, exploiting the centered-square-grid symmetry RMSE(-x,y) =
    RMSE(x,y) = RMSE(x,-y). Column 0 (x=0) and row 0 (y=0) aren't
    duplicated when mirrored."""
    right = np.concatenate([quadrant[:, 1:][:, ::-1], quadrant], axis=1)
    return np.concatenate([right[1:][::-1], right], axis=0)


def compute_crb_heatmap(
    points: np.ndarray, n_xy: int, n_z: int, z_min: float, z_max: float, magnet_kwargs: dict | None
) -> np.ndarray:
    xs, ys = first_quadrant_grid(n_xy)
    zs = np.linspace(z_min, z_max, n_z)
    rmse = np.zeros((n_xy, n_xy))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            vals = []
            for z in zs:
                jac = fisher.position_jacobian(np.array([x, y, z]), points, magnet_kwargs=magnet_kwargs)
                fim = fisher.fisher_information_matrix(jac, DEFAULT_NOISE)
                bound = fisher.crlb(fim)
                vals.append(float(np.sqrt(np.sum(bound**2))))
            rmse[i, j] = np.mean(vals)
    return rmse


def compute_mc_heatmap(
    points: np.ndarray, n_xy: int, n_z: int, n_trials: int, warm_start_sigma: float,
    rng: np.random.Generator, progress_cb, z_min: float, z_max: float, magnet_kwargs: dict | None,
) -> np.ndarray:
    xs, ys = first_quadrant_grid(n_xy)
    zs = np.linspace(z_min, z_max, n_z)
    rmse = np.zeros((n_xy, n_xy))
    total = n_xy * n_xy * n_z
    done = 0
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            vals = []
            for z in zs:
                result = run_monte_carlo(
                    np.array([x, y, z]), points,
                    noise=DEFAULT_NOISE, n_trials=n_trials,
                    warm_start_sigma=warm_start_sigma, rng=rng, magnet_kwargs=magnet_kwargs,
                )
                vals.append(result.rmse_mm)
                done += 1
                progress_cb(done / total)
            rmse[i, j] = np.mean(vals)
    return rmse


def plot_heatmap(quadrant_rmse: np.ndarray | None, points: np.ndarray, title: str) -> go.Figure:
    """Full 100x100mm workspace view. Only the first quadrant is actually
    computed (see module docstring), but it's mirrored into all four here
    so the heatmap fills the whole workspace it's plotted against."""
    half_x, half_y = WORKSPACE_X_MM / 2, WORKSPACE_Y_MM / 2
    half_s = SENSOR_SIZE_MM / 2

    fig = go.Figure()

    if quadrant_rmse is not None:
        full_rmse = mirror_to_full(quadrant_rmse)
        n = quadrant_rmse.shape[0]
        xs = np.concatenate([-np.linspace(0.0, half_x, n)[1:][::-1], np.linspace(0.0, half_x, n)])
        ys = np.concatenate([-np.linspace(0.0, half_y, n)[1:][::-1], np.linspace(0.0, half_y, n)])
        fig.add_trace(go.Heatmap(
            z=full_rmse, x=xs, y=ys,
            colorscale=SEQUENTIAL_BLUE,
            colorbar=dict(title=dict(text="RMSE (mm)", font=dict(color=INK)),
                           outlinewidth=0, tickcolor=MUTED, tickfont=dict(color=MUTED)),
            hovertemplate="x=%{x:.1f}mm<br>y=%{y:.1f}mm<br>RMSE=%{z:.3f}mm<extra></extra>",
        ))

    # full workspace boundary
    fig.add_shape(type="rect", x0=-half_x, y0=-half_y, x1=half_x, y1=half_y,
                  line=dict(color=MUTED, width=1.5, dash="dash"), fillcolor="rgba(0,0,0,0)")

    # sensors drawn as true-scale 3mm x 3mm squares
    for x, y in points[:, :2]:
        fig.add_shape(type="rect", x0=x - half_s, x1=x + half_s, y0=y - half_s, y1=y + half_s,
                      line=dict(color=INK, width=0.5), fillcolor=SENSOR_COLOR)

    # legend proxies for the two shape types (shapes themselves don't get legend entries)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="sensor",
                              marker=dict(symbol="square", size=10, color=SENSOR_COLOR,
                                          line=dict(color=INK, width=0.5))))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="workspace edge",
                              line=dict(color=MUTED, width=1.5, dash="dash")))

    margin = half_s + 1.0
    fig.update_layout(
        title=dict(text=title, font=dict(color=INK, size=14)),
        xaxis=dict(title="x (mm)", range=[-half_x - margin, half_x + margin], constrain="domain",
                   showgrid=True, gridcolor=GRIDLINE, zeroline=False, color=MUTED),
        yaxis=dict(title="y (mm)", range=[-half_y - margin, half_y + margin],
                   showgrid=True, gridcolor=GRIDLINE, zeroline=False, color=MUTED,
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="right", x=0.99,
                    bgcolor=SURFACE, bordercolor=GRIDLINE, borderwidth=1,
                    font=dict(color=INK)),
        margin=dict(l=60, r=20, t=50, b=50),
        height=560,
    )
    return fig


def sensors_to_svg(points: np.ndarray, sensor_size_mm: float, margin_mm: float = 2.0) -> str:
    """Vector outline of every sensor footprint (unfilled square, true mm
    scale) as a standalone SVG -- for dropping into a PCB/CAD tool. Just
    outlines, no fill, so it overlays cleanly on existing artwork."""
    half_s = sensor_size_mm / 2
    min_x, min_y = points[:, 0].min() - half_s - margin_mm, points[:, 1].min() - half_s - margin_mm
    max_x, max_y = points[:, 0].max() + half_s + margin_mm, points[:, 1].max() + half_s + margin_mm
    width, height = max_x - min_x, max_y - min_y

    rects = "\n".join(
        f'  <rect x="{x - half_s:.3f}" y="{y - half_s:.3f}" '
        f'width="{sensor_size_mm:.3f}" height="{sensor_size_mm:.3f}" '
        f'fill="none" stroke="black" stroke-width="0.1"/>'
        for x, y in points[:, :2]
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width:.3f}mm" height="{height:.3f}mm" '
        f'viewBox="{min_x:.3f} {min_y:.3f} {width:.3f} {height:.3f}">\n'
        f"{rects}\n"
        f"</svg>\n"
    )


st.title("Kresling sensor-array sweep")

with st.sidebar:
    st.header("Target")
    target_name = st.radio("magnet / geometry", list(TARGETS), horizontal=True)
    target = TARGETS[target_name]
    z_min, z_max, magnet_kwargs = target["z_min"], target["z_max"], target["magnet_kwargs"]

    st.header("Sensor array")
    n_side = st.slider("Sensor count (N x N)", 3, 12, 7)
    pitch = st.slider("Sensor pitch (mm)", 8.0, 30.0, 14.0, step=1.0)
    st.caption(f"{n_side * n_side} sensors, {(n_side - 1) * pitch:.0f}mm span")

    st.header("Heatmap resolution")
    grid_spacing = st.slider("grid spacing (mm)", 0.5, 25.0, 5.0, step=0.5)
    n_xy = n_points_for_spacing(grid_spacing)
    st.caption(f"{n_xy}x{n_xy} points in the computed quadrant ({n_xy * n_xy} total)")
    n_z = st.slider("z levels averaged", 1, 6, 3)
    st.caption(f"z range: {z_min:.2f}-{z_max:.2f}mm")

    st.header("Monte Carlo")
    n_trials = st.slider("trials per position", 5, 100, 15)
    warm_start_sigma = st.slider("warm-start sigma (mm)", 0.0, 10.0, 3.0)
    total_solves = n_xy * n_xy * n_z * n_trials
    st.caption(f"~{total_solves:,} solver calls for this sweep")

sensors = sensor_grid(num_across=n_side, num_below=n_side,
                       sep_horizontal=pitch, sep_vertical=pitch)
points = positions_array(sensors)

col1, col2, col3 = st.columns(3)
run_crb = col1.button("Compute CRB heatmap (quick)", width="stretch")
run_mc = col2.button("Run Monte Carlo sweep", width="stretch")
col3.download_button(
    "Export sensor outlines (SVG)",
    data=sensors_to_svg(points, SENSOR_SIZE_MM),
    file_name=f"sensor_outlines_{n_side}x{n_side}_{pitch:.0f}mm.svg",
    mime="image/svg+xml",
    width="stretch",
)

status = st.empty()
plot_area = st.empty()

if run_crb:
    t0 = time.time()
    with st.spinner("computing CRB..."):
        quadrant = compute_crb_heatmap(points, n_xy, n_z, z_min, z_max, magnet_kwargs)
    elapsed = time.time() - t0
    fig = plot_heatmap(quadrant, points,
                        f"CRB position-RMSE ({target_name}) -- {n_side}x{n_side} @ {pitch:.0f}mm pitch")
    plot_area.plotly_chart(fig, width="stretch")
    status.success(f"mean={quadrant.mean():.3f}mm  max={quadrant.max():.3f}mm  ({elapsed:.1f}s)")
elif run_mc:
    progress = st.progress(0.0)
    rng = np.random.default_rng(42)
    t0 = time.time()
    quadrant = compute_mc_heatmap(points, n_xy, n_z, n_trials, warm_start_sigma,
                                   rng, progress.progress, z_min, z_max, magnet_kwargs)
    elapsed = time.time() - t0
    progress.empty()
    fig = plot_heatmap(quadrant, points,
                        f"Monte Carlo RMSE ({target_name}) -- {n_side}x{n_side} @ {pitch:.0f}mm pitch, {n_trials} trials")
    plot_area.plotly_chart(fig, width="stretch")
    status.success(f"mean={quadrant.mean():.3f}mm  max={quadrant.max():.3f}mm  ({elapsed:.1f}s)")
else:
    plot_area.plotly_chart(plot_heatmap(None, points, "sensor layout preview -- pick a sweep above"), width="stretch")
