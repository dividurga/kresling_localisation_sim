"""sensor array sweep GUI. pick a target and grid count/pitch, see it on the 100x100mm workspace,
run a CRB heatmap (seconds) or full MC sweep (slower). both average over n_z z levels like crb_heatmap.py / mc_spatial_heatmaps.py.
only the first quadrant is computed, mirrored into four for display

uv run streamlit run scripts/sensor_sweep_app.py
"""
from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial.transform import Rotation

from kresling_sim import cube_robot_config, fisher, kresling_2026_supp_config
from kresling_sim.circular_array import RadiusSpec, RingSpec, concentric_rings, shared_spokes
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
    "kresling_2026_supp": {
        "z_min": kresling_2026_supp_config.Z_MIN_MM,
        "z_max": kresling_2026_supp_config.Z_MAX_MM,
        "magnet_kwargs": kresling_2026_supp_config.MAGNET_KWARGS,
    },
}

st.set_page_config(page_title="Kresling sensor sweep", layout="wide")

# palette: references/palette.md sequential-blue ramp + categorical slot 2,
# swapped light/dark so the chart card matches the app theme
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
    """grid points per quadrant axis for ~spacing_mm apart"""
    return max(2, round((WORKSPACE_X_MM / 2) / spacing_mm) + 1)


def mirror_to_full(quadrant: np.ndarray) -> np.ndarray:
    """(n,n) quadrant -> (2n-1,2n-1) mirrored into all four (centered square grid is symmetric). x=0/y=0 not duplicated"""
    right = np.concatenate([quadrant[:, 1:][:, ::-1], quadrant], axis=1)
    return np.concatenate([right[1:][::-1], right], axis=0)


def compute_crb_heatmap(
    sensors, n_xy: int, n_z: int, z_min: float, z_max: float, magnet_kwargs: dict | None
) -> np.ndarray:
    xs, ys = first_quadrant_grid(n_xy)
    zs = np.linspace(z_min, z_max, n_z)
    rmse = np.zeros((n_xy, n_xy))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            vals = []
            for z in zs:
                jac = fisher.position_jacobian(np.array([x, y, z]), sensors, magnet_kwargs=magnet_kwargs)
                fim = fisher.fisher_information_matrix(jac, DEFAULT_NOISE)
                bound = fisher.crlb(fim)
                vals.append(float(np.sqrt(np.sum(bound**2))))
            rmse[i, j] = np.mean(vals)
    return rmse


def compute_mc_heatmap(
    sensors, n_xy: int, n_z: int, n_trials: int, warm_start_sigma: float,
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
                    np.array([x, y, z]), sensors,
                    noise=DEFAULT_NOISE, n_trials=n_trials,
                    warm_start_sigma=warm_start_sigma, rng=rng, magnet_kwargs=magnet_kwargs,
                )
                vals.append(result.rmse_mm)
                done += 1
                progress_cb(done / total)
            rmse[i, j] = np.mean(vals)
    return rmse


def plot_heatmap(
    quadrant_rmse: np.ndarray | None, points: np.ndarray, title: str,
    orientations: list[Rotation] | None = None,
) -> go.Figure:
    """full workspace view, quadrant mirrored into four.
    orientations (one Rotation per point) rotates each sensor square to its local z projected into xy,
    None for axis aligned (untilted grid)"""
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

    # sensors as true scale 3mm squares, rotated to local z projected into xy to show tilt.
    # axis aligned if local z is ~perpendicular to the plane (no azimuth)
    for i, (x, y) in enumerate(points[:, :2]):
        azimuth = None
        if orientations is not None:
            dx, dy, _ = orientations[i].apply([0.0, 0.0, 1.0])
            if np.hypot(dx, dy) > 1e-6:
                azimuth = float(np.arctan2(dy, dx))

        if azimuth is None:
            fig.add_shape(type="rect", x0=x - half_s, x1=x + half_s, y0=y - half_s, y1=y + half_s,
                          line=dict(color=INK, width=0.5), fillcolor=SENSOR_COLOR)
        else:
            cos_a, sin_a = np.cos(azimuth), np.sin(azimuth)
            corners = [
                (x + cx * cos_a - cy * sin_a, y + cx * sin_a + cy * cos_a)
                for cx, cy in [(-half_s, -half_s), (half_s, -half_s), (half_s, half_s), (-half_s, half_s)]
            ]
            path = "M " + " L ".join(f"{px:.3f},{py:.3f}" for px, py in corners) + " Z"
            fig.add_shape(type="path", path=path, line=dict(color=INK, width=0.5), fillcolor=SENSOR_COLOR)

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


def sensors_to_svg(
    points: np.ndarray, sensor_size_mm: float,
    orientations: list[Rotation] | None = None, margin_mm: float = 2.0,
) -> str:
    """sensor footprints as an unfilled SVG at true mm scale for PCB/CAD. orientations rotates them like plot_heatmap"""
    half_s = sensor_size_mm / 2
    min_x, min_y = points[:, 0].min() - half_s - margin_mm, points[:, 1].min() - half_s - margin_mm
    max_x, max_y = points[:, 0].max() + half_s + margin_mm, points[:, 1].max() + half_s + margin_mm
    width, height = max_x - min_x, max_y - min_y

    def _footprint(x: float, y: float, azimuth: float | None) -> str:
        if azimuth is None:
            return (f'  <rect x="{x - half_s:.3f}" y="{y - half_s:.3f}" '
                     f'width="{sensor_size_mm:.3f}" height="{sensor_size_mm:.3f}" '
                     f'fill="none" stroke="black" stroke-width="0.1"/>')
        cos_a, sin_a = np.cos(azimuth), np.sin(azimuth)
        corners = [
            (x + cx * cos_a - cy * sin_a, y + cx * sin_a + cy * cos_a)
            for cx, cy in [(-half_s, -half_s), (half_s, -half_s), (half_s, half_s), (-half_s, half_s)]
        ]
        pts = " ".join(f"{px:.3f},{py:.3f}" for px, py in corners)
        return f'  <polygon points="{pts}" fill="none" stroke="black" stroke-width="0.1"/>'

    azimuths: list[float | None] = [None] * len(points)
    if orientations is not None:
        azimuths = []
        for rot in orientations:
            dx, dy, _ = rot.apply([0.0, 0.0, 1.0])
            azimuths.append(float(np.arctan2(dy, dx)) if np.hypot(dx, dy) > 1e-6 else None)

    rects = "\n".join(_footprint(x, y, az) for (x, y), az in zip(points[:, :2], azimuths))
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
    array_type = st.radio("array type", ["rectangular grid", "circular array"], horizontal=True)

    if array_type == "rectangular grid":
        n_side = st.slider("Sensor count (N x N)", 3, 12, 7)
        pitch = st.slider("Sensor pitch (mm)", 8.0, 30.0, 14.0, step=1.0)
        st.caption(f"{n_side * n_side} sensors, {(n_side - 1) * pitch:.0f}mm span")
    else:
        layout_mode = st.radio("layout", ["concentric rings", "shared spokes"], horizontal=True)
        radius_mode = st.radio("radius spec", ["uniform pitch", "custom"], horizontal=True)

        radius_spec = None
        try:
            if radius_mode == "uniform pitch":
                inner_radius = st.number_input("inner radius (mm)", min_value=0.1, value=20.0, step=1.0)
                radial_pitch = st.number_input("radial pitch (mm)", min_value=0.1, value=20.0, step=1.0)
                st.session_state.setdefault("ring_count", 3)
                ring_count = st.number_input("rings", min_value=1, step=1, key="ring_count")
                radius_spec = RadiusSpec(
                    mode="uniform", inner_radius=inner_radius, radial_pitch=radial_pitch, ring_count=int(ring_count)
                )
            elif layout_mode == "shared spokes":
                # concentric rings take radii from the ring table, shared spokes has no table so radii go here
                radii_text = st.text_area("radii (mm, one per line)", value="20\n40\n60")
                custom_radii = [float(x) for x in radii_text.splitlines() if x.strip()]
                radius_spec = RadiusSpec(mode="custom", custom_radii=custom_radii)
        except ValueError as e:
            st.error(f"invalid radii: {e}")
            st.stop()

        orientation_mode = st.radio("orientation", ["radial", "tangential", "fixed"], horizontal=True)
        fixed_azimuth_deg = st.number_input("fixed angle (deg)", value=0.0, step=5.0) if orientation_mode == "fixed" else 0.0
        orientation_offset_deg = st.number_input("orientation offset (deg)", value=0.0, step=5.0)
        orientation_kwargs = dict(
            orientation_mode=orientation_mode,
            tilt_polar_deg=90.0,
            fixed_polar_deg=90.0,
            fixed_azimuth_deg=fixed_azimuth_deg,
            orientation_offset_deg=orientation_offset_deg,
        )

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

if array_type == "rectangular grid":
    sensors = sensor_grid(num_across=n_side, num_below=n_side,
                           sep_horizontal=pitch, sep_vertical=pitch)
    layout_caption = f"{n_side}x{n_side} @ {pitch:.0f}mm pitch"
    layout_slug = f"{n_side}x{n_side}_{pitch:.0f}mm"
else:
    try:
        if layout_mode == "concentric rings":
            uniform = radius_spec is not None  # None only for concentric-rings + custom radius mode

            def _add_ring() -> None:
                if uniform:
                    st.session_state.ring_count += 1
                else:
                    prev = st.session_state.rings_state
                    last = prev[-1] if prev else {"radius": 20.0, "count": 8, "offset_deg": 0.0, "arc_deg": 360.0}
                    st.session_state.rings_state = prev + [{**last, "radius": last["radius"] + 20.0}]

            def _remove_last_ring() -> None:
                if uniform:
                    st.session_state.ring_count = max(1, st.session_state.ring_count - 1)
                elif len(st.session_state.rings_state) > 1:
                    st.session_state.rings_state = st.session_state.rings_state[:-1]

            col_add, col_remove = st.columns(2)
            col_add.button("+ Add ring", on_click=_add_ring, width="stretch")
            col_remove.button("- Remove last ring", on_click=_remove_last_ring, width="stretch")

            if uniform:
                radii = radius_spec.radii()
                # radius column comes from the Radii section (locked), count/offset/arc keep edited values on resize.
                # prev_rows must be the tables last returned values, not overwritten before the widget renders,
                # otherwise edits are lost when the row count changes (new widget key)
                prev_rows = st.session_state.get("rings_state", [])
                default_rows = [
                    {**(prev_rows[i] if i < len(prev_rows) else {"count": 8, "offset_deg": 0.0, "arc_deg": 360.0}),
                     "radius": r}
                    for i, r in enumerate(radii)
                ]
                ring_rows = st.data_editor(
                    default_rows, num_rows="fixed", key=f"ring_table_{len(radii)}",
                    column_config={"radius": st.column_config.NumberColumn("radius (mm)", disabled=True)},
                )
                st.session_state.rings_state = list(ring_rows)
            else:
                st.session_state.setdefault(
                    "rings_state", [{"radius": 20.0, "count": 8, "offset_deg": 0.0, "arc_deg": 360.0}]
                )
                ring_rows = st.data_editor(
                    st.session_state.rings_state, num_rows="fixed",
                    key=f"ring_table_{len(st.session_state.rings_state)}",
                    column_config={"radius": st.column_config.NumberColumn("radius (mm)")},
                )
                st.session_state.rings_state = list(ring_rows)

            rings = [
                RingSpec(radius=float(row["radius"]), count=int(row["count"]),
                          offset_deg=float(row["offset_deg"]), arc_deg=float(row["arc_deg"]))
                for row in ring_rows
            ]
            sensors = concentric_rings(rings, **orientation_kwargs)
            layout_caption = f"{len(sensors)} sensors, {len(rings)} ring(s)"
            layout_slug = f"{len(rings)}rings_{len(sensors)}sensors"
        else:  # shared spokes
            radii = radius_spec.radii()
            st.caption(f"radii: {', '.join(f'{r:g}mm' for r in radii)}")
            shared_count = st.number_input("sensors per spoke radius", min_value=1, value=8, step=1)
            shared_offset = st.number_input("offset (deg)", value=0.0, step=5.0)
            shared_arc = st.number_input("arc (deg)", min_value=0.1, max_value=360.0, value=360.0, step=5.0)
            sensors = shared_spokes(
                radii=radii, count=int(shared_count), offset_deg=shared_offset, arc_deg=shared_arc,
                **orientation_kwargs,
            )
            layout_caption = f"{len(sensors)} sensors, {len(radii)} spoke radii"
            layout_slug = f"{len(radii)}spokes_{len(sensors)}sensors"
    except ValueError as e:
        st.error(f"invalid array parameters: {e}")
        st.stop()
    st.caption(layout_caption)

points = positions_array(sensors)
orientations = [s.orientation for s in sensors] if array_type == "circular array" else None

col1, col2, col3 = st.columns(3)
run_crb = col1.button("Compute CRB heatmap (quick)", width="stretch")
run_mc = col2.button("Run Monte Carlo sweep", width="stretch")
col3.download_button(
    "Export sensor outlines (SVG)",
    data=sensors_to_svg(points, SENSOR_SIZE_MM, orientations=orientations),
    file_name=f"sensor_outlines_{layout_slug}.svg",
    mime="image/svg+xml",
    width="stretch",
)

status = st.empty()
plot_area = st.empty()

if run_crb:
    t0 = time.time()
    with st.spinner("computing CRB..."):
        quadrant = compute_crb_heatmap(sensors, n_xy, n_z, z_min, z_max, magnet_kwargs)
    elapsed = time.time() - t0
    fig = plot_heatmap(quadrant, points,
                        f"CRB position-RMSE ({target_name}) -- {layout_caption}",
                        orientations=orientations)
    plot_area.plotly_chart(fig, width="stretch")
    status.success(f"mean={quadrant.mean():.3f}mm  max={quadrant.max():.3f}mm  ({elapsed:.1f}s)")
elif run_mc:
    progress = st.progress(0.0)
    rng = np.random.default_rng(42)
    t0 = time.time()
    quadrant = compute_mc_heatmap(sensors, n_xy, n_z, n_trials, warm_start_sigma,
                                   rng, progress.progress, z_min, z_max, magnet_kwargs)
    elapsed = time.time() - t0
    progress.empty()
    fig = plot_heatmap(quadrant, points,
                        f"Monte Carlo RMSE ({target_name}) -- {layout_caption}, {n_trials} trials",
                        orientations=orientations)
    plot_area.plotly_chart(fig, width="stretch")
    status.success(f"mean={quadrant.mean():.3f}mm  max={quadrant.max():.3f}mm  ({elapsed:.1f}s)")
else:
    plot_area.plotly_chart(
        plot_heatmap(None, points, "sensor layout preview -- pick a sweep above", orientations=orientations),
        width="stretch",
    )
