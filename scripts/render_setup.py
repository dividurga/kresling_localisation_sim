"""Rough geometric render of the kresling_sim experimental setup, built
purely from constants in src/kresling_sim/config.py + sensors.py, so we can
sanity-check what the codebase currently thinks the hardware looks like.

Not a CAD model -- primitives only (circles for coil loops, a prism for the
Kresling cell, a small cylinder for the magnet, squares for sensors).
"""
import sys
sys.path.insert(0, "/Users/divija/Divi Drive/workplace/Princeton/Sem 7/origami/kresling_localisation_sim/src")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from kresling_sim.config import (
    COIL_WINDINGS, CELL_POLYGON_RADIUS_MM, MAGNET_DIAMETER_MM, MAGNET_HEIGHT_MM,
    SENSOR_PCB_GAP_MM, Z_DEPLOYED_MM, WORKSPACE_X_MM, WORKSPACE_Y_MM,
)
from kresling_sim.sensors import sensor_grid, positions_array

fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection="3d")

AXIS_UNIT = {"x": np.array([1, 0, 0]), "y": np.array([0, 1, 0]), "z": np.array([0, 0, 1])}
COLORS = {"x": "tab:red", "y": "tab:green", "z": "tab:blue"}


def circle_loop(radius, axis, offset_sign):
    """Points (N,3) for a circular loop of given radius, centered on the
    origin, whose own normal points along `axis`, offset along that axis."""
    t = np.linspace(0, 2 * np.pi, 64)
    if axis == "x":
        pts = np.stack([np.zeros_like(t), radius * np.cos(t), radius * np.sin(t)], axis=1)
    elif axis == "y":
        pts = np.stack([radius * np.cos(t), np.zeros_like(t), radius * np.sin(t)], axis=1)
    else:
        pts = np.stack([radius * np.cos(t), radius * np.sin(t), np.zeros_like(t)], axis=1)
    return pts


# --- coil loops (12 total: 3 axes x 2 sides x 2 windings) ---
for axis, sides in COIL_WINDINGS.items():
    axis_vec = AXIS_UNIT[axis]
    for winding_name, w in sides.items():
        for sign in (+1, -1):
            pts = circle_loop(w.radius_mm, axis, sign)
            pts = pts + sign * w.offset_mm * axis_vec
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=COLORS[axis], alpha=0.5,
                    linewidth=1.2 if winding_name == "outer" else 0.8)

# legend proxies for coil axes
for axis, c in COLORS.items():
    ax.plot([], [], [], color=c, label=f"{axis}-axis coil pair (outer+inner windings)")

# --- workspace volume (translucent box, 100x100 in x/y, spanning sensor
#     plane to deployed height in z as a stand-in since no explicit z bound
#     is given) ---
z_lo, z_hi = 0.0, Z_DEPLOYED_MM
x_half, y_half = WORKSPACE_X_MM / 2, WORKSPACE_Y_MM / 2
box_corners = np.array([
    [-x_half, -y_half, z_lo], [x_half, -y_half, z_lo], [x_half, y_half, z_lo], [-x_half, y_half, z_lo],
])
ax.plot(np.append(box_corners[:, 0], box_corners[0, 0]),
        np.append(box_corners[:, 1], box_corners[0, 1]),
        np.append(box_corners[:, 2], box_corners[0, 2]),
        color="gray", linestyle="--", linewidth=1, label="workspace (100x100mm, XY)")

# --- Kresling cell: hexagonal prism (n=6, per Nature 2025 SI S10),
#     circumradius 16.8mm, from PCB gap to deployed height ---
N_GON = 6
poly_angles = np.arange(N_GON) * 2 * np.pi / N_GON + np.pi / N_GON
poly_x = CELL_POLYGON_RADIUS_MM * np.cos(poly_angles)
poly_y = CELL_POLYGON_RADIUS_MM * np.sin(poly_angles)
z_base, z_top = SENSOR_PCB_GAP_MM, Z_DEPLOYED_MM

verts = []
for i in range(N_GON):
    j = (i + 1) % N_GON
    verts.append([
        (poly_x[i], poly_y[i], z_base), (poly_x[j], poly_y[j], z_base),
        (poly_x[j], poly_y[j], z_top), (poly_x[i], poly_y[i], z_top),
    ])
verts.append([(poly_x[i], poly_y[i], z_base) for i in range(N_GON)])
verts.append([(poly_x[i], poly_y[i], z_top) for i in range(N_GON)])

cell = Poly3DCollection(verts, facecolor="peru", alpha=0.25, edgecolor="saddlebrown", linewidth=1)
ax.add_collection3d(cell)
ax.plot([], [], [], color="saddlebrown", label=f"Kresling cell (hexagonal, R={CELL_POLYGON_RADIUS_MM}mm)")

# --- magnet: small cylinder at the cell's free (top) end, out-of-plane ---
r_mag = MAGNET_DIAMETER_MM / 2
h_mag = MAGNET_HEIGHT_MM
theta = np.linspace(0, 2 * np.pi, 24)
mag_z_bot, mag_z_top = z_top - h_mag, z_top
xc, yc = r_mag * np.cos(theta), r_mag * np.sin(theta)
mag_verts = []
for i in range(len(theta) - 1):
    mag_verts.append([
        (xc[i], yc[i], mag_z_bot), (xc[i + 1], yc[i + 1], mag_z_bot),
        (xc[i + 1], yc[i + 1], mag_z_top), (xc[i], yc[i], mag_z_top),
    ])
mag_verts.append([(xc[i], yc[i], mag_z_bot) for i in range(len(theta) - 1)])
mag_verts.append([(xc[i], yc[i], mag_z_top) for i in range(len(theta) - 1)])
magnet = Poly3DCollection(mag_verts, facecolor="black", alpha=0.9)
ax.add_collection3d(magnet)
ax.plot([], [], [], color="black", label=f"magnet ({MAGNET_DIAMETER_MM}mm dia x {MAGNET_HEIGHT_MM}mm, N50)")

# --- sensor array: illustrative 5x5 @ 10mm pitch at z=0 (matches
#     montecarlo.py's __main__ example; sweep.py actually sweeps side in
#     [3..7]-ish and pitch in [5,8,10,15,20]mm -- this is NOT a "chosen"
#     config, just a representative one) ---
sensors = sensor_grid(num_across=5, num_below=5, sep_horizontal=10.0, sep_vertical=10.0, z=0.0)
sensor_pts = positions_array(sensors)
sensor_half = 1.5  # mm, half of 3x3mm sensor footprint
for p in sensor_pts:
    x0, y0, z0 = p
    sq = np.array([
        [x0 - sensor_half, y0 - sensor_half, z0], [x0 + sensor_half, y0 - sensor_half, z0],
        [x0 + sensor_half, y0 + sensor_half, z0], [x0 - sensor_half, y0 + sensor_half, z0],
        [x0 - sensor_half, y0 - sensor_half, z0],
    ])
    ax.plot(sq[:, 0], sq[:, 1], sq[:, 2], color="orange", linewidth=1.5)
ax.plot([], [], [], color="orange", label="Hall sensors (illustrative 5x5 @ 10mm pitch,\n3x3mm package footprint)")

ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_zlabel("z (mm)")
ax.set_title("kresling_sim: geometric-primitive render of experimental setup\n(from config.py + sensors.py -- straight hexagonal prism stand-in,\nreal Kresling cell twists between top/bottom faces -- NOT to be trusted as CAD)")
ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(-0.15, 1.0))

# 10mm scale reference (bottom-left of the workspace plane) so cell size
# can be checked by eye against a known length.
scale_x0, scale_y0 = -x_half - 15, -y_half - 5
ax.plot([scale_x0, scale_x0 + 10], [scale_y0, scale_y0], [0, 0], color="black", linewidth=2.5)
ax.text(scale_x0 + 5, scale_y0 - 8, 0, "10mm", color="black", fontsize=8, ha="center")

max_range = 170
ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-40, 110)  # matches xy half-range so aspect=[1,1,1] doesn't visually stretch the cell
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=22, azim=-50)

out_path = "/Users/divija/Divi Drive/workplace/Princeton/Sem 7/origami/kresling_localisation_sim/scripts/setup_render.png"
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print("saved:", out_path)
