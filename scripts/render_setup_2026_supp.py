"""rough geometric render of the kresling_2026_supp experimental setup, built
purely from constants in src/kresling_sim/config.py + kresling_2026_supp_config.py,
so we can sanity-check the two-magnet horizontal-axis sliding setup at
relative scale.

not a CAD model -- primitives only (circles for coil loops, a box for the
truss cell footprint, two small cylinders for the magnets, squares for
sensors). Sibling to render_setup.py (the original vertical single-magnet
`kresling` target) rather than a modification of it -- that script stays
correct/untouched as documentation of the old target.
"""
import sys
sys.path.insert(0, "/Users/divija/Divi Drive/workplace/Princeton/Sem 7/origami/kresling_localisation_sim/src")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from kresling_sim.config import COIL_WINDINGS, WORKSPACE_X_MM, WORKSPACE_Y_MM
from kresling_sim import kresling_2026_supp_config as cfg
from kresling_sim.sensors import sensor_grid, positions_array

fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection="3d")

AXIS_UNIT = {"x": np.array([1, 0, 0]), "y": np.array([0, 1, 0]), "z": np.array([0, 0, 1])}
COLORS = {"x": "tab:red", "y": "tab:green", "z": "tab:blue"}


def circle_loop(radius, axis):
    """(N,3) points for a circular loop of given radius, centered on the
    origin, normal along `axis`. Caller applies any offset/sign."""
    t = np.linspace(0, 2 * np.pi, 64)
    if axis == "x":
        pts = np.stack([np.zeros_like(t), radius * np.cos(t), radius * np.sin(t)], axis=1)
    elif axis == "y":
        pts = np.stack([radius * np.cos(t), np.zeros_like(t), radius * np.sin(t)], axis=1)
    else:
        pts = np.stack([radius * np.cos(t), radius * np.sin(t), np.zeros_like(t)], axis=1)
    return pts


# coil loops: 12 total (3 axes x 2 sides x 2 windings) -- same hardware as
# the other targets
for axis, sides in COIL_WINDINGS.items():
    axis_vec = AXIS_UNIT[axis]
    for winding_name, w in sides.items():
        for sign in (+1, -1):
            pts = circle_loop(w.radius_mm, axis)
            pts = pts + sign * w.offset_mm * axis_vec
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=COLORS[axis], alpha=0.5,
                    linewidth=1.2 if winding_name == "outer" else 0.8)

for axis, c in COLORS.items():
    ax.plot([], [], [], color=c, label=f"{axis}-axis coil pair (outer+inner windings)")

# workspace volume: translucent box outline, 100x100 in xy; z spans a small
# band around the sliding standoff (this target's z is narrow -- ground
# contact jitter, not a wide sweep)
z_lo, z_hi = cfg.Z_MIN_MM, cfg.Z_MAX_MM
x_half, y_half = WORKSPACE_X_MM / 2, WORKSPACE_Y_MM / 2
box_corners = np.array([
    [-x_half, -y_half, 0.0], [x_half, -y_half, 0.0], [x_half, y_half, 0.0], [-x_half, y_half, 0.0],
])
ax.plot(np.append(box_corners[:, 0], box_corners[0, 0]),
        np.append(box_corners[:, 1], box_corners[0, 1]),
        np.append(box_corners[:, 2], box_corners[0, 2]),
        color="gray", linestyle="--", linewidth=1, label="workspace (100x100mm, XY)")

# sliding standoff band: a thin translucent slab at Z_CENTER_MM +/- 2mm
slab_verts = [[
    (-x_half, -y_half, z_lo), (x_half, -y_half, z_lo), (x_half, y_half, z_lo), (-x_half, y_half, z_lo),
]]
ax.add_collection3d(Poly3DCollection(slab_verts, facecolor="skyblue", alpha=0.08))
ax.plot([], [], [], color="skyblue", label=f"sliding z-band ({z_lo:.2f}-{z_hi:.2f}mm)")

# two magnets: axis horizontal along global y (sliding orientation), offset
# +/- separation/2 along y from the tracked center point, at the z-center
# standoff above the sensor plane
r_mag = cfg.MAGNET_DIAMETER_MM / 2
h_mag = cfg.MAGNET_HEIGHT_MM
sep = cfg.MAGNET_SEPARATION_MM
z_center = cfg.Z_CENTER_MM
theta = np.linspace(0, 2 * np.pi, 24)

for sign, label_suffix in [(+1, "top plate"), (-1, "bottom plate")]:
    y_center = sign * sep / 2
    # cylinder axis along y: circular cross-section in the x-z plane,
    # extruded along y from y_center-h_mag/2 to y_center+h_mag/2
    xc, zc = r_mag * np.cos(theta) , z_center + r_mag * np.sin(theta)
    y_bot, y_top = y_center - h_mag / 2, y_center + h_mag / 2
    verts = []
    for i in range(len(theta) - 1):
        verts.append([
            (xc[i], y_bot, zc[i]), (xc[i + 1], y_bot, zc[i + 1]),
            (xc[i + 1], y_top, zc[i + 1]), (xc[i], y_top, zc[i]),
        ])
    verts.append([(xc[i], y_bot, zc[i]) for i in range(len(theta) - 1)])
    verts.append([(xc[i], y_top, zc[i]) for i in range(len(theta) - 1)])
    ax.add_collection3d(Poly3DCollection(verts, facecolor="black", alpha=0.9))

ax.plot([], [], [], color="black",
        label=f"2x magnets ({cfg.MAGNET_DIAMETER_MM}mm dia x {cfg.MAGNET_HEIGHT_MM}mm, N42,\n"
              f"axis horizontal along y, separated {sep}mm)")

# sensor array: illustrative 5x5 @ 10mm pitch at z=0 -- same convention as
# render_setup.py, NOT a "chosen" config
sensors = sensor_grid(num_across=5, num_below=5, sep_horizontal=10.0, sep_vertical=10.0, z=0.0)
sensor_pts = positions_array(sensors)
sensor_half = 1.5  # mm, half of 3x3mm sensor footprint
for p in sensor_pts:
    x0, y0, zp = p
    sq = np.array([
        [x0 - sensor_half, y0 - sensor_half, zp], [x0 + sensor_half, y0 - sensor_half, zp],
        [x0 + sensor_half, y0 + sensor_half, zp], [x0 - sensor_half, y0 + sensor_half, zp],
        [x0 - sensor_half, y0 - sensor_half, zp],
    ])
    ax.plot(sq[:, 0], sq[:, 1], sq[:, 2], color="orange", linewidth=1.5)
ax.plot([], [], [], color="orange", label="Hall sensors (illustrative 5x5 @ 10mm pitch,\n3x3mm package footprint)")

ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_zlabel("z (mm)")
ax.set_title(
    "kresling_2026_supp: two-magnet sliding setup, relative scale\n"
    "(magnets horizontal along y, z-band is a PLACEHOLDER standoff -- see config docstring -- NOT to be trusted as CAD)"
)
ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(-0.15, 1.0))

# 10mm scale bar so magnet/separation size can be checked by eye
scale_x0, scale_y0 = -x_half - 15, -y_half - 5
ax.plot([scale_x0, scale_x0 + 10], [scale_y0, scale_y0], [0, 0], color="black", linewidth=2.5)
ax.text(scale_x0 + 5, scale_y0 - 8, 0, "10mm", color="black", fontsize=8, ha="center")

max_range = 170
ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-40, 110)
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=22, azim=-50)

out_path = "/Users/divija/Divi Drive/workplace/Princeton/Sem 7/origami/kresling_localisation_sim/scripts/setup_render_2026_supp.png"
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print("saved:", out_path)
