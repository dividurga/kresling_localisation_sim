"""Physical constants for the IEEE-paper cube-robot target.

Values are taken directly from the IEEE APEC 2025 paper ("Wireless
Actuation of Magnetic Robots with a 3-D Helmholtz Coil System"), Table III
"Robotic Block Parameters": a single N52 disc magnet embedded at the
geometric center of a 3D-printed cube -- a different physical target than
the Kresling cell in config.py, demonstrated in that paper to actually roll
under this same coil system.

Coil, workspace and sensor-noise constants are the same physical hardware
as the Kresling rig, so they're imported from config.py rather than
duplicated here.
"""
from __future__ import annotations

import math

from kresling_sim.config import SENSOR_PCB_GAP_MM

# --- Magnet + block (IEEE APEC 2025, Table III) ---
MAGNET_DIAMETER_MM = 6.35
MAGNET_HEIGHT_MM = 3.18
MAGNET_BR_MT = 1480.0          # 1.48 T, N52 grade
CUBE_SIDE_MM = 20.0
ROBOT_MASS_G = 10.3

# --- z-range (sensor-to-magnet distance) ---
# The magnet sits at the cube's geometric center, so resting flat on any
# face puts it at CUBE_SIDE_MM/2 above the surface. The paper's demo moves
# the cube by rolling face-to-face, pivoting on a bottom edge; at the
# balanced mid-roll point (2-D corner-pivot geometry, square of side
# CUBE_SIDE_MM rotating about one bottom edge) the center rises to
# CUBE_SIDE_MM*sqrt(2)/2. Both endpoints get the same SENSOR_PCB_GAP_MM
# acrylic-sheet standoff used in the Kresling rig -- no separate standoff
# figure is documented for this setup, so the existing hardware value is
# reused (confirmed with user, not assumed).
Z_MIN_MM = SENSOR_PCB_GAP_MM + CUBE_SIDE_MM / 2
Z_MAX_MM = SENSOR_PCB_GAP_MM + CUBE_SIDE_MM * math.sqrt(2) / 2

MAGNET_KWARGS = {
    "diameter": MAGNET_DIAMETER_MM,
    "height": MAGNET_HEIGHT_MM,
    "remanence": MAGNET_BR_MT,
}
