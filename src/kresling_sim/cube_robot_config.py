"""physical constants for the IEEE-paper cube-robot target.

from the IEEE APEC 2025 paper ("wireless actuation of magnetic robots with
a 3-D helmholtz coil system"), table III "robotic block parameters": a
single N52 disc magnet at the center of a 3D-printed cube -- a different
target than the kresling cell in config.py, but one the paper actually
demos rolling under this same coil system.

coil, workspace and sensor-noise constants are the same hardware as the
kresling rig, so they're imported from config.py instead of duplicated.
"""
from __future__ import annotations

import math

from kresling_sim.config import SENSOR_PCB_GAP_MM

# magnet + block (IEEE APEC 2025, table III)
MAGNET_DIAMETER_MM = 6.35
MAGNET_HEIGHT_MM = 3.18
MAGNET_BR_MT = 1480.0          # 1.48T, N52 grade
CUBE_SIDE_MM = 20.0
ROBOT_MASS_G = 10.3

# z-range (sensor-to-magnet distance): magnet sits at the cube's center, so
# resting flat on a face puts it at CUBE_SIDE_MM/2 above the surface.
# mid-roll (pivoting on a bottom edge, corner-pivot geometry) it rises to
# CUBE_SIDE_MM*sqrt(2)/2. both use the same SENSOR_PCB_GAP_MM acrylic
# standoff as the kresling rig -- no separate figure documented for this
# setup, so reusing the existing one (confirmed with user, not assumed)
Z_MIN_MM = SENSOR_PCB_GAP_MM + CUBE_SIDE_MM / 2
Z_MAX_MM = SENSOR_PCB_GAP_MM + CUBE_SIDE_MM * math.sqrt(2) / 2

MAGNET_KWARGS = {
    "diameter": MAGNET_DIAMETER_MM,
    "height": MAGNET_HEIGHT_MM,
    "remanence": MAGNET_BR_MT,
}
