"""constants for the IEEE APEC 2025 cube robot (table III), single N52 disc magnet at the cube center.
coil, workspace and noise constants are shared with config.py
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

# z range: flat on a face the magnet is at side/2, mid roll (pivot on edge) side*sqrt2/2.
# same acrylic standoff as the kresling rig, no separate number documented
Z_MIN_MM = SENSOR_PCB_GAP_MM + CUBE_SIDE_MM / 2
Z_MAX_MM = SENSOR_PCB_GAP_MM + CUBE_SIDE_MM * math.sqrt(2) / 2

MAGNET_KWARGS = {
    "diameter": MAGNET_DIAMETER_MM,
    "height": MAGNET_HEIGHT_MM,
    "remanence": MAGNET_BR_MT,
}
