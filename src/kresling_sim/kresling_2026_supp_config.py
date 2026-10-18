"""constants for the hexagonal prism two magnet target (kresling_2026_supp), sliding only.
geometry from sci adv 2026 SI. two parallel N42 discs, one per end plate (eq S68), prism rests flat on a side face.
paper unit is a twisted truss, twist ignored since only the magnet positions matter.
axis fixed horizontal along +y (eq S66-S68), position only (x,y,z) problem.
coil, workspace and noise constants are shared with config.py
"""
from __future__ import annotations

import numpy as np

from kresling_sim.config import SENSOR_PCB_GAP_MM
from kresling_sim.sensors import tilt_rotation

# magnet (sci adv 2026 SI p S2 + table S9)
MAGNET_DIAMETER_MM = 4.0
MAGNET_HEIGHT_MM = 0.5
# Br = mu0*M0, M0 = 1.05e6 A/m (table S9), paper value not a datasheet one
MAGNET_BR_MT = 4 * np.pi * 1e-7 * 1.05e6 * 1000  # ~1319.5 mT

# magnet separation = prism height h (table S1), rigid frame so it never changes
MAGNET_SEPARATION_MM = 12.0

# hex plate inscribed radius (table S9), matches side a=6.93mm from table S1
HEX_PLATE_INSCRIBED_RADIUS_MM = 6.0

# resting tilt off horizontal. flat on a side face the axis is horizontal so 0 is nominal.
# small rotations are covered by the z band below
RESTING_TILT_DEG = 0.0

# magnetization along global +y (eq S66-S68) tilted RESTING_TILT_DEG toward +z.
# tilt_rotation(pi/2, pi/2) maps local z onto +y
SLIDING_ORIENTATION = tilt_rotation(np.pi / 2 - np.radians(RESTING_TILT_DEG), np.pi / 2)

# z standoff: magnet axis sits at SENSOR_PCB_GAP_MM + inscribed radius. z is still free for the solver,
# this only sets which true z get swept
Z_CENTER_MM = SENSOR_PCB_GAP_MM + HEX_PLATE_INSCRIBED_RADIUS_MM  # 12.35mm

# allowed rotation off flat. pitch theta moves each magnet z by (h/2)*sin(theta), 10deg is +/-1.04mm
MAX_ROTATION_DEG = 10.0
Z_HALF_BAND_MM = (MAGNET_SEPARATION_MM / 2) * np.sin(np.radians(MAX_ROTATION_DEG))
Z_MIN_MM = Z_CENTER_MM - Z_HALF_BAND_MM  # ~11.31mm
Z_MAX_MM = Z_CENTER_MM + Z_HALF_BAND_MM  # ~13.39mm

MAGNET_KWARGS = {
    "diameter": MAGNET_DIAMETER_MM,
    "height": MAGNET_HEIGHT_MM,
    "remanence": MAGNET_BR_MT,
    "separation_mm": MAGNET_SEPARATION_MM,
    "orientation": SLIDING_ORIENTATION,
}
