"""physical constants for the Sci Adv 2026 SI truss-based Kresling robot,
constrained to sliding locomotion.

from Sci_Adv_2026_SI.pdf ("Multimodal robotic matter enabled by unified
magnetic control primitives", Manos, Rao, Zhao, Chen, Paulino, Sci Adv 2026):
the truss-based robotic unit carries TWO N42 disc magnets (top + bottom
plate), not the one N50 magnet config.py's default `kresling` target
assumes -- a different target, added alongside it rather than replacing it
(see coil_hardware_audit memory). Constrained to sliding: magnetization/robot
axis is fixed horizontal (M=[0,1,0], eq. S66-S68), not solved for -- this
stays a position-only (x,y,z) problem like every other target.

coil, workspace and sensor-noise constants are the same hardware as the
kresling rig, so they're imported from config.py instead of duplicated.
"""
from __future__ import annotations

import numpy as np

from kresling_sim.config import SENSOR_PCB_GAP_MM
from kresling_sim.sensors import tilt_rotation

# magnet (Sci Adv 2026 SI, page S2 + Table S9)
MAGNET_DIAMETER_MM = 4.0
MAGNET_HEIGHT_MM = 0.5
# Br = mu0 * M0, M0 = 1.05e6 A/m (Table S9) -- the paper's own value, not a
# datasheet guess. mu0 = 4*pi*1e-7 T*m/A
MAGNET_BR_MT = 4 * np.pi * 1e-7 * 1.05e6 * 1000  # ~1319.5 mT

# two magnets, rigidly separated by the truss cell's fixed height (Table S1
# -- this is a rigid 3D-printed frame, no deployed/folded states, so this
# separation never varies across any locomotion mode)
MAGNET_SEPARATION_MM = 12.0  # h, Table S1

# hexagonal end-plate inscribed-circle radius (Table S9), self-consistent
# with Table S1's side length a=6.93mm (a*sqrt(3)/2 = 6.00mm)
HEX_PLATE_INSCRIBED_RADIUS_MM = 6.0

# resting tilt off pure-horizontal: PLACEHOLDER, 0.0 pending a real check
# against the physical robot (see divijas_notes.md). The twisted truss (no
# flat side face, see Z_MIN/MAX comment below) has no reason to rest
# perfectly level -- if it settles tilted by theta, the two magnets end up
# at slightly different z (z_magnet ~= Z_CENTER_MM +/- (MAGNET_SEPARATION_MM/2)*sin(theta)),
# and -- verified empirically this session -- a wrong theta here biases the
# solver's y-estimate, not just z (a resting-tilt error is confined to the
# y-z plane at this azimuth, and the least-squares fit can only "explain"
# the missing tilt by adjusting position within that same plane). Update
# this single number once measured; SLIDING_ORIENTATION picks it up
# automatically, no other code changes needed.
RESTING_TILT_DEG = 0.0

# fixed sliding orientation: magnetization horizontal along global y (Sci
# Adv 2026 SI eq. S66-S68, M=[0,1,0]) tilted by RESTING_TILT_DEG off pure
# horizontal. tilt_rotation(pi/2, pi/2) alone (RESTING_TILT_DEG=0) rotates
# the magnet's local z (magnetization axis) onto global +y exactly, verified
# empirically against sensors.tilt_rotation's convention; reducing the polar
# angle by RESTING_TILT_DEG tilts that axis toward +z by the same angle.
SLIDING_ORIENTATION = tilt_rotation(np.pi / 2 - np.radians(RESTING_TILT_DEG), np.pi / 2)

# z-standoff: PLACEHOLDER, deliberately WIDE, not paper-given. The truss
# cell's top/bottom hex plates are rotationally twisted relative to each
# other (mountain rod b=10.5mm != valley rod c=12.8mm, Table S1 -- the
# defining Kresling feature), so it has no flat side face to rest flush on
# while sliding -- "half the hex width" isn't a rigorous resting height,
# just the paper's own level of simplification for a DIFFERENT maneuver
# (rolling's contact-geometry uses this same Ri/h prism approximation, eq.
# ~S57). Rather than assert a falsely-precise narrow band around that
# guess, this range honestly brackets the plausible resting height:
# SENSOR_PCB_GAP_MM alone (near-zero projection) to
# SENSOR_PCB_GAP_MM + MAGNET_SEPARATION_MM (the full cell height projecting up,
# conservative upper bound). Revisit with a real measurement or a proper
# twisted-truss resting-geometry derivation. z stays a genuine free
# parameter for the inverse solver regardless, same as x/y, never assumed
# known (see coil_hardware_audit / session notes: baking z in would hand
# the solver privileged information about which locomotion mode is active)
# -- this range only sets what true z-values get swept/tested.
Z_CENTER_MM = SENSOR_PCB_GAP_MM + HEX_PLATE_INSCRIBED_RADIUS_MM  # ~12.35mm, illustrative single value (e.g. for tests)
Z_MIN_MM = SENSOR_PCB_GAP_MM + 1.0  # ~7.35mm
Z_MAX_MM = SENSOR_PCB_GAP_MM + MAGNET_SEPARATION_MM  # ~18.35mm (MAGNET_SEPARATION_MM == h)

MAGNET_KWARGS = {
    "diameter": MAGNET_DIAMETER_MM,
    "height": MAGNET_HEIGHT_MM,
    "remanence": MAGNET_BR_MT,
    "separation_mm": MAGNET_SEPARATION_MM,
    "orientation": SLIDING_ORIENTATION,
}
