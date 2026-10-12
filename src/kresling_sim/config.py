"""physical constants for the kresling localisation setup.

all dimensions in mm and mT unless noted. single source of truth for
hardware geometry and sensor params -- everything else should import from
here instead of hardcoding values.
"""
from dataclasses import dataclass


# coil workspace (IEEE APEC 2025) -- usable volume of the 3-D Helmholtz coil system
WORKSPACE_X_MM = 100.0
WORKSPACE_Y_MM = 100.0

# field-per-unit-current at the origin (mT per amp, one same-axis coil side
# -- inner+outer winding in series -- driven with unit current). reference
# only -- coils.py computes the field directly from COIL_WINDINGS geometry
# via magpy, not from these. values from Sci Adv 2026 SI eq. S8, which
# confirmed COIL_WINDINGS matches this exact coil rig (also consistent with
# IEEE APEC 2025 eq. 9).
COIL_FIELD_PER_AMP_X_MT_A = 0.326
COIL_FIELD_PER_AMP_Y_MT_A = 0.324
COIL_FIELD_PER_AMP_Z_MT_A = 0.321

# max per-submodule current (IEEE APEC 2025, section III-B). coils are OFF
# during localisation (actuation-off sensing window) -- this is just for
# reference / future closed-loop actuation work
COIL_MAX_CURRENT_A = 100.0

# coil winding geometry (IEEE APEC 2025, Table I) -- each of the 6 sides is
# an outer+inner winding pair in series. R = mean winding radius (mm),
# d = offset from origin along the coil axis (mm)
@dataclass(frozen=True)
class Winding:
    radius_mm: float
    turns: int
    offset_mm: float


COIL_WINDINGS = {
    "x": {"outer": Winding(152.0, 56, 97.0), "inner": Winding(165.0, 64, 72.0)},
    "y": {"outer": Winding(122.0, 48, 85.0), "inner": Winding(135.0, 54, 63.0)},
    "z": {"outer": Winding(100.0, 40, 66.0), "inner": Winding(111.0, 40, 48.0)},
}

# kresling cell geometry (nature 2025 SI, S10)
CELL_POLYGON_RADIUS_MM = 16.8  # circumradius of the cell's hexagonal (n=6) footprint

# kresling magnet (N50 NdFeB, supermagnetman D1002-10)
MAGNET_DIAMETER_MM = 2.0
MAGNET_HEIGHT_MM = 0.5
# N50 remanence midpoint of the 1.40-1.45T datasheet range -- confirm
# against the real D1002-10 datasheet before trusting absolute magnitudes
MAGNET_BR_MT = 1425.0

# deployment z-range
SENSOR_PCB_GAP_MM = 6.35   # ~0.25in PCB standoff below kresling base

# sensor-to-magnet distance at full deployment: h1=28.3mm (nature 2025 SI)
# + the acrylic spacer between the sensor PCB and kresling base -- not
# simply h1 + SENSOR_PCB_GAP_MM, the acrylic adds extra offset
Z_DEPLOYED_MM = 31.65

# sensor noise (Melexis MLX90393, GAIN_SEL=0x7)
# RMS noise stdev from datasheet Rev 012 Figs 2 & 3 at ~10-14ms conversion
# time (middle ground between actuation-off window and noise floor)
# TODO: update once the real conversion time is confirmed on the board --
# noise shifts with conversion time
SENSOR_NOISE_SIGMA_XY_MT = 0.0008   # X and Y axes, mT
SENSOR_NOISE_SIGMA_Z_MT = 0.0014    # Z axis (consistently noisier per datasheet)
