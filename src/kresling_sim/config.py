"""Physical constants for the Kresling localisation setup.

All dimensions in mm and mT unless noted. This is the single source of
truth for hardware geometry and sensor parameters -- every other module
should import from here rather than hardcoding values.
"""
from dataclasses import dataclass


# --- Coil workspace (IEEE APEC 2025) ---
# Usable volume of the 3-D Helmholtz coil system.
WORKSPACE_X_MM = 100.0
WORKSPACE_Y_MM = 100.0

# Field-per-unit-current at the origin, from eq. (9) of IEEE APEC 2025
# (mT per amp, one same-axis coil pair driven with equal currents).
COIL_FIELD_PER_AMP_X_MT_A = 0.326
COIL_FIELD_PER_AMP_YZ_MT_A = 0.322

# Maximum per-submodule current (IEEE APEC 2025, Section III-B).
# During localisation the coils are OFF (actuation-off sensing window);
# this is here for reference / future closed-loop actuation work.
COIL_MAX_CURRENT_A = 100.0

# --- Coil winding geometry (IEEE APEC 2025, Table I) ---
# Each of the 6 coil sides is an outer + inner winding pair in series.
# R: mean winding radius (mm), d: offset from origin along the coil axis (mm).
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

# --- Kresling cell geometry (Nature 2025 SI, S10) ---
CELL_POLYGON_RADIUS_MM = 16.8  # circumradius of the cell's hexagonal (n=6) footprint

# --- Kresling magnet (N50 NdFeB, Supermagnetman D1002-10) ---
MAGNET_DIAMETER_MM = 2.0
MAGNET_HEIGHT_MM = 0.5
# N50 remanence midpoint of the 1.40-1.45 T datasheet range; confirm against
# the actual D1002-10 datasheet before trusting absolute field magnitudes.
MAGNET_BR_MT = 1425.0

# --- Deployment z-range ---
SENSOR_PCB_GAP_MM = 6.35   # ~0.25 in PCB standoff below Kresling base
Z_DEPLOYED_MM = 31.65      # sensor-to-magnet distance at full deployment: h1 = 28.3mm (Nature 2025 SI)
                           # + the acrylic spacer sheet between the sensor PCB and Kresling base
                           # (not simply h1 + SENSOR_PCB_GAP_MM -- the acrylic adds extra offset)

# --- Sensor noise (Melexis MLX90393, GAIN_SEL=0x7) ---
# RMS noise stdev read from datasheet Rev 012 Figs 2 & 3 at ~10-14 ms
# conversion time (middle ground between actuation-off window and noise floor).
# TODO: update these figures once the final operating conversion time is
#       confirmed on the actual board -- noise shifts with conversion time.
SENSOR_NOISE_SIGMA_XY_MT = 0.0008   # X and Y axes, mT
SENSOR_NOISE_SIGMA_Z_MT = 0.0014    # Z axis (consistently noisier per datasheet)
