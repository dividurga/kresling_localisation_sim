"""physical constants for the kresling setup. mm and mT unless noted, import from here dont hardcode"""
from dataclasses import dataclass


# coil workspace (IEEE APEC 2025) -- usable volume of the 3-D Helmholtz coil system
WORKSPACE_X_MM = 100.0
WORKSPACE_Y_MM = 100.0

# field per amp at origin (mT/A, one coil side). reference only, coils.py uses COIL_WINDINGS.
# from sci adv 2026 SI eq S8 (matches IEEE APEC 2025 eq 9)
COIL_FIELD_PER_AMP_X_MT_A = 0.326
COIL_FIELD_PER_AMP_Y_MT_A = 0.324
COIL_FIELD_PER_AMP_Z_MT_A = 0.321

# max current per submodule (IEEE APEC 2025 III-B). coils are off while sensing, reference only
COIL_MAX_CURRENT_A = 100.0

# coil windings (IEEE APEC 2025 table I), 6 sides each outer+inner in series.
# radius = mean radius, offset = distance from origin along the axis
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

# sensor to magnet distance fully deployed: h1=28.3mm (nature 2025 SI) + acrylic spacer.
# not just h1 + SENSOR_PCB_GAP_MM, the acrylic adds extra
Z_DEPLOYED_MM = 31.65

# MLX90393 noise (GAIN_SEL=0x7), rms from datasheet rev 012 figs 2 and 3 at ~10-14ms conversion
# TODO redo for the real conversion time, noise changes with it (see hardware_notes.md)
SENSOR_NOISE_SIGMA_XY_MT = 0.0008   # X and Y axes, mT
SENSOR_NOISE_SIGMA_Z_MT = 0.0014    # Z axis, noisier
