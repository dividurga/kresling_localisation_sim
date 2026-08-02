from __future__ import annotations
import numpy as np
import magpylib as magpy

from kresling_sim.units import MM_TO_M, MT_TO_T, T_TO_MT

# N50 remanence Br is typically 1.40-1.45 T; not stated in the paper itself
# (Nature 2025 SI, S6) beyond the grade name and geometry. Using the midpoint
# of the datasheet range as a placeholder - confirm against the actual
# Supermagnetman D1002-10 datasheet before trusting field magnitudes.
BR_N50_MT = 1425.0


class Kresling:
    """A single NdFeB cylindrical magnet glued to one Kresling cell.

    Matches the paper's magnet (Nature 2025 SI, S6): N50 grade, out-of-plane
    magnetization, 2 mm diameter x 0.5 mm thickness (Supermagnetman D1002-10).
    One such magnet is installed per individual chiral cell.
    """

    def __init__(
        self,
        position: np.ndarray,
        orientation=None,
        diameter: float = 2.0,
        height: float = 0.5,
        remanence: float = BR_N50_MT,
    ):
        """position in mm; diameter/height in mm; remanence (Br) in mT.

        Magnetization is out-of-plane, i.e. along the cylinder's own
        symmetry axis (magpylib's local z for a Cylinder), which is set via
        `orientation` (a scipy Rotation, or None for the default z-axis).
        """
        self.diameter = diameter
        self.height = height
        self.remanence = remanence
        self.magnet = magpy.magnet.Cylinder(
            dimension=(diameter * MM_TO_M, height * MM_TO_M),
            polarization=(0, 0, remanence * MT_TO_T),
            position=np.asarray(position, dtype=float) * MM_TO_M,
            orientation=orientation,
        )

    def getB(self, points):
        """Field in mT at the given point(s) (mm)."""
        points_m = np.asarray(points, dtype=float) * MM_TO_M
        return self.magnet.getB(points_m) * T_TO_MT
