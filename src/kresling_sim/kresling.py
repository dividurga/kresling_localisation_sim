from __future__ import annotations
import numpy as np
import magpylib as magpy

from kresling_sim.config import MAGNET_BR_MT, MAGNET_DIAMETER_MM, MAGNET_HEIGHT_MM
from kresling_sim.units import MM_TO_M, MT_TO_T, T_TO_MT

# Re-exported for scripts that imported BR_N50_MT directly from this module.
BR_N50_MT = MAGNET_BR_MT


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
        diameter: float = MAGNET_DIAMETER_MM,
        height: float = MAGNET_HEIGHT_MM,
        remanence: float = MAGNET_BR_MT,
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
