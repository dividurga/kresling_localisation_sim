from __future__ import annotations
import numpy as np
import magpylib as magpy

from kresling_sim.config import MAGNET_BR_MT, MAGNET_DIAMETER_MM, MAGNET_HEIGHT_MM
from kresling_sim.units import MM_TO_M, MT_TO_T, T_TO_MT

# re-exported for scripts that imported BR_N50_MT directly from this module
BR_N50_MT = MAGNET_BR_MT


class Kresling:
    """a single cylindrical NdFeB magnet, out-of-plane magnetized. despite
    the name, not kresling-specific -- defaults describe the nature 2025 SI
    (S6) magnet (N50, 2x0.5mm, supermagnetman D1002-10), but the cube_robot
    magnet (N52, 6.35x3.18mm) is the same class, just different
    diameter/height/remanence
    """

    def __init__(
        self,
        position: np.ndarray,
        orientation=None,
        diameter: float = MAGNET_DIAMETER_MM,
        height: float = MAGNET_HEIGHT_MM,
        remanence: float = MAGNET_BR_MT,
    ):
        """position in mm, diameter/height in mm, remanence (Br) in mT.

        magnetization is along the cylinder's own local z (magpylib
        convention), `orientation` (scipy Rotation, or None for default z)
        rotates that.
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
        """field in mT at the given point(s) (mm)."""
        points_m = np.asarray(points, dtype=float) * MM_TO_M
        return self.magnet.getB(points_m) * T_TO_MT
