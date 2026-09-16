from __future__ import annotations
import numpy as np
import magpylib as magpy

from kresling_sim.config import MAGNET_BR_MT, MAGNET_DIAMETER_MM, MAGNET_HEIGHT_MM
from kresling_sim.units import MM_TO_M, MT_TO_T, T_TO_MT

# Re-exported for scripts that imported BR_N50_MT directly from this module.
BR_N50_MT = MAGNET_BR_MT


class Kresling:
    """A single cylindrical NdFeB magnet, out-of-plane magnetized -- the
    forward-model source for any target whose diameter/height/remanence are
    supplied. Despite the name, not specific to the Kresling cell: the
    default arguments describe the Nature 2025 SI (S6) magnet (N50 grade,
    2 mm x 0.5 mm, Supermagnetman D1002-10, one per individual chiral cell),
    but cube_robot_config.py's IEEE APEC 2025 magnet (N52, 6.35 mm x 3.18 mm)
    is the same class with different diameter/height/remanence.
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
