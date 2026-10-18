from __future__ import annotations
import numpy as np
import magpylib as magpy

from kresling_sim.config import MAGNET_BR_MT, MAGNET_DIAMETER_MM, MAGNET_HEIGHT_MM
from kresling_sim.units import MM_TO_M, MT_TO_T, T_TO_MT

# re-exported for scripts that imported BR_N50_MT directly from this module
BR_N50_MT = MAGNET_BR_MT


class Kresling:
    """one or two identical disc magnets (separation_mm), magnetized along local z.
    not kresling specific, defaults are the nature 2025 SI magnet (N50, 2x0.5mm),
    cube_robot and kresling_2026_supp just use different params
    """

    def __init__(
        self,
        position: np.ndarray,
        orientation=None,
        diameter: float = MAGNET_DIAMETER_MM,
        height: float = MAGNET_HEIGHT_MM,
        remanence: float = MAGNET_BR_MT,
        separation_mm: float = 0.0,
    ):
        """position, diameter, height in mm, remanence in mT. orientation (scipy Rotation) rotates local z.
        separation_mm 0 = one magnet, >0 = two magnets at +/- sep/2 along local z moving as one rigid body
        """
        self.diameter = diameter
        self.height = height
        self.remanence = remanence
        self.separation_mm = separation_mm

        if separation_mm == 0.0:
            self.magnet = magpy.magnet.Cylinder(
                dimension=(diameter * MM_TO_M, height * MM_TO_M),
                polarization=(0, 0, remanence * MT_TO_T),
                position=np.asarray(position, dtype=float) * MM_TO_M,
                orientation=orientation,
            )
        else:
            offset_m = separation_mm / 2 * MM_TO_M
            mag_top = magpy.magnet.Cylinder(
                dimension=(diameter * MM_TO_M, height * MM_TO_M),
                polarization=(0, 0, remanence * MT_TO_T),
                position=(0, 0, offset_m),
            )
            mag_bottom = magpy.magnet.Cylinder(
                dimension=(diameter * MM_TO_M, height * MM_TO_M),
                polarization=(0, 0, remanence * MT_TO_T),
                position=(0, 0, -offset_m),
            )
            self.magnet = magpy.Collection(mag_top, mag_bottom)
            self.magnet.position = np.asarray(position, dtype=float) * MM_TO_M
            self.magnet.orientation = orientation

    def getB(self, points):
        """field in mT at the given point(s) (mm)."""
        points_m = np.asarray(points, dtype=float) * MM_TO_M
        return self.magnet.getB(points_m) * T_TO_MT
