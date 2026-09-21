from __future__ import annotations

import numpy as np
import magpylib as magpy
from scipy.spatial.transform import Rotation

from kresling_sim.config import COIL_WINDINGS
from kresling_sim.units import MM_TO_M, T_TO_MT

_AXIS_UNIT = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}

# magpy.current.Circle's normal is +z by default; rotate it onto the
# relevant axis so the loop's own axis (not its position) points along x/y/z.
_AXIS_ORIENTATION = {
    "x": Rotation.from_euler("y", 90, degrees=True),
    "y": Rotation.from_euler("x", -90, degrees=True),
    "z": None,
}


class Coils:
    """the 3-D Helmholtz coil system (Manos et al., IEEE APEC 2025 / nature 2025 SI S6).

    6 independently driven sides (+/- of each axis), each a series pair of
    an outer + inner filamentary winding. an N-turn winding is modeled as a
    single loop carrying current N*i, matching the paper's own C = mu0*N*i/pi
    """

    def __init__(
        self,
        i_xf: float = 0.0,
        i_xb: float = 0.0,
        i_yl: float = 0.0,
        i_yr: float = 0.0,
        i_zt: float = 0.0,
        i_zb: float = 0.0,
    ):
        """currents in amps, one per coil side. names match the paper's
        current vector i = [iXF, iXB, iYL, iYR, iZT, iZB]: front/back for x,
        left/right for y, top/bottom for z. '+' and '-' side of an axis are
        independent -- same current on both gives a near-uniform field,
        opposite signs give a high-gradient field (Fig. 2 of the IEEE paper)"""
        self.currents = {
            "x": {"+": i_xf, "-": i_xb},
            "y": {"+": i_yl, "-": i_yr},
            "z": {"+": i_zt, "-": i_zb},
        }
        self.collection = magpy.Collection()
        self._build()

    def _build(self) -> None:
        for axis, sides in self.currents.items():
            axis_vec = np.array(_AXIS_UNIT[axis])
            orientation = _AXIS_ORIENTATION[axis]
            for sign, current in sides.items():
                offset_sign = 1.0 if sign == "+" else -1.0
                for winding in COIL_WINDINGS[axis].values():
                    position = offset_sign * winding.offset_mm * MM_TO_M * axis_vec
                    loop = magpy.current.Circle(
                        diameter=2 * winding.radius_mm * MM_TO_M,
                        current=winding.turns * current,
                        position=position,
                        orientation=orientation,
                    )
                    self.collection.add(loop)

    def getB(self, points):
        """field in mT at the given point(s) (mm), summed over all 12 windings."""
        points_m = np.asarray(points, dtype=float) * MM_TO_M
        return self.collection.getB(points_m) * T_TO_MT
