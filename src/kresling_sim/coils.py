from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import magpylib as magpy
from scipy.spatial.transform import Rotation

from kresling_sim.units import MM_TO_M, T_TO_MT


@dataclass(frozen=True)
class _Winding:
    radius_mm: float
    turns: int
    offset_mm: float


# Table S1 (Nature 2025 SI, S6 / IEEE APEC 2025 Table I): each of the 6 coil
# sides is an outer + inner winding pair in series, sharing one current.
_WINDINGS = {
    "x": {"outer": _Winding(152.0, 56, 97.0), "inner": _Winding(165.0, 64, 72.0)},
    "y": {"outer": _Winding(122.0, 48, 85.0), "inner": _Winding(135.0, 54, 63.0)},
    "z": {"outer": _Winding(100.0, 40, 66.0), "inner": _Winding(111.0, 40, 48.0)},
}

_AXIS_UNIT = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}

# magpy.current.Circle's normal is +z by default; rotate it onto the
# relevant axis so the loop's own axis (not its position) points along x/y/z.
_AXIS_ORIENTATION = {
    "x": Rotation.from_euler("y", 90, degrees=True),
    "y": Rotation.from_euler("x", -90, degrees=True),
    "z": None,
}


class Coils:
    """The 3-D Helmholtz coil system (Manos et al., IEEE APEC 2025 / Nature 2025 SI S6).

    6 independently driven sides (+/- of each axis), each a series pair of
    an outer + inner filamentary winding. An N-turn winding is modeled as a
    single loop carrying current N*i, matching the paper's own C = mu0*N*i/pi.
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
        """Currents in amps, one per coil side. Names match the paper's
        current vector i = [iXF, iXB, iYL, iYR, iZT, iZB]: front/back for x,
        left/right for y, top/bottom for z. Positive current on the '+'-side
        of an axis and '-'-side are independent -- driving both sides with
        the same current gives a near-uniform field; opposite signs give a
        high-gradient field (Fig. 2 of the IEEE paper)."""
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
                for winding in _WINDINGS[axis].values():
                    position = offset_sign * winding.offset_mm * MM_TO_M * axis_vec
                    loop = magpy.current.Circle(
                        diameter=2 * winding.radius_mm * MM_TO_M,
                        current=winding.turns * current,
                        position=position,
                        orientation=orientation,
                    )
                    self.collection.add(loop)

    def getB(self, points):
        """Field in mT at the given point(s) (mm), summed over all 12 windings."""
        points_m = np.asarray(points, dtype=float) * MM_TO_M
        return self.collection.getB(points_m) * T_TO_MT
