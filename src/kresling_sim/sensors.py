"""Hall-sensor placement: a single-sensor primitive (`Sensor`), a
parametrized rectangular grid built from repeated calls to that primitive
(`sensor_grid`), and save/load helpers for layouts produced interactively
(see scripts/sensor_gui.py).

rn ive written some of the code for placement with a tilt but the crb stuff
and other stuff is not yet integrated with the tilt. so for now, just skip untilted
"""
from __future__ import annotations

import numpy as np
import magpylib as magpy
from scipy.spatial.transform import Rotation

from kresling_sim.kresling import Kresling
from kresling_sim.units import MM_TO_M, T_TO_MT


def tilt_rotation(polar: float, azimuth: float) -> Rotation:
    """The minimal (no extra roll) rotation tilting local z onto the
    direction given by spherical angles polar (from +z) and azimuth
    (around z), both radians. polar=0 gives the identity regardless of
    azimuth. This is the same "tilt z onto a direction" convention used in
    coils.py's _AXIS_ORIENTATION, generalized from the 3 cardinal axes to
    an arbitrary direction.
    """
    if polar == 0.0:
        return Rotation.identity()
    axis = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    return Rotation.from_rotvec(axis * polar)


class Sensor:
    """A single 3-axis Hall sensor: a position and orientation. Wraps
    magpy.Sensor so a tilted sensor's local-frame reading (its own x/y/z
    measurement axes need not align with global x/y/z) uses magpylib's own
    rotation handling, the same way Kresling wraps magpy.magnet.Cylinder.
    """

    def __init__(self, position: np.ndarray, orientation: Rotation | None = None):
        """position in mm; orientation a scipy Rotation describing how the
        sensor's local measurement axes are rotated relative to global
        (None/identity = untilted, local axes aligned with global x/y/z)."""
        self.position = np.asarray(position, dtype=float)
        self.orientation = orientation if orientation is not None else Rotation.identity()
        self._sensor = magpy.Sensor(position=self.position * MM_TO_M, orientation=self.orientation)

    @classmethod
    def at_angle(cls, position: np.ndarray, polar: float = 0.0, azimuth: float = 0.0) -> "Sensor":
        """Convenience constructor from spherical tilt angles (radians)
        instead of building a Rotation yourself."""
        return cls(position, tilt_rotation(polar, azimuth))

    def measure(self, magnet: Kresling) -> np.ndarray:
        """Local-frame field (mT) this sensor reads from `magnet`."""
        return magpy.getB(magnet.magnet, self._sensor) * T_TO_MT


def sensor_grid(
    num_across: int,
    num_below: int,
    sep_horizontal: float,
    sep_vertical: float,
    z: float = 0.0,
    polar: float = 0.0,
    azimuth: float = 0.0,
    center: bool = True,
) -> list[Sensor]:
    """A num_below (rows, y-direction) x num_across (columns, x-direction)
    rectangular grid of Sensors in the z=z plane (mm), all sharing one tilt
    (polar, azimuth radians). Built from repeated Sensor.at_angle(...)
    calls -- the single-sensor placement primitive -- not a separate
    implementation. `center=True` (default) centers the grid on (x=0,y=0);
    otherwise the grid's corner sensor sits at (0,0,z).
    """
    orientation = tilt_rotation(polar, azimuth)
    xs = np.arange(num_across, dtype=float) * sep_horizontal
    ys = np.arange(num_below, dtype=float) * sep_vertical
    if center:
        xs = xs - xs.mean()
        ys = ys - ys.mean()

    sensors = []
    for y in ys:
        for x in xs:
            sensors.append(Sensor(position=np.array([x, y, z]), orientation=orientation))
    return sensors


def positions_array(sensors: list[Sensor]) -> np.ndarray:
    """(N,3) mm positions -- the bridge to fisher.py/montecarlo.py, which
    currently take plain position arrays (see module docstring)."""
    return np.array([s.position for s in sensors])


def save_sensors(sensors: list[Sensor], path: str) -> None:
    """Save positions + orientations (as rotation vectors, radians) to an
    .npz file, for a GUI-designed or script-generated layout to be
    reloaded later by name rather than regenerated."""
    positions = positions_array(sensors)
    rotvecs = np.array([s.orientation.as_rotvec() for s in sensors])
    np.savez(path, positions=positions, rotvecs=rotvecs)


def load_sensors(path: str) -> list[Sensor]:
    """Load a sensor layout previously written by save_sensors."""
    data = np.load(path)
    return [
        Sensor(position=pos, orientation=Rotation.from_rotvec(rotvec))
        for pos, rotvec in zip(data["positions"], data["rotvecs"])
    ]
