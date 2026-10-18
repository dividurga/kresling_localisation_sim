"""hall sensor placement. Sensor, sensor_grid, and save/load for layouts

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
    """rotation tilting local z onto (polar from +z, azimuth around z), radians, no roll.
    polar 0 is identity. same convention as coils.py
    """
    if polar == 0.0:
        return Rotation.identity()
    axis = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    return Rotation.from_rotvec(axis * polar)


class Sensor:
    """3 axis hall sensor, position + orientation. wraps magpy.Sensor for local frame readings
    """

    def __init__(self, position: np.ndarray, orientation: Rotation | None = None):
        """position in mm, orientation a scipy Rotation (None = untilted)"""
        self.position = np.asarray(position, dtype=float)
        self.orientation = orientation if orientation is not None else Rotation.identity()
        self._sensor = magpy.Sensor(position=self.position * MM_TO_M, orientation=self.orientation)

    @classmethod
    def at_angle(cls, position: np.ndarray, polar: float = 0.0, azimuth: float = 0.0) -> "Sensor":
        """constructor from tilt angles (radians)"""
        return cls(position, tilt_rotation(polar, azimuth))

    def measure(self, magnet: Kresling) -> np.ndarray:
        """local frame field (mT) from magnet"""
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
    """num_below (rows, y) x num_across (cols, x) grid at z, one shared tilt.
    center=True centers on (0,0), else corner sensor is at (0,0,z)
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
    """(N,3) positions in mm"""
    return np.array([s.position for s in sensors])


def save_sensors(sensors: list[Sensor], path: str) -> None:
    """save positions + rotvecs (radians) to .npz"""
    positions = positions_array(sensors)
    rotvecs = np.array([s.orientation.as_rotvec() for s in sensors])
    np.savez(path, positions=positions, rotvecs=rotvecs)


def load_sensors(path: str) -> list[Sensor]:
    """load a layout from save_sensors"""
    data = np.load(path)
    return [
        Sensor(position=pos, orientation=Rotation.from_rotvec(rotvec))
        for pos, rotvec in zip(data["positions"], data["rotvecs"])
    ]

def stacked_measure(sensors: list[Sensor], magnet: Kresling) -> np.ndarray:
    """field (mT) at every sensor in its local frame, always (N,3).
    reshape needed, magpy squeezes when N is 1
    """
    magpy_sensors = [s._sensor for s in sensors]
    return (magpy.getB(magnet.magnet, magpy_sensors) * T_TO_MT).reshape(len(sensors), 3)