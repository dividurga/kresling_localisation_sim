from __future__ import annotations

import numpy as np

from kresling_sim.kresling import Kresling
from kresling_sim.sensors import Sensor, stacked_measure


def magneticFieldAtPoint(magnet: Kresling, point: np.ndarray) -> np.ndarray:
    """field (mT) at a raw point (mm) due to a kresling magnet. NOT
    orientation-aware -- a global-frame field query, not a sensor
    measurement (see stacked_forward_prediction for that)."""
    return magnet.getB(point)

def stacked_forward_prediction(magnet: Kresling, sensors: list[Sensor]) -> np.ndarray:
    """field (mT) each sensor in `sensors` reads from `magnet`, expressed
    in that sensor's own local (possibly tilted) frame. shape (N,3)."""
    return stacked_measure(sensors, magnet)

def __main__():
    # example usage
    kresling_magnet = Kresling(position=np.array([0, 0, 0]))
    point = np.array([1, 1, 1])
    field = magneticFieldAtPoint(kresling_magnet, point)
    print(f"Magnetic field at {point}: {field}")


if __name__ == "__main__":
    __main__()