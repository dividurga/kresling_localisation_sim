from __future__ import annotations

import numpy as np

from kresling_sim.kresling import Kresling


def magneticFieldAtPoint(magnet: Kresling, point: np.ndarray) -> np.ndarray:
    """Compute the magnetic field (mT) at a given point (mm) due to a Kresling magnet.

    Args:
        magnet (Kresling): The Kresling-cell magnet wrapper.
        point (np.ndarray): The point in space (mm) where the field is calculated.

    Returns:
        np.ndarray: The magnetic field vector (mT) at the specified point.
    """
    return magnet.getB(point)

def stacked_forward_prediction(magnet: Kresling, points: np.ndarray) -> np.ndarray:
    """compute the magnetic field in mT of a kresling magnet at multiple points in space in mm
    Args:
        magnet (Kresling): The Kresling-cell magnet wrapper.
        points (np.ndarray (N, 3)): The points in space (mm) where the field is calculated.

    Returns:
        np.ndarray (N, 3): The magnetic field vectors (mT) at the specified points.
    """
    return magnet.getB(points)

def __main__():
    # Example usage
    kresling_magnet = Kresling(position=np.array([0, 0, 0]))
    point = np.array([1, 1, 1])
    field = magneticFieldAtPoint(kresling_magnet, point)
    print(f"Magnetic field at {point}: {field}")


if __name__ == "__main__":
    __main__()