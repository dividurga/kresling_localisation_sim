from __future__ import annotations
import magpylib as magpy

from kresling_sim.kresling import Kresling


def magneticFieldAtPoint(magnet: magpy.magnet.Cylinder, point: tuple[float, float, float]) -> tuple[float, float, float]:
    """Compute the magnetic field at a given point due to a cylindrical magnet.

    Args:
        magnet (magpy.magnet.Cylinder): The cylindrical magnet object.
        point (tuple[float, float, float]): The point in space where the magnetic field is calculated.

    Returns:
        tuple[float, float, float]: The magnetic field vector at the specified point.
    """
    return tuple(magnet.getB(point))

def __main__():
    # Example usage
    kresling_magnet = Kresling(position=(0, 0, 0))
    point = (1, 1, 1)
    field = magneticFieldAtPoint(kresling_magnet.magnet, point)
    print(f"Magnetic field at {point}: {field}")


if __name__ == "__main__":
    __main__()