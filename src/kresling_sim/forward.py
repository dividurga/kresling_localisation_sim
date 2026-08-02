from __future__ import annotations

from kresling_sim.kresling import Kresling


def magneticFieldAtPoint(magnet: Kresling, point: tuple[float, float, float]) -> tuple[float, float, float]:
    """Compute the magnetic field (mT) at a given point (mm) due to a Kresling magnet.

    Args:
        magnet (Kresling): The Kresling-cell magnet wrapper.
        point (tuple[float, float, float]): The point in space (mm) where the field is calculated.

    Returns:
        tuple[float, float, float]: The magnetic field vector (mT) at the specified point.
    """
    return tuple(magnet.getB(point))

def __main__():
    # Example usage
    kresling_magnet = Kresling(position=(0, 0, 0))
    point = (1, 1, 1)
    field = magneticFieldAtPoint(kresling_magnet, point)
    print(f"Magnetic field at {point}: {field}")


if __name__ == "__main__":
    __main__()