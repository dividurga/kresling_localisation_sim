"""circular sensor placement, concentric rings and shared spokes, built on sensors.py.
no center param, origin is (0,0) and sensor plane is z=0 like sensor_grid
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from kresling_sim.sensors import Sensor, tilt_rotation


@dataclass
class RadiusSpec:
    """uniform pitch (inner_radius + j*radial_pitch) or an explicit custom radii list"""

    mode: Literal["uniform", "custom"]
    inner_radius: float | None = None
    radial_pitch: float | None = None
    ring_count: int | None = None
    custom_radii: list[float] | None = None

    def __post_init__(self) -> None:
        if self.mode == "uniform":
            if self.inner_radius is None or self.radial_pitch is None or self.ring_count is None:
                raise ValueError("uniform mode requires inner_radius, radial_pitch, and ring_count")
            if self.inner_radius <= 0:
                raise ValueError(f"inner_radius must be > 0, got {self.inner_radius}")
            if self.radial_pitch <= 0:
                raise ValueError(f"radial_pitch must be > 0, got {self.radial_pitch}")
            if self.ring_count < 1:
                raise ValueError(f"ring_count must be >= 1, got {self.ring_count}")
        elif self.mode == "custom":
            if not self.custom_radii:
                raise ValueError("custom mode requires a non-empty custom_radii list")
            for r in self.custom_radii:
                if not np.isfinite(r) or r <= 0:
                    raise ValueError(f"custom_radii entries must all be finite and > 0, got {r}")
        else:
            raise ValueError(f"mode must be 'uniform' or 'custom', got {self.mode!r}")

    def radii(self) -> list[float]:
        """ring radii"""
        if self.mode == "uniform":
            return uniform_ring_radii(self.inner_radius, self.radial_pitch, self.ring_count)
        return list(self.custom_radii)


@dataclass
class RingSpec:
    """one ring, radius/count/offset/arc all explicit"""

    radius: float
    count: int
    offset_deg: float = 0.0
    arc_deg: float = 360.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError(f"radius must be finite and > 0, got {self.radius}")
        if self.count < 1:
            raise ValueError(f"count must be >= 1, got {self.count}")
        if not (0 < self.arc_deg <= 360):
            raise ValueError(f"arc_deg must be in (0, 360], got {self.arc_deg}")


def uniform_ring_radii(inner_radius: float, radial_pitch: float, ring_count: int) -> list[float]:
    """radius[j] = inner_radius + j*radial_pitch, for j in [0, ring_count)."""
    list_of_radii = [inner_radius + j * radial_pitch for j in range(ring_count)]
    return list_of_radii


def _ring_angles_rad(count: int, arc_rad: float, offset_rad: float) -> np.ndarray:
    """count 1 -> single angle. full circle -> step arc/count (no duplicate 0/2pi).
    partial arc -> step arc/(count-1) so last angle is offset + arc"""
    if count == 1:
        return np.array([offset_rad])
    full_circle = np.isclose(arc_rad, 2 * np.pi)
    step = arc_rad / count if full_circle else arc_rad / (count - 1)
    return offset_rad + step * np.arange(count)


def concentric_rings(
    rings: list[RingSpec],
    z: float = 0.0,
    orientation_mode: Literal["radial", "tangential", "fixed"] = "radial",
    tilt_polar_deg: float = 90.0,
    fixed_azimuth_deg: float = 0.0,
    fixed_polar_deg: float = 0.0,
    orientation_offset_deg: float = 0.0,
) -> list[Sensor]:
    """independent count/radius/offset/arc per ring. orientation per sensor via tilt_rotation:
      radial:     polar=tilt_polar, azimuth=theta + offset
      tangential: polar=tilt_polar, azimuth=theta + 90deg + offset
      fixed:      polar=fixed_polar, azimuth=fixed_azimuth + offset
    position is (r*cos(theta), r*sin(theta), z)"""
    if orientation_mode not in ("radial", "tangential", "fixed"):
        raise ValueError(f"orientation_mode must be 'radial', 'tangential', or 'fixed', got {orientation_mode!r}")

    orientation_offset_rad = np.radians(orientation_offset_deg)
    tilt_polar_rad = np.radians(tilt_polar_deg)
    fixed_polar_rad = np.radians(fixed_polar_deg)
    fixed_azimuth_rad = np.radians(fixed_azimuth_deg)

    sensors = []
    for ring in rings:
        angles = _ring_angles_rad(ring.count, np.radians(ring.arc_deg), np.radians(ring.offset_deg))
        for theta in angles:
            x = ring.radius * np.cos(theta)
            y = ring.radius * np.sin(theta)

            if orientation_mode == "radial":
                polar, azimuth = tilt_polar_rad, theta + orientation_offset_rad
            elif orientation_mode == "tangential":
                polar, azimuth = tilt_polar_rad, theta + np.pi / 2 + orientation_offset_rad
            else:  # fixed
                polar, azimuth = fixed_polar_rad, fixed_azimuth_rad + orientation_offset_rad

            orientation = tilt_rotation(polar, azimuth)
            sensors.append(Sensor(position=np.array([x, y, z]), orientation=orientation))
    return sensors


def shared_spokes(
    radii: list[float],
    count: int,
    offset_deg: float = 0.0,
    arc_deg: float = 360.0,
    z: float = 0.0,
    **orientation_kwargs,
) -> list[Sensor]:
    """same angular pattern at every radius, thin wrapper over concentric_rings"""
    rings = [RingSpec(radius=r, count=count, offset_deg=offset_deg, arc_deg=arc_deg) for r in radii]
    return concentric_rings(rings, z=z, **orientation_kwargs)
