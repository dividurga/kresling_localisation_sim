"""Circular/polar sensor placement: concentric-ring and shared-spoke
patterns built on top of the single-sensor primitive in sensors.py
(Sensor, tilt_rotation), the same way sensor_grid builds a rectangular
grid. No center/origin parameter -- inherits the implicit (0,0) xy origin
and z=0 sensor-plane convention already used by sensor_grid/config.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from kresling_sim.sensors import Sensor, tilt_rotation


@dataclass
class RadiusSpec:
    """Either uniform-pitch (radius[j] = inner_radius + j*radial_pitch,
    derived on demand via radii() -- stays parametric, not flattened) or
    an explicit custom radii list."""

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
        """Resolved list of ring radii for this spec."""
        if self.mode == "uniform":
            return uniform_ring_radii(self.inner_radius, self.radial_pitch, self.ring_count)
        return list(self.custom_radii)


@dataclass
class RingSpec:
    """One ring's fully-resolved angular pattern -- radius/count/offset/arc
    always explicit here regardless of which RadiusSpec mode produced
    `radius`."""

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
    """count==1 -> single angle at offset_rad, no division. Full circle
    (arc_rad == 2*pi) -> step = arc_rad/count, no duplicated 0/2pi
    endpoint. Partial arc (arc_rad < 2*pi) -> step = arc_rad/(count-1), so
    the first angle is offset_rad and the last is offset_rad + arc_rad."""
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
    """General primitive: independent count/radius/offset/arc per ring.
    For each sensor at angle theta_i (radians, from _ring_angles_rad),
    call tilt_rotation(polar, azimuth) once per sensor with:
      radial:     polar=tilt_polar_deg (rad), azimuth = theta_i + orientation_offset
      tangential: polar=tilt_polar_deg (rad), azimuth = theta_i + 90deg + orientation_offset
      fixed:      polar=fixed_polar_deg (rad), azimuth = fixed_azimuth_deg + orientation_offset (same for every sensor)
    position = (radius*cos(theta_i), radius*sin(theta_i), z).
    No modification to tilt_rotation needed -- just call it per-sensor
    instead of once per array (contrast sensor_grid, which reuses one
    Rotation for the whole grid)."""
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
    """One shared angular pattern repeated at every radius -- build
    [RingSpec(r, count, offset_deg, arc_deg) for r in radii] and delegate
    to concentric_rings(..., z=z, **orientation_kwargs). Not a separate
    algorithm, a thin composition (mirrors how sensor_grid is built from
    repeated Sensor.at_angle calls)."""
    rings = [RingSpec(radius=r, count=count, offset_deg=offset_deg, arc_deg=arc_deg) for r in radii]
    return concentric_rings(rings, z=z, **orientation_kwargs)
