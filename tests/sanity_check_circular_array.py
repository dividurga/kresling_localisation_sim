"""sanity checks for circular_array.py, asserts + printed values, not pytest

Run with: uv run python tests/sanity_check_circular_array.py
"""
from __future__ import annotations

import numpy as np

from kresling_sim.circular_array import RadiusSpec, RingSpec, concentric_rings, shared_spokes, uniform_ring_radii


def check_single_sensor_edge_case() -> None:
    """count 1 on a partial arc shouldnt crash (divide by 0) and sits at the offset angle"""
    sensors = concentric_rings([RingSpec(radius=15.0, count=1, offset_deg=45.0, arc_deg=90.0)])
    expected = np.array([15.0 * np.cos(np.radians(45.0)), 15.0 * np.sin(np.radians(45.0)), 0.0])
    print(f"single-sensor position: {sensors[0].position}  (expect {expected})")
    assert len(sensors) == 1
    assert np.allclose(sensors[0].position, expected)


def check_partial_arc_endpoints() -> None:
    """count 5, arc 90, offset 0 -> first at 0, last at 90deg (not 72, the full circle formula)"""
    sensors = concentric_rings([RingSpec(radius=10.0, count=5, offset_deg=0.0, arc_deg=90.0)])
    first_theta = np.degrees(np.arctan2(sensors[0].position[1], sensors[0].position[0]))
    last_theta = np.degrees(np.arctan2(sensors[-1].position[1], sensors[-1].position[0]))
    print(f"partial-arc first/last angles: {first_theta:.2f}deg, {last_theta:.2f}deg (expect 0, 90)")
    assert abs(first_theta - 0.0) < 1e-6
    assert abs(last_theta - 90.0) < 1e-6


def check_full_circle_no_duplicate_endpoint() -> None:
    """count 6, arc 360 -> 60deg steps, no duplicate at 0 and 360"""
    sensors = concentric_rings([RingSpec(radius=10.0, count=6, offset_deg=0.0, arc_deg=360.0)])
    positions = np.array([s.position for s in sensors])
    pairwise_min_dist = min(
        np.linalg.norm(positions[i] - positions[j])
        for i in range(len(positions)) for j in range(len(positions)) if i != j
    )
    print(f"full-circle min pairwise sensor distance: {pairwise_min_dist:.3f}mm (expect ~10mm, not ~0)")
    assert pairwise_min_dist > 5.0, "no two sensors should coincide on a full circle"


def check_uniform_pitch_vs_custom_equivalence() -> None:
    """uniform and custom RadiusSpec with the same radii give identical rings"""
    uniform = RadiusSpec(mode="uniform", inner_radius=10.0, radial_pitch=5.0, ring_count=3)
    custom = RadiusSpec(mode="custom", custom_radii=[10.0, 15.0, 20.0])
    print(f"uniform radii: {uniform.radii()}  custom radii: {custom.radii()}")
    assert uniform.radii() == custom.radii()

    rings_uniform = [RingSpec(radius=r, count=6) for r in uniform.radii()]
    rings_custom = [RingSpec(radius=r, count=6) for r in custom.radii()]
    pos_uniform = np.array([s.position for s in concentric_rings(rings_uniform)])
    pos_custom = np.array([s.position for s in concentric_rings(rings_custom)])
    assert np.allclose(pos_uniform, pos_custom)


def check_shared_spokes_cross_product() -> None:
    """shared_spokes radii [10,20,30] count 4 -> 12 sensors, same angles at every radius"""
    sensors = shared_spokes(radii=[10.0, 20.0, 30.0], count=4)
    print(f"shared_spokes sensor count: {len(sensors)} (expect 12)")
    assert len(sensors) == 12

    by_radius: dict[float, list[float]] = {}
    for s in sensors:
        r = float(np.linalg.norm(s.position[:2]))
        theta = float(np.degrees(np.arctan2(s.position[1], s.position[0])))
        by_radius.setdefault(round(r), []).append(round(theta, 3))
    angle_sets = [sorted(v) for v in by_radius.values()]
    assert all(a == angle_sets[0] for a in angle_sets), "every radius must share the same angular pattern"


def check_validation_rejections() -> None:
    """bad inputs raise ValueError"""
    bad_inputs = [
        lambda: RadiusSpec(mode="uniform", inner_radius=-1.0, radial_pitch=5.0, ring_count=3),
        lambda: RadiusSpec(mode="uniform", inner_radius=10.0, radial_pitch=0.0, ring_count=3),
        lambda: RadiusSpec(mode="uniform", inner_radius=10.0, radial_pitch=5.0, ring_count=0),
        lambda: RadiusSpec(mode="custom", custom_radii=[10.0, float("nan")]),
        lambda: RadiusSpec(mode="custom", custom_radii=[10.0, -5.0]),
        lambda: RingSpec(radius=5.0, count=0),
        lambda: RingSpec(radius=0.0, count=4),
        lambda: RingSpec(radius=5.0, count=4, arc_deg=0.0),
        lambda: RingSpec(radius=5.0, count=4, arc_deg=361.0),
    ]
    for i, make_bad in enumerate(bad_inputs):
        try:
            make_bad()
            raise AssertionError(f"bad_inputs[{i}] should have raised ValueError but didn't")
        except ValueError as e:
            print(f"  bad_inputs[{i}] correctly rejected: {e}")


def check_orientation_direction_sanity() -> None:
    """radial: local z along (cos t, sin t, 0). tangential: local z perpendicular to that"""
    theta_deg = 30.0
    radial_sensor = concentric_rings([RingSpec(radius=20.0, count=1, offset_deg=theta_deg)], orientation_mode="radial")[0]
    tangential_sensor = concentric_rings(
        [RingSpec(radius=20.0, count=1, offset_deg=theta_deg)], orientation_mode="tangential"
    )[0]

    radial_dir = np.array([np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg)), 0.0])
    radial_local_z = radial_sensor.orientation.apply([0, 0, 1])
    tangential_local_z = tangential_sensor.orientation.apply([0, 0, 1])

    print(f"radial local-z:     {radial_local_z.round(4)}  (expect ~{radial_dir.round(4)})")
    print(f"tangential local-z: {tangential_local_z.round(4)}  (expect perpendicular to radial dir)")
    assert np.allclose(radial_local_z, radial_dir, atol=1e-6)
    assert abs(np.dot(tangential_local_z, radial_dir)) < 1e-6


if __name__ == "__main__":
    check_single_sensor_edge_case()
    check_partial_arc_endpoints()
    check_full_circle_no_duplicate_endpoint()
    check_uniform_pitch_vs_custom_equivalence()
    check_shared_spokes_cross_product()
    check_validation_rejections()
    check_orientation_direction_sanity()
    print("all sanity checks passed")
