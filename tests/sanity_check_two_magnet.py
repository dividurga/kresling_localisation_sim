"""sanity checks for two magnet support (separation_mm) and the kresling_2026_supp target, asserts + printed values, not pytest

Run with: uv run python tests/sanity_check_two_magnet.py
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from kresling_sim import fisher, kresling_2026_supp_config
from kresling_sim.kresling import Kresling
from kresling_sim.montecarlo import DEFAULT_NOISE, run_monte_carlo
from kresling_sim.sensors import sensor_grid


def check_separation_zero_matches_single_magnet() -> None:
    """separation_mm 0 is identical to not passing it (kresling/cube_robot unchanged)"""
    pos = np.array([1.0, 2.0, 15.0])
    point = np.array([5.0, 3.0, 25.0])
    k_default = Kresling(position=pos)
    k_explicit_zero = Kresling(position=pos, separation_mm=0.0)
    b_default = k_default.getB(point)
    b_explicit = k_explicit_zero.getB(point)
    print(f"default: {b_default}  explicit separation_mm=0.0: {b_explicit}")
    assert np.array_equal(b_default, b_explicit), "separation_mm=0.0 must match the old default exactly"


def check_two_magnet_matches_manual_superposition() -> None:
    """two magnets = sum of two single magnets at the rotated +/- sep/2 offsets (checks the Collection transform)"""
    sep = 12.0
    center = np.array([0.0, 0.0, 12.35])
    orientation = Rotation.from_euler("x", -90, degrees=True)  # local z -> +y
    point = np.array([5.0, 3.0, 25.0])
    magnet_kwargs = dict(diameter=4.0, height=0.5, remanence=1319.0)

    k_two = Kresling(position=center, orientation=orientation, separation_mm=sep, **magnet_kwargs)

    k_a = Kresling(position=center + orientation.apply([0, 0, sep / 2]), orientation=orientation, **magnet_kwargs)
    k_b = Kresling(position=center + orientation.apply([0, 0, -sep / 2]), orientation=orientation, **magnet_kwargs)
    manual_sum = k_a.getB(point) + k_b.getB(point)

    b_two = k_two.getB(point)
    print(f"two-magnet Collection: {b_two}")
    print(f"manual superposition:  {manual_sum}")
    assert np.allclose(b_two, manual_sum), "two-magnet Collection must equal manual superposition"


def check_orientation_axis_points_along_y() -> None:
    """SLIDING_ORIENTATION puts local z on global +y (sci adv 2026 SI M=[0,1,0])"""
    axis = kresling_2026_supp_config.SLIDING_ORIENTATION.apply([0, 0, 1])
    print(f"sliding orientation axis: {axis} (expect ~[0,1,0])")
    assert np.allclose(axis, [0.0, 1.0, 0.0], atol=1e-9)


def check_kresling_2026_supp_end_to_end() -> None:
    """CRB and small MC with kresling_2026_supp MAGNET_KWARGS through the normal pipeline, no special casing"""
    sensors = sensor_grid(num_across=5, num_below=5, sep_horizontal=12.0, sep_vertical=12.0)
    true_position = np.array([2.0, -1.0, kresling_2026_supp_config.Z_CENTER_MM])

    jacobian = fisher.position_jacobian(true_position, sensors, magnet_kwargs=kresling_2026_supp_config.MAGNET_KWARGS)
    fim = fisher.fisher_information_matrix(jacobian, DEFAULT_NOISE)
    crlb = fisher.crlb(fim)
    print(f"kresling_2026_supp CRLB (x,y,z) mm: {crlb}")
    assert np.all(np.isfinite(crlb)) and np.all(crlb > 0)

    result = run_monte_carlo(true_position, sensors, n_trials=50, rng=np.random.default_rng(0),
                              magnet_kwargs=kresling_2026_supp_config.MAGNET_KWARGS)
    print(f"kresling_2026_supp MC: converged {result.converged.sum()}/50, "
          f"empirical RMSE={result.rmse_mm:.4f}mm, CRB RMSE={result.crb_rmse_mm:.4f}mm")
    assert result.converged.sum() >= 45, "almost all trials should converge for this well-conditioned target"
    assert result.rmse_mm < 3 * result.crb_rmse_mm, "empirical RMSE should track the CRB, not blow up"


if __name__ == "__main__":
    check_separation_zero_matches_single_magnet()
    check_two_magnet_matches_manual_superposition()
    check_orientation_axis_points_along_y()
    check_kresling_2026_supp_end_to_end()
    print("all sanity checks passed")
