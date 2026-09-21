"""Sanity checks for sensor-orientation wiring through forward/fisher/
montecarlo. Not a pytest suite -- just asserts + printed values to eyeball,
same convention as tests/sanity_check_coils_actuation.py.

Run with: uv run python tests/sanity_check_orientation.py
"""
from __future__ import annotations

import numpy as np

from kresling_sim import fisher
from kresling_sim.kresling import Kresling
from kresling_sim.montecarlo import run_monte_carlo
from kresling_sim.sensors import Sensor, positions_array, sensor_grid, stacked_measure, tilt_rotation


def check_identity_orientation_regression() -> None:
    """Untilted sensors must give byte-for-byte the same field as the old
    orientation-blind path (magnet.getB on a raw position array) -- the
    safety net for every existing grid-based caller."""
    sensors = sensor_grid(num_across=4, num_below=4, sep_horizontal=12.0, sep_vertical=12.0)
    points = positions_array(sensors)
    magnet = Kresling(position=np.array([2.0, -1.0, 20.0]))

    via_sensors = stacked_measure(sensors, magnet)
    via_raw = magnet.getB(points)
    max_diff = float(np.max(np.abs(via_sensors - via_raw)))
    print(f"identity-orientation max abs diff vs raw getB: {max_diff:.2e} mT")
    assert max_diff < 1e-12, "untilted Sensor path must exactly match the old raw-points path"


def check_tilted_sensor_reads_zero_on_perpendicular_axis() -> None:
    """A sensor sitting on-axis above the magnet reads pure Bz globally
    (Bx=By=0 by symmetry of an out-of-plane dipole). tilt_rotation(pi/2, 0)
    rotates local z onto global +x -- so the tilted sensor's local-z
    channel should read ~0 (no global-x field component there), while the
    field magnitude is preserved (just redistributed across the other
    local axes)."""
    magnet = Kresling(position=np.array([0.0, 0.0, 0.0]))
    pos = np.array([0.0, 0.0, 20.0])

    flat = Sensor(position=pos)
    tilted = Sensor(position=pos, orientation=tilt_rotation(np.pi / 2, 0.0))

    flat_reading = flat.measure(magnet)
    tilt_reading = tilted.measure(magnet)
    print(f"flat reading:   {flat_reading}")
    print(f"tilted reading: {tilt_reading}")

    assert abs(flat_reading[0]) < 1e-9 and abs(flat_reading[1]) < 1e-9, "flat sensor should read pure Bz on-axis"
    assert abs(tilt_reading[2]) < 1e-9, "tilted sensor's local-z channel should read ~0 (rotated onto global x)"
    assert abs(np.linalg.norm(tilt_reading) - np.linalg.norm(flat_reading)) < 1e-9, (
        "tilting only re-expresses the field in a different frame -- magnitude must be preserved"
    )


def check_jacobian_differs_for_tilted_vs_untilted_at_same_position() -> None:
    """Same sensor position, identity vs a real tilt -> fisher.position_jacobian
    should differ (not be numerically identical), confirming the pipeline is
    now actually orientation-sensitive."""
    pos = np.array([2.0, -1.0, 20.0])
    flat_sensors = [Sensor(position=np.array([0.0, 0.0, 0.0]))]
    tilt_sensors = [Sensor(position=np.array([0.0, 0.0, 0.0]), orientation=tilt_rotation(np.pi / 2, 0.0))]

    jac_flat = fisher.position_jacobian(pos, flat_sensors)
    jac_tilt = fisher.position_jacobian(pos, tilt_sensors)
    max_diff = float(np.max(np.abs(jac_flat - jac_tilt)))
    print(f"jacobian max abs diff (tilted vs untilted): {max_diff:.4e}")
    assert max_diff > 1e-6, "tilting a sensor must change its Jacobian -- pipeline must be orientation-sensitive"


def check_monte_carlo_converges_with_tilted_grid() -> None:
    """End-to-end: a warm-start MC run on a genuinely tilted grid should
    converge and track its own CRB, exercising the previously
    orientation-blind simulate_measurement/LeastSquaresSolver.solve path."""
    sensors = sensor_grid(
        num_across=5, num_below=5, sep_horizontal=12.0, sep_vertical=12.0, polar=0.3, azimuth=0.7
    )
    true_position = np.array([3.0, -2.0, 19.0])
    result = run_monte_carlo(true_position, sensors, n_trials=100, rng=np.random.default_rng(0))

    print(f"tilted-grid MC: converged {result.converged.sum()}/{len(result.converged)}, "
          f"empirical RMSE={result.rmse_mm:.4f} mm, CRB RMSE={result.crb_rmse_mm:.4f} mm")
    assert result.converged.sum() >= 95, "almost all trials should converge for a well-conditioned tilted grid"
    assert result.rmse_mm < 3 * result.crb_rmse_mm, "empirical RMSE should track the CRB, not blow up"


if __name__ == "__main__":
    check_identity_orientation_regression()
    check_tilted_sensor_reads_zero_on_perpendicular_axis()
    check_jacobian_differs_for_tilted_vs_untilted_at_same_position()
    check_monte_carlo_converges_with_tilted_grid()
    print("all sanity checks passed")
