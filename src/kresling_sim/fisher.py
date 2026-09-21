"""fisher information matrix for kresling magnet localization. owns the
jacobian/hessian calcs"""


from __future__ import annotations
import numpy as np

from kresling_sim import forward
from kresling_sim.kresling import Kresling
from kresling_sim.noise import NoiseModel
from kresling_sim.sensors import Sensor

def position_jacobian(
    position: np.ndarray, sensors: list[Sensor], step_size: float = 1e-3, magnet_kwargs: dict | None = None
) -> np.ndarray:
    """jacobian via finite differences. position (3,) mm, sensors: list of
    N Sensors (position + orientation), step_size mm. magnet_kwargs passed
    through to Kresling(...) so the CRB can be computed for a different
    magnet than config.py's default (e.g. cube_robot_config). returns
    (N,3,3)."""
    magnet_kwargs = magnet_kwargs or {}

    magnet_plus_dx = Kresling(position=position + np.array([step_size, 0, 0]), **magnet_kwargs)
    magnet_minus_dx = Kresling(position=position - np.array([step_size, 0, 0]), **magnet_kwargs)
    field_at_plus_dx = forward.stacked_forward_prediction(magnet_plus_dx, sensors)
    field_at_minus_dx = forward.stacked_forward_prediction(magnet_minus_dx, sensors)
    dB_dx = (field_at_plus_dx - field_at_minus_dx) / (2 * step_size)

    magnet_plus_dy = Kresling(position=position + np.array([0, step_size, 0]), **magnet_kwargs)
    magnet_minus_dy = Kresling(position=position - np.array([0, step_size, 0]), **magnet_kwargs)
    field_at_plus_dy = forward.stacked_forward_prediction(magnet_plus_dy, sensors)
    field_at_minus_dy = forward.stacked_forward_prediction(magnet_minus_dy, sensors)
    dB_dy = (field_at_plus_dy - field_at_minus_dy) / (2 * step_size)

    magnet_plus_dz = Kresling(position=position + np.array([0, 0, step_size]), **magnet_kwargs)
    magnet_minus_dz = Kresling(position=position - np.array([0, 0, step_size]), **magnet_kwargs)
    field_at_plus_dz = forward.stacked_forward_prediction(magnet_plus_dz, sensors)
    field_at_minus_dz = forward.stacked_forward_prediction(magnet_minus_dz, sensors)
    dB_dz = (field_at_plus_dz - field_at_minus_dz) / (2 * step_size)

    jacobian = np.stack((dB_dx, dB_dy, dB_dz), axis=-1)  # shape (N, 3, 3)

    return jacobian



def fisher_information_matrix(jacobian: np.ndarray, noise: NoiseModel) -> np.ndarray:
    """position-only fisher info matrix. agnostic to which NoiseModel this
    is, just delegates to noise.fisher_contribution since the formula
    depends on the noise distribution (see noise.py). jacobian: (N,3,3)
    per-sensor jacobian from position_jacobian. returns (3,3)."""
    return noise.fisher_contribution(jacobian)


def crlb(fisher_info_matrix: np.ndarray) -> np.ndarray:
    """cramer-rao lower bound: best-possible stdev per position param
    (x,y,z), mm -- sqrt of the diagonal of the inverted fisher info matrix.
    it's a floor, so an estimator meeting it is about as good as it gets.
    takes the (3,3) matrix from fisher_information_matrix, returns (3,)."""
    covariance = np.linalg.inv(fisher_info_matrix)
    variances = np.diag(covariance)
    return np.sqrt(variances)

    