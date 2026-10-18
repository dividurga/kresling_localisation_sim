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
    """central difference jacobian, (N,3,3). position in mm, step_size mm.
    magnet_kwargs goes to Kresling so other targets work (cube_robot etc)"""
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
    """position fisher info (3,3), delegates to noise.fisher_contribution since the formula depends on the noise"""
    return noise.fisher_contribution(jacobian)


def crlb(fisher_info_matrix: np.ndarray) -> np.ndarray:
    """cramer rao bound, stdev per axis in mm, sqrt of diag of inv(fisher). returns (3,)"""
    covariance = np.linalg.inv(fisher_info_matrix)
    variances = np.diag(covariance)
    return np.sqrt(variances)

    