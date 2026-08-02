"""Fisher information matrix for Kresling magnet localization.
Owns the Jcobian and Hessian calculations"""


from __future__ import annotations
import numpy as np

from kresling_sim import forward
from kresling_sim.kresling import Kresling
from kresling_sim.noise import GaussianNoise

def position_jacobian(position: np.ndarray, points: np.ndarray, step_size: float = 1e-3) -> np.ndarray:
    """compute the jacobian using finite differences
    Args:
        position (np.ndarray)(3): The position of the Kresling magnet (mm).
        points (np.ndarray)(N,3): The points in space (mm) where the field is calculated.
        step_size (float): The step size for finite differences (mm).
    Returns:
        np.ndarray: The jacobian matrix (N, 3).
    """


    magnet_plus_dx = Kresling(position=position + np.array([step_size, 0, 0]))
    magnet_minus_dx = Kresling(position=position - np.array([step_size, 0, 0]))
    field_at_plus_dx = forward.stacked_forward_prediction(magnet_plus_dx, points)
    field_at_minus_dx = forward.stacked_forward_prediction(magnet_minus_dx, points)
    dB_dx = (field_at_plus_dx - field_at_minus_dx) / (2 * step_size)

    magnet_plus_dy = Kresling(position=position + np.array([0, step_size, 0]))
    magnet_minus_dy = Kresling(position=position - np.array([0, step_size, 0]))
    field_at_plus_dy = forward.stacked_forward_prediction(magnet_plus_dy, points)
    field_at_minus_dy = forward.stacked_forward_prediction(magnet_minus_dy, points)
    dB_dy = (field_at_plus_dy - field_at_minus_dy) / (2 * step_size)

    magnet_plus_dz = Kresling(position=position + np.array([0, 0, step_size]))
    magnet_minus_dz = Kresling(position=position - np.array([0, 0, step_size]))
    field_at_plus_dz = forward.stacked_forward_prediction(magnet_plus_dz, points)
    field_at_minus_dz = forward.stacked_forward_prediction(magnet_minus_dz, points)
    dB_dz = (field_at_plus_dz - field_at_minus_dz) / (2 * step_size)

    jacobian = np.stack((dB_dx, dB_dy, dB_dz), axis=-1)  # shape (N, 3, 3)

    return jacobian


def fisher_information_matrix(jacobian: np.ndarray, noise: GaussianNoise) -> np.ndarray:
    """Position-only Fisher information matrix: I = J^T Sigma^-1 J.

    Takes a `GaussianNoise` (see noise.py) rather than a bare sigma float,
    so the noise model (uniform, per-axis, or per-sensor-per-axis) isn't
    hardcoded here -- only Gaussian noise models expose the `precision()`
    this formula needs, since I = J^T Sigma^-1 J is specifically derived
    from a Gaussian log-likelihood (see noise.py's module docstring).

    Args:
        jacobian (np.ndarray) (N,3,3): per-sensor Jacobian
            [sensor, field_component, pose_param], as returned by
            position_jacobian.
        noise (GaussianNoise): the sensor noise model.

    Returns:
        np.ndarray (3,3): Fisher information matrix for the 3 position
            parameters (x, y, z).
    """
    n_sensors = jacobian.shape[0]
    jacobian_flat = jacobian.reshape(-1, jacobian.shape[-1])  # (3N, 3)
    precision = noise.precision(n_sensors)  # (3N, 3N)
    return jacobian_flat.T @ precision @ jacobian_flat


def crlb(fisher_info_matrix: np.ndarray) -> np.ndarray:
    """Cramer-Rao lower bound on the covariance of any unbiased estimator
    of the Kresling magnet's position, given the Fisher information matrix.

    Args:
        fisher_info_matrix (np.ndarray) (3,3): Fisher information matrix for
            the 3 position parameters (x, y, z).
            
            okay so here per our derivation, this is J^T Sigma^-1 J, which is the Fisher information matrix for the position parameters."""

    # sigma in this scenario is 

    