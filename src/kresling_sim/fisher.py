"""Fisher information matrix for Kresling magnet localization.
Owns the Jcobian and Hessian calculations"""


from __future__ import annotations
import numpy as np

from kresling_sim import forward
from kresling_sim.kresling import Kresling
from kresling_sim.noise import NoiseModel

def position_jacobian(
    position: np.ndarray, points: np.ndarray, step_size: float = 1e-3, magnet_kwargs: dict | None = None
) -> np.ndarray:
    """compute the jacobian using finite differences
    Args:
        position (np.ndarray)(3): The position of the magnet (mm).
        points (np.ndarray)(N,3): The points in space (mm) where the field is calculated.
        step_size (float): The step size for finite differences (mm).
        magnet_kwargs (dict | None): passed through to Kresling(...) --
            diameter/height/remanence, so the CRB can be computed for a
            different magnet than config.py's default (e.g. cube_robot_config).
    Returns:
        np.ndarray: The jacobian matrix (N, 3).
    """
    magnet_kwargs = magnet_kwargs or {}

    magnet_plus_dx = Kresling(position=position + np.array([step_size, 0, 0]), **magnet_kwargs)
    magnet_minus_dx = Kresling(position=position - np.array([step_size, 0, 0]), **magnet_kwargs)
    field_at_plus_dx = forward.stacked_forward_prediction(magnet_plus_dx, points)
    field_at_minus_dx = forward.stacked_forward_prediction(magnet_minus_dx, points)
    dB_dx = (field_at_plus_dx - field_at_minus_dx) / (2 * step_size)

    magnet_plus_dy = Kresling(position=position + np.array([0, step_size, 0]), **magnet_kwargs)
    magnet_minus_dy = Kresling(position=position - np.array([0, step_size, 0]), **magnet_kwargs)
    field_at_plus_dy = forward.stacked_forward_prediction(magnet_plus_dy, points)
    field_at_minus_dy = forward.stacked_forward_prediction(magnet_minus_dy, points)
    dB_dy = (field_at_plus_dy - field_at_minus_dy) / (2 * step_size)

    magnet_plus_dz = Kresling(position=position + np.array([0, 0, step_size]), **magnet_kwargs)
    magnet_minus_dz = Kresling(position=position - np.array([0, 0, step_size]), **magnet_kwargs)
    field_at_plus_dz = forward.stacked_forward_prediction(magnet_plus_dz, points)
    field_at_minus_dz = forward.stacked_forward_prediction(magnet_minus_dz, points)
    dB_dz = (field_at_plus_dz - field_at_minus_dz) / (2 * step_size)

    jacobian = np.stack((dB_dx, dB_dy, dB_dz), axis=-1)  # shape (N, 3, 3)

    return jacobian



def fisher_information_matrix(jacobian: np.ndarray, noise: NoiseModel) -> np.ndarray:
    """Position-only Fisher information matrix. Fully agnostic to which
    NoiseModel this is -- just delegates to `noise.fisher_contribution`,
    since the correct formula depends on the noise distribution and isn't
    generic (see noise.py's module docstring). This used to be two
    functions (a Gaussian-specific one plus an unfinished stub with a
    different signature); consolidated into one now that the noise model
    itself owns the math.

    Args:
        jacobian (np.ndarray) (N,3,3): per-sensor Jacobian
            [sensor, field_component, pose_param], as returned by
            position_jacobian.
        noise (NoiseModel): the sensor noise model.

    Returns:
        np.ndarray (3,3): Fisher information matrix for the 3 position
            parameters (x, y, z).
    """
    return noise.fisher_contribution(jacobian)


def crlb(fisher_info_matrix: np.ndarray) -> np.ndarray:
    """Cramer-Rao lower bound: best-possible standard deviation for each
    position parameter (x, y, z), in mm -- the square root of the diagonal
    of the inverted Fisher information matrix. This is a floor, so if an estimator 
    meets this to a good degree, we can be haPPYYYYYY

    Args:
        fisher_info_matrix (np.ndarray) (3,3): Fisher information matrix for
            the 3 position parameters (x, y, z), as returned by
            fisher_information_matrix.

    Returns:
        np.ndarray (3,): best-possible position standard deviation, mm,
            one entry per parameter (x, y, z).
    """
    covariance = np.linalg.inv(fisher_info_matrix)
    variances = np.diag(covariance)
    return np.sqrt(variances)

    