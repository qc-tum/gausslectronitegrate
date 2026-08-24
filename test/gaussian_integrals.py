"""
Evaluate Gaussian overlap integrals.
"""

import numpy as np
from scipy.special import erf


def gaussian_nuclear_integral_3d(w: float, dist: float):
    """
    Evaluate the nuclear overlap integral of a normalizes Gaussians
    `g(r) = exp(-||r / w||^2) / (sqrt(pi) * w)^3` with a nuclear point charge
    in three dimensions with relative distance `dist`.
    """
    if dist > 0:
        return erf(dist / w) / dist
    else:
        return 2 / (np.sqrt(np.pi) * w)


def gaussian_coulomb_integral_3d(w: float, dist: float):
    """
    Evaluate the Coulomb overlap integral of two normalizes Gaussians
    `g(r) = exp(-||r / w||^2) / (sqrt(pi) * w)^3` in three dimensions
    with relative distance `dist`.
    """
    if dist > 0:
        return erf(dist / (np.sqrt(2) * w)) / dist
    else:
        return np.sqrt(2 / np.pi) / w
