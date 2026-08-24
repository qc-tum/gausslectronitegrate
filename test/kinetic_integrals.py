"""
Evaluate the kinetic overlap integrals of Gausslets in one dimension.
"""

import numpy as np


def elementary_gaussian_derivative_integral(d):
    """
    Evaluate the overlap integral of the derivative functions of two elementary Gaussians
    with relative index shift `d`.
    """
    return 1.5 * np.sqrt(np.pi) * (1 - 0.5 * d**2) * np.exp(-0.25 * d**2)


def kinetic_gausslet_integral(gauss_coeffs, coeff_indices, d: int) -> float:
    """
    Evaluate the kinetic overlap integral of two Gausslet functions with relative index shift `d`.
    """
    return 0.5 * sum(a * b * elementary_gaussian_derivative_integral(3*d + i - j)
                     for a, i in zip(gauss_coeffs, coeff_indices)
                     for b, j in zip(gauss_coeffs, coeff_indices))
