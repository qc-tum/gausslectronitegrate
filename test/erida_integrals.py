"""
Evaluate the electron repulsion integrals
using the integral diagonal approximation (ERIDA)
for Gausslet orbitals.
"""

import numpy as np
from gaussian_integrals import gaussian_coulomb_integral_3d


def _erida_gausslet_factors(gauss_coeffs, coeff_indices):
    """
    Pre-compute products of Gausslet coefficients
    for evaluating electron repulsion integrals.
    """
    assert len(gauss_coeffs) == len(coeff_indices)
    max_diff = max(coeff_indices) - min(coeff_indices)
    factor_indices = list(range(-max_diff, max_diff + 1))
    factors = np.zeros(len(factor_indices))
    prefac = 2 * np.pi / 9
    for a, coeff_a in zip(coeff_indices, gauss_coeffs):
        for b, coeff_b in zip(coeff_indices, gauss_coeffs):
            idx = factor_indices.index(a - b)
            factors[idx] += prefac * coeff_a * coeff_b
    return factors, factor_indices


def erida_gausslet_integrals(gauss_coeffs, coeff_indices, centers_list, tol: float = 1e-16):
    """
    Evaluate the electron repulsion integrals using the integral diagonal approximation (ERIDA)
    for Gausslet orbitals with coordinates at integer `centers`.
    """
    factors, factor_indices = _erida_gausslet_factors(gauss_coeffs, coeff_indices)
    # filter out very small entries
    idx = np.where(np.abs(factors) > tol)[0]
    factors = factors[idx]
    factor_indices = [factor_indices[i] for i in idx]

    overlaps = {}
    for centers in centers_list:
        assert len(centers) == 2
        assert len(centers[0]) == 3 and len(centers[1]) == 3
        overlaps[centers] = sum(
            d0 * d1 * d2 *
            gaussian_coulomb_integral_3d(np.sqrt(2) / 3, np.linalg.norm(
                np.array([
                    (centers[0][0] - centers[1][0]) + i0 / 3.,
                    (centers[0][1] - centers[1][1]) + i1 / 3.,
                    (centers[0][2] - centers[1][2]) + i2 / 3.])))
            for i2, d2 in zip(factor_indices, factors)
            for i1, d1 in zip(factor_indices, factors)
            for i0, d0 in zip(factor_indices, factors))
    return overlaps
