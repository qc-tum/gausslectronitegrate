"""
Evaluate the nuclear overlap integrals for Gausslet orbitals.
"""

import numpy as np
from gaussian_integrals import gaussian_nuclear_integral_3d


def _nuclear_gausslet_factors(gauss_coeffs, coeff_indices, shifts):
    """
    Pre-compute products of Gausslet coefficients and Gaussian factors
    for evaluating nuclear overlap integrals.
    """
    assert len(gauss_coeffs) == len(coeff_indices)
    factor_indices = list(range(2*min(coeff_indices), 2*max(coeff_indices) + 1))
    # dictionary indexed by a "shift"
    factors = {}
    for shift in shifts:
        assert isinstance(shift, int)
        fact = np.zeros(len(factor_indices))
        for a, coeff_a in zip(coeff_indices, gauss_coeffs):
            for b, coeff_b in zip(coeff_indices, gauss_coeffs):
                idx = factor_indices.index(a + b)
                fact[idx] += coeff_a * coeff_b * np.exp(-(0.5 * (3*shift + (a - b)))**2)
        factors[shift] = fact
    return factors, factor_indices


def nuclear_gausslet_integrals(gauss_coeffs, coeff_indices, centers_list,
                               nuclear_pos, tol: float = 1e-16):
    """
    Evaluate the nuclear overlap integrals for two Gausslet orbitals
    with coordinates at integer `centers`.
    """
    prefac = (np.sqrt(np.pi) / 3)**3
    # unique shifts
    shifts = list({
        centers[0][n] - centers[1][n]
            for centers in centers_list
            for n in range(3)})
    factors, factor_indices = _nuclear_gausslet_factors(gauss_coeffs, coeff_indices, shifts)
    overlaps = {}
    for centers in centers_list:
        assert len(centers) == 2
        assert len(centers[0]) == 3 and len(centers[1]) == 3
        factor = [factors[centers[0][n] - centers[1][n]] for n in range(3)]
        # filter out very small entries
        idx = [np.where(np.abs(factor[n]) > tol)[0] for n in range(3)]
        factor = [factor[n][idx[n]] for n in range(3)]
        indices = [[factor_indices[i] for i in idx[n]] for n in range(3)]
        overlaps[centers] = prefac * sum(
            d0 * d1 * d2 *
            gaussian_nuclear_integral_3d(1. / 3, np.linalg.norm(
                0.5 * np.array([
                    (centers[0][0] + centers[1][0]) + i0 / 3.,
                    (centers[0][1] + centers[1][1]) + i1 / 3.,
                    (centers[0][2] + centers[1][2]) + i2 / 3.])
                - nuclear_pos))
            for i2, d2 in zip(indices[2], factor[2])
            for i1, d1 in zip(indices[1], factor[1])
            for i0, d0 in zip(indices[0], factor[0]))
    return overlaps
