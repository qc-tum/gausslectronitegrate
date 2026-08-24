"""
Evaluate the electron repulsion integrals for Gausslet orbitals.
"""

import numpy as np
from gaussian_integrals import gaussian_coulomb_integral_3d


def _eri_gausslet_factors(gauss_coeffs, coeff_indices, shifts):
    """
    Pre-compute products of Gausslet coefficients and Gaussian factors
    for evaluating electron repulsion integrals.
    """
    assert len(gauss_coeffs) == len(coeff_indices)
    factor_indices = list(range(2*min(coeff_indices), 2*max(coeff_indices) + 1))
    # dictionary indexed by a "shift"
    factors = {}
    for shift in shifts:
        fact = np.zeros(len(factor_indices))
        for a, coeff_a in zip(coeff_indices, gauss_coeffs):
            for b, coeff_b in zip(coeff_indices, gauss_coeffs):
                idx = factor_indices.index(a + b)
                fact[idx] += coeff_a * coeff_b * np.exp(-(0.5 * (3*shift + (a - b)))**2)
        factors[shift] = fact
    return factors, factor_indices


def _eri_gausslet_products(gauss_coeffs, coeff_indices, pair_shifts):
    """
    Pre-compute products of Gausslet factors for evaluating electron repulsion integrals.
    """
    # unique shifts
    shifts = list({s for pair in pair_shifts for s in pair})
    factors, factor_indices = _eri_gausslet_factors(gauss_coeffs, coeff_indices, shifts)
    product_indices = list(range(min(factor_indices) - max(factor_indices),
                                 max(factor_indices) - min(factor_indices) + 1))
    # dictionary indexed by shift pair tuples
    products = {}
    for pair in pair_shifts:
        assert len(pair) == 2
        # convolution
        product = np.zeros(len(product_indices))
        for i, f in zip(factor_indices, factors[pair[0]]):
            for j, g in zip(factor_indices, factors[pair[1]]):
                idx = product_indices.index(i - j)
                product[idx] += f * g
        products[pair] = product
    return products, product_indices


def eri_gausslet_integrals(gauss_coeffs, coeff_indices, centers_list, tol: float = 1e-16):
    """
    Evaluate the electron repulsion integrals for Gausslet orbitals
    with coordinates at integer `centers`.
    """
    prefac = (np.pi / 9)**3
    # unique pair shifts
    pair_shifts = list({
        (centers[0][n] - centers[1][n],
         centers[2][n] - centers[3][n])
            for centers in centers_list
            for n in range(3)})
    products, product_indices = _eri_gausslet_products(gauss_coeffs, coeff_indices, pair_shifts)
    overlaps = {}
    for centers in centers_list:
        assert len(centers) == 4
        assert (len(centers[0]) == 3
            and len(centers[1]) == 3
            and len(centers[2]) == 3
            and len(centers[3]) == 3)
        product = [products[(centers[0][n] - centers[1][n], centers[2][n] - centers[3][n])]
                   for n in range(3)]
        # filter out very small entries
        idx = [np.where(np.abs(product[n]) > tol)[0] for n in range(3)]
        product = [product[n][idx[n]] for n in range(3)]
        indices = [[product_indices[i] for i in idx[n]] for n in range(3)]
        overlaps[centers] = prefac * sum(
            d0 * d1 * d2 *
            gaussian_coulomb_integral_3d(1. / 3, np.linalg.norm(0.5 * np.array([
                (centers[0][0] + centers[1][0]) - (centers[2][0] + centers[3][0]) + i0 / 3.,
                (centers[0][1] + centers[1][1]) - (centers[2][1] + centers[3][1]) + i1 / 3.,
                (centers[0][2] + centers[1][2]) - (centers[2][2] + centers[3][2]) + i2 / 3.])))
            for i2, d2 in zip(indices[2], product[2])
            for i1, d1 in zip(indices[1], product[1])
            for i0, d0 in zip(indices[0], product[0]))
    return overlaps
