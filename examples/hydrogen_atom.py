"""
Ground state energy and wavefunction of the hydrogen atom based on Gausslet orbital discretization.
"""

import numpy as np
import gausslectronitegrate as gli
import sys
sys.path.append("../gausslets/")
from gausslets import evaluate_gausslet_3d


def hydrogen_atom_groundstate_gausslets(gausslet_coeffs, grid_length, scaling_list, tol_nuc=1e-5):
    """
    Compute the ground state energies and wavefunctions based on a Gausslet
    orbital discretization on the specified grid and scaling factors.

    Args:
        gausslet_coeffs: Gausslet coefficients
        grid_length:     size length of the grid (odd positive integer); the overall grid
                         has dimension grid_length x grid_length x grid_length and
                         is centered at the origin
        scaling_list:    list of grid scaling factors
        tol_nuc:         truncation tolerance for the nuclear overlap integrals
                         (values close to zero imply higher accuracy but longer computation time)

    Returns:
        tuple: tuple containing
          - en0_list:  list of ground state energies for each scaling factor
          - psi0_list: list of corresponding eigenstates

    """
    assert isinstance(grid_length, int)
    assert grid_length % 2 == 1
    grid = [
        (-(grid_length - 1) // 2, grid_length),  # x
        (-(grid_length - 1) // 2, grid_length),  # y
        (-(grid_length - 1) // 2, grid_length),  # z
    ]

    # kinetic overlap integrals
    kgi = gli.compute_kinetic_gausslet_integrals(gausslet_coeffs, grid)
    # nuclear overlap integrals (for a nucleus at the origin with charge 1)
    ngi  = gli.compute_nuclear_gausslet_integrals(gausslet_coeffs, grid, [(0, 0, 0)], [1], tol_nuc)

    en0_list  = []
    psi0_list = []
    for scaling in scaling_list:
        # one-body Hamiltonian: kinetic energy and nuclear attraction
        h = scaling**2 * kgi - scaling * ngi
        # diagonalize 'h'
        en, psi = np.linalg.eigh(h)
        en0_list.append(en[0])
        psi0_list.append(psi[:, 0])

    return en0_list, psi0_list


def evaluate_gausslet_wavefunction(gausslet_coeffs, gausslet_indices, grid, scaling: float, psi, r):
    """
    Evaluate a linear combination of Gausslet orbitals on a cubic grid
    with coefficients `psi` at the spatial point `r`.
    """

    g0_range = range(grid[0][0], grid[0][0] + grid[0][1])
    g1_range = range(grid[1][0], grid[1][0] + grid[1][1])
    g2_range = range(grid[2][0], grid[2][0] + grid[2][1])

    prefactor = scaling**(3/2)

    val = 0
    for i0 in g0_range:
        for i1 in g1_range:
            for i2 in g2_range:
                idx = gli.grid_point_to_linear_index(grid, [i0, i1, i2])
                val += psi[idx] * prefactor \
                    * evaluate_gausslet_3d(gausslet_coeffs, gausslet_indices,
                                           [scaling*r[0] - i0,
                                            scaling*r[1] - i1,
                                            scaling*r[2] - i2])
    return val
