"""
Ground state energy and wavefunction of the hydrogen molecule
based on Gausslet orbital discretization,
using the integral diagonal approximation for the electron repulsion integrals.
"""

import numpy as np
import scipy.sparse.linalg as spla
import gausslectronitegrate as gli


def apply_two_particle_hamiltonian(kinetic, nuclear, erida, psi):
    """
    Apply a molecular Hamiltonian to a two-particle state
    represented in first quantization convention.

    The spin state is assumed to be a spin singlet or triplet which has been factored out.

    Args:
        kinetic: kinetic overlap integrals
        nuclear: nuclear overlap integrals
        erida:   electron repulsion integrals using the integral diagonal approximation
        psi:     spatial wavefunction psi(r1, r2) stored as a vector

    Returns:
        np.array: wavefunction after applying the Hamiltonian
    """
    # one-body integrals: kinetic energy and nuclear attraction
    h = kinetic - nuclear
    n = h.shape[0]
    psi = np.reshape(psi, (n, n))
    # one-body terms, and electron repulsion via pointwise multiplication with 'erida' matrix
    return np.reshape((h @ psi + psi @ h.T) + (erida * psi), n**2)


def hydrogen_molecule_groundstate_gausslets_erida(
        nuclear_dist: float,
        gausslet_coeffs, grid_length: int, grid_scaling: float,
        tol=1e-5, rng=None):
    """
    Compute the ground state energy and wavefunction of the hydrogen molecule
    based on a Gausslet orbital discretization on the specified grid,
    using the integral diagonal approximation for the electron repulsion integrals.

    Args:
        nuclear_dist:    nuclear distance (in atomic units)
        gausslet_coeffs: Gausslet coefficients
        grid_length:     length of the grid along one coordinate axis (odd positive integer);
                         the overall grid has dimension grid_length x grid_length x grid_length
                         and is centered at the origin
        grid_scaling:    grid scaling factor
        tol:             truncation tolerance for the nuclear and electron repulsion integrals
                         (values close to zero imply higher accuracy but longer computation time)
        rng:             random number generator for the sparse eigenvalue solver

    Returns:
        tuple: tuple containing
          - en0:  ground state energy, including nuclear-nuclear repulsion
          - psi0: corresponding eigenstate in first quantization convention
    """

    assert isinstance(grid_length, int)
    assert grid_length % 2 == 1
    grid = [
        (-(grid_length - 1) // 2, grid_length),  # x
        (-(grid_length - 1) // 2, grid_length),  # y
        (-(grid_length - 1) // 2, grid_length),  # z
    ]

    # kinetic overlap integrals
    kgi = grid_scaling**2 * gli.compute_kinetic_gausslet_integrals(gausslet_coeffs, grid)

    # nuclear overlap integrals (for two nuclei with distance 'nuclear_dist' and charge 1)
    ngi = grid_scaling * gli.compute_nuclear_gausslet_integrals(gausslet_coeffs, grid,
                            [(-grid_scaling * 0.5 * nuclear_dist, 0, 0),
                             ( grid_scaling * 0.5 * nuclear_dist, 0, 0)], [1, 1], tol)

    # electron repulsion integrals using the integral diagonal approximation (ERIDA),
    # taking translational, octahedral and permutational symmetries into account
    sparse_erida = gli.compute_sparse_erida_gausslet_integrals(gausslet_coeffs, grid, tol)
    # dense matrix representation for all grid points, including scaling factor
    erida = grid_scaling * sparse_erida.fill_matrix(grid)

    op = spla.LinearOperator(
        shape  = 2*(gli.grid_num_points(grid)**2,),
        matvec = lambda psi: apply_two_particle_hamiltonian(kgi, ngi, erida, psi),
        dtype  = float)

    en_list, psi_list = spla.eigsh(op, k=8, which="SA", rng=rng)
    # add nuclear repulsion energy
    en0 = en_list[0] + 1 / nuclear_dist

    return en0, psi_list[:, 0]
