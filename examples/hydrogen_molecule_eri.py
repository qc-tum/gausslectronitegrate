"""
Ground state energy and wavefunction of the hydrogen molecule
based on Gausslet orbital discretization.
"""

import functools
import numpy as np
import scipy.sparse.linalg as spla
import gausslectronitegrate as gli


def apply_two_particle_hamiltonian(tkin, vint, psi):
    """
    Apply a molecular Hamiltonian to a two-particle state
    represented in first quantization convention.

    The spin state is assumed to be a spin singlet or triplet which has been factored out.

    Args:
        tkin: coefficients of the one-body term: kinetic energy and nuclear attraction
        vint: electron repulsion integral tensor in physicists' convention, reshaped into a matrix
        psi:  spatial wavefunction psi(r1, r2) stored as a vector

    Returns:
        np.array: wavefunction after applying the Hamiltonian
    """
    n = tkin.shape[0]
    psi_mat = np.reshape(psi, (n, n))
    # one-body terms and electron-electron repulsion
    return np.reshape((tkin @ psi_mat + psi_mat @ tkin.T), n**2) + vint @ psi


def hydrogen_molecule_groundstate_gausslets_eri(
        nuclear_dist: float,
        gausslet_coeffs, grid_length: int, scaling_list,
        tol=1e-5, rng=None):
    """
    Compute the ground state energy and wavefunction of the hydrogen molecule
    based on a Gausslet orbital discretization on the specified grid.

    Args:
        nuclear_dist:    nuclear distance (in atomic units)
        gausslet_coeffs: Gausslet coefficients
        grid_length:     length of the grid along one coordinate axis (odd positive integer);
                         the overall grid has dimension grid_length x grid_length x grid_length
                         and is centered at the origin
        scaling_list:    list of grid scaling factors
        tol:             truncation tolerance for the nuclear and electron repulsion integrals
                         (values close to zero imply higher accuracy but longer computation time)
        rng:             random number generator for the sparse eigenvalue solver

    Returns:
        tuple: tuple containing
          - en0_list:  list of ground state energies for each scaling factor
          - psi0_list: list of corresponding eigenstates in first quantization convention
    """

    assert isinstance(grid_length, int)
    assert grid_length % 2 == 1
    grid = [
        (-(grid_length - 1) // 2, grid_length),  # x
        (-(grid_length - 1) // 2, grid_length),  # y
        (-(grid_length - 1) // 2, grid_length),  # z
    ]

    # electron repulsion integrals (ERIs) for Gausslet orbitals centered at the grid points,
    # taking translational, octahedral and permutational symmetries into account
    sparse_eri = gli.compute_sparse_eri_gausslet_integrals(gausslet_coeffs, grid, tol)
    # dense tensor representation for all grid points
    eri_tensor_dense = sparse_eri.fill_tensor(grid)

    num_points = gli.grid_num_points(grid)

    en0_list  = []
    psi0_list = []
    for scaling in scaling_list:

        # kinetic overlap integrals
        kgi = scaling**2 * gli.compute_kinetic_gausslet_integrals(gausslet_coeffs, grid)
        # nuclear overlap integrals (for two nuclei with distance 'nuclear_dist' and charge 1)
        ngi = scaling * gli.compute_nuclear_gausslet_integrals(gausslet_coeffs, grid,
                            [(-scaling * 0.5 * nuclear_dist, 0, 0),
                             ( scaling * 0.5 * nuclear_dist, 0, 0)], [1, 1], tol)
        # one-body Hamiltonian term: kinetic energy and nuclear attraction
        tkin = kgi - ngi
        # two-body Hamiltonian term: electron-electron repulsion;
        # transposition due to integral convention
        vint = scaling * eri_tensor_dense.transpose((0, 2, 1, 3)).reshape(2*(num_points**2,))

        op = spla.LinearOperator(
            shape  = 2*(num_points**2,),
            matvec = functools.partial(apply_two_particle_hamiltonian, tkin, vint),
            dtype  = float)

        en, psi = spla.eigsh(op, k=8, which="SA", rng=rng)
        # add nuclear repulsion energy
        en0 = en[0] + 1 / nuclear_dist

        en0_list.append(en0)
        psi0_list.append(psi[:, 0])

    return en0_list, psi0_list
