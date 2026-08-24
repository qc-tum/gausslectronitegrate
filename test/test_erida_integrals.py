import numpy as np
import h5py
import sys
sys.path.append("../gausslets/")
from gausslet_coefficients import get_gausslet_g10_coeffs
from erida_integrals import erida_gausslet_integrals


def _tuple_diff(ta, tb) -> tuple:
    """
    Element-wise difference of two tuples.
    """
    return tuple(a - b for a, b in zip(ta, tb))


def compute_erida_integrals(gauss_coeffs, coeff_indices, centers_list, tol: float):
    """
    Compute electron repulsion integrals
    using the integral diagonal approximation (ERIDA)
    for Gausslet orbitals.
    """
    # use symmetries to avoid redundant calculations
    center_tuples = set()
    for i, ca in enumerate(centers_list):
        for j, cb in enumerate(centers_list):
            if i < j:
                continue
            center_tuples.add((_tuple_diff(ca, cb),
                               _tuple_diff(cb, cb)))
    center_tuples = sorted(list(center_tuples))

    erida_symm = erida_gausslet_integrals(gauss_coeffs, coeff_indices, center_tuples, tol)

    # reconstruct the full electron repulsion integral tensor given the symmetry-reduced integrals
    erida = np.zeros(2 * (len(centers_list),))
    for i in range(len(centers_list)):
        for j in range(len(centers_list)):
            ip = i
            jp = j
            if ip < jp:
                ip, jp = jp, ip
            ca = centers_list[ip]
            cb = centers_list[jp]
            erida[i, j] = erida_symm[(
                _tuple_diff(ca, cb),
                _tuple_diff(cb, cb))]
    return erida


def erida_integrals_data():

    # Gausslet coefficients
    b_list, j_list = get_gausslet_g10_coeffs()

    # grid points
    centers_list = [(x, y, z) for x in range(0, 2) for y in range(-1, 2) for z in range(-1, 1)]

    # set truncation tolerance to an artificially high value
    tol = 0.001

    erida = compute_erida_integrals(b_list, j_list, centers_list, tol)

    with h5py.File("data/test_erida_integrals.hdf5", "w") as file:
        file.attrs["gausslet_coeffs"] = b_list
        file.attrs["tol"] = tol
        file["erida"] = erida


if __name__ == "__main__":
    erida_integrals_data()
