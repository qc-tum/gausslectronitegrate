import numpy as np
import h5py
import sys
sys.path.append("../gausslets/")
from gausslets import get_gausslet_g10_coeffs
from eri_integrals import eri_gausslet_integrals


def _tuple_diff(ta, tb) -> tuple:
    """
    Element-wise difference of two tuples.
    """
    return tuple(a - b for a, b in zip(ta, tb))


def compute_eri_integrals(gauss_coeffs, coeff_indices, centers_list, tol: float):
    """
    Compute the electron repulsion integrals (ERIs) for Gausslet orbitals.
    """
    # use symmetries to avoid redundant calculations
    center_quadruples = set()
    for i, ca in enumerate(centers_list):
        for j, cb in enumerate(centers_list):
            if i < j:
                continue
            for k, cc in enumerate(centers_list):
                for l, cd in enumerate(centers_list):
                    if k < l:
                        continue
                    if (i, j) < (k, l):
                        continue
                    center_quadruples.add((_tuple_diff(ca, cd),
                                           _tuple_diff(cb, cd),
                                           _tuple_diff(cc, cd),
                                           _tuple_diff(cd, cd)))
    center_quadruples = sorted(list(center_quadruples))

    eri_symm = eri_gausslet_integrals(gauss_coeffs, coeff_indices, center_quadruples, tol)

    # reconstruct the full electron repulsion integral tensor given the symmetry-reduced integrals
    eri = np.zeros(4 * (len(centers_list),))
    for i in range(len(centers_list)):
        for j in range(len(centers_list)):
            for k in range(len(centers_list)):
                for l in range(len(centers_list)):
                    ip = i
                    jp = j
                    if ip < jp:
                        ip, jp = jp, ip
                    kp = k
                    lp = l
                    if kp < lp:
                        kp, lp = lp, kp
                    if (ip, jp) < (kp, lp):
                        # swap (ip, jp) <-> (kp, lp)
                        (ip, jp), (kp, lp) = (kp, lp), (ip, jp)
                    ca = centers_list[ip]
                    cb = centers_list[jp]
                    cc = centers_list[kp]
                    cd = centers_list[lp]
                    eri[i, j, k, l] = eri_symm[(
                        _tuple_diff(ca, cd),
                        _tuple_diff(cb, cd),
                        _tuple_diff(cc, cd),
                        _tuple_diff(cd, cd))]
    return eri


def eri_integrals_data():

    # Gausslet coefficients
    b_list, j_list = get_gausslet_g10_coeffs()

    # grid points
    centers_list = [(x, y, z) for x in range(-1, 1) for y in range(0, 2) for z in range(-1, 2)]

    # set truncation tolerance to an artificially high value
    tol = 0.001

    eri = compute_eri_integrals(b_list, j_list, centers_list, tol)

    with h5py.File("data/test_eri_integrals.hdf5", "w") as file:
        file.attrs["gausslet_coeffs"] = b_list
        file.attrs["tol"] = tol
        file["eri"] = eri


if __name__ == "__main__":
    eri_integrals_data()
