import numpy as np
import h5py
from gausslectronic import (
    get_gausslet_g10_coeffs,
    kinetic_gausslet_integral)


def delta(x, y):
    """
    Evaluate the 'delta' function.
    """
    return 1 if x == y else 0


def kinetic_integrals_data():

    # Gausslet coefficients
    b_list, j_list = get_gausslet_g10_coeffs()

    # grid points
    centers_list = [(x, y, z) for x in range(0, 2) for y in range(1, 4) for z in range(-1, 2)]

    integrals_1d = {d: kinetic_gausslet_integral(b_list, j_list, d) for d in range(5)}

    kgi = np.zeros(2 * (len(centers_list),))
    for i, ci in enumerate(centers_list):
        for j, cj in enumerate(centers_list):
            kgi[i, j] = (
                integrals_1d[abs(ci[0] - cj[0])] * delta(ci[1], cj[1]) * delta(ci[2], cj[2]) +
                delta(ci[0], cj[0]) * integrals_1d[abs(ci[1] - cj[1])] * delta(ci[2], cj[2]) +
                delta(ci[0], cj[0]) * delta(ci[1], cj[1]) * integrals_1d[abs(ci[2] - cj[2])])

    with h5py.File("data/test_kinetic_integrals.hdf5", "w") as file:
        file.attrs["gausslet_coeffs"] = b_list
        file["kgi"] = kgi


if __name__ == "__main__":
    kinetic_integrals_data()
