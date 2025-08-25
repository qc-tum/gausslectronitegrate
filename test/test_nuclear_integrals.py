import numpy as np
import h5py
from gausslectronic import (
    get_gausslet_g10_coeffs,
    nuclear_gausslet_integrals)


def nuclear_integrals_data():

    # Gausslet coefficients
    b_list, j_list = get_gausslet_g10_coeffs()

    # grid points
    centers_list = [(x, y, z) for x in range(0, 2) for y in range(2, 4) for z in range(-1, 2)]
    center_pairs = [(ci, cj) for ci in centers_list for cj in centers_list]

    # nuclei
    nuclear_pos = [
        np.array([-0.7,  1.2,  0.3]),
        np.array([ 0.4,  1,   -1.5])]
    nuclear_charge = [2, 3]

    # set truncation tolerance to an artificially high value
    tol = 0.003

    ngi_list = [nuclear_gausslet_integrals(b_list, j_list, center_pairs, pos, tol=tol)
           for pos in nuclear_pos]
    ngi = np.zeros(2 * (len(centers_list),))
    for i, ci in enumerate(centers_list):
        for j, cj in enumerate(centers_list):
            ngi[i, j] = sum(z * ngi_list[k][(ci, cj)] for k, z in enumerate(nuclear_charge))

    with h5py.File("data/test_nuclear_integrals.hdf5", "w") as file:
        file.attrs["gausslet_coeffs"] = b_list
        file.attrs["num_nuclei"]      = len(nuclear_pos)
        for i in range(len(nuclear_pos)):
            file.attrs[f"nuclear_pos_{i}"]    = nuclear_pos[i]
            file.attrs[f"nuclear_charge_{i}"] = nuclear_charge[i]
        file.attrs["tol"] = tol
        file["ngi"] = ngi


if __name__ == "__main__":
    nuclear_integrals_data()
