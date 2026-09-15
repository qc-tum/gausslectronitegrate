import h5py
import sys
sys.path.append("../gausslets/")
from gausslets import get_gausslet_g10_coeffs


def perf_eri_integrals_gausslet_data():

    # Gausslet G10 coefficients
    b_list, _ = get_gausslet_g10_coeffs()

    with h5py.File("perf_eri_integrals_gausslet_data.hdf5", "w") as file:
        file.attrs["gausslet_coeffs"] = b_list


if __name__ == "__main__":
    perf_eri_integrals_gausslet_data()
