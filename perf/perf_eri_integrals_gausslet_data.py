import h5py
from gausslectronic import get_gausslet_g4_coeffs


def perf_eri_integrals_gausslet_data():

    # Gausslet G4 coefficients
    b4_list, _ = get_gausslet_g4_coeffs()

    with h5py.File("perf_eri_integrals_gausslet_data.hdf5", "w") as file:
        file.attrs["gausslet_coeffs"] = b4_list


if __name__ == "__main__":
    perf_eri_integrals_gausslet_data()
