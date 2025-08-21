#include "eri_integrals.h"
#include "aligned_memory.h"
#include "hdf5_util.h"
#include "util.h"


char* test_eri_integrals()
{
	hid_t file = H5Fopen("../test/data/test_eri_integrals.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_eri_integrals failed";
	}

	struct gausslet_data gdata;
	{
		hsize_t dim[1];
		if (get_hdf5_attribute_dims(file, "gausslet_coeffs", dim) < 0) {
			return "obtaining dimensions of Gausslet coefficients failed";
		}
		gdata.indices.num = dim[0];
		if (gdata.indices.num % 2 != 1) {
			return "expecting an odd number of Gausslet coefficients";
		}
		gdata.indices.istart = -(gdata.indices.num - 1) / 2;

		gdata.coefficients = aligned_malloc(gdata.indices.num * sizeof(gdata.coefficients[0]));
		if (read_hdf5_attribute(file, "gausslet_coeffs", H5T_NATIVE_DOUBLE, gdata.coefficients) < 0) {
			return "reading Gausslet coefficients from disk failed";
		}
	}

	const struct cartesian_grid_3d grid = {
		.coord_range = {
			{ .istart = -1, .num = 2 },
			{ .istart =  0, .num = 2 },
			{ .istart = -1, .num = 3 },
		}
	};

	double tol;
	if (read_hdf5_attribute(file, "tol", H5T_NATIVE_DOUBLE, &tol) < 0) {
		return "reading tolerance from disk failed";
	}

	struct eri_gausslet_integrals eri;
	compute_eri_gausslet_integrals(&gdata, &grid, tol, &eri);

	const long num_points = cartesian_grid_3d_num_points(&grid);
	const long num_entries = num_points * num_points * num_points * num_points;

	// reconstruct full tensor
	double* full_tensor = aligned_malloc(num_entries * sizeof(full_tensor[0]));
	reconstruct_full_eri_tensor(&eri, full_tensor);

	// reference data
	double* full_tensor_ref = aligned_malloc(num_entries * sizeof(full_tensor_ref[0]));
	if (read_hdf5_dataset(file, "eri", H5T_NATIVE_DOUBLE, full_tensor_ref) < 0) {
		return "reading electron repulsion integral values from disk failed";
	}

	// compare
	if (uniform_distance(num_entries, full_tensor, full_tensor_ref) > 1e-13) {
		return "electron repulsion integral values do not match reference";
	}

	aligned_free(full_tensor);
	aligned_free(full_tensor_ref);
	delete_eri_gausslet_integrals(&eri);
	aligned_free(gdata.coefficients);

	H5Fclose(file);

	return 0;
}
