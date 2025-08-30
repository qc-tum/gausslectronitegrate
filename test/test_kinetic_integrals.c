#include "kinetic_integrals.h"
#include "aligned_memory.h"
#include "hdf5_util.h"
#include "util.h"


char* test_kinetic_integrals()
{
	hid_t file = H5Fopen("../test/data/test_kinetic_integrals.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_kinetic_integrals failed";
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
			{ .istart =  0, .num = 2 },
			{ .istart =  1, .num = 3 },
			{ .istart = -1, .num = 3 },
		}
	};

	struct kinetic_gausslet_integrals kgi;
	compute_kinetic_gausslet_integrals(&gdata, &grid, &kgi);

	// reference data
	const long num_points = cartesian_grid_3d_num_points(&grid);
	double* integral_values_ref = aligned_malloc(num_points * num_points * sizeof(integral_values_ref[0]));
	if (read_hdf5_dataset(file, "kgi", H5T_NATIVE_DOUBLE, integral_values_ref) < 0) {
		return "reading kinetic integral values from disk failed";
	}

	// compare
	if (uniform_distance(num_points * num_points, kgi.integral_values, integral_values_ref) > 1e-13) {
		return "kinetic integral values do not match reference";
	}

	aligned_free(integral_values_ref);
	delete_kinetic_gausslet_integrals(&kgi);
	aligned_free(gdata.coefficients);

	H5Fclose(file);

	return 0;
}
