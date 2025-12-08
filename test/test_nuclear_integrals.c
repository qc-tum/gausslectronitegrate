#include "nuclear_integrals.h"
#include "aligned_memory.h"
#include "hdf5_util.h"
#include "util.h"


char* test_nuclear_integrals()
{
	hid_t file = H5Fopen("../test/data/test_nuclear_integrals.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_nuclear_integrals failed";
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
			{ .istart =  2, .num = 2 },
			{ .istart = -1, .num = 3 },
		}
	};

	int num_nuclei;
	if (read_hdf5_attribute(file, "num_nuclei", H5T_NATIVE_INT, &num_nuclei) < 0) {
		return "reading number of nuclei from disk failed";
	}
	struct atomic_nucleus* nuclei = aligned_malloc(num_nuclei * sizeof(nuclei[0]));
	for (int i = 0; i < num_nuclei; ++i)
	{
		char varname[1024];
		sprintf(varname, "nuclear_pos_%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_DOUBLE, nuclei[i].pos) < 0) {
			return "reading nuclear position from disk failed";
		}
		sprintf(varname, "nuclear_charge_%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, &nuclei[i].charge) < 0) {
			return "reading nuclear charge from disk failed";
		}
	}

	double tol;
	if (read_hdf5_attribute(file, "tol", H5T_NATIVE_DOUBLE, &tol) < 0) {
		return "reading tolerance from disk failed";
	}

	struct nuclear_gausslet_integrals ngi;
	compute_nuclear_gausslet_integrals(&gdata, &grid, nuclei, num_nuclei, tol, &ngi);

	// reference data
	const glong num_points = cartesian_grid_3d_num_points(&grid);
	double* integral_values_ref = aligned_malloc(num_points * num_points * sizeof(integral_values_ref[0]));
	if (read_hdf5_dataset(file, "ngi", H5T_NATIVE_DOUBLE, integral_values_ref) < 0) {
		return "reading nuclear integral values from disk failed";
	}

	// compare
	if (uniform_distance(num_points * num_points, ngi.integral_values, integral_values_ref) > 1e-13) {
		return "nuclear integral values do not match reference";
	}

	aligned_free(integral_values_ref);
	delete_nuclear_gausslet_integrals(&ngi);
	aligned_free(nuclei);
	aligned_free(gdata.coefficients);

	H5Fclose(file);

	return 0;
}
