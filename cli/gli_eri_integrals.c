#include <stdlib.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "eri_integrals.h"
#include "aligned_memory.h"
#include "hdf5_util.h"
#include "timing.h"


int main(int argc, char* argv[])
{
	const char* syntax = "gli_eri_integrals <Gausslet coefficient data filename (HDF5 format)> <grid length> <truncation tolerance> <output filename (HDF5 format)>";

	// parse command line arguments
	if (argc != 5)
	{
		fprintf(stderr, "expecting four input parameters; syntax: %s\n", syntax);
		return -1;
	}
	const char* gausslet_filename = argv[1];
	const glong grid_length = strtol(argv[2], NULL, 10);
	if (grid_length <= 0 || (grid_length % 2 == 0))
	{
		fprintf(stderr, "'grid_length' must be a positive odd integer; syntax: %s\n", syntax);
		return -1;
	}
	const double tol = atof(argv[3]);
	if (tol < 0)
	{
		fprintf(stderr, "'tol' cannot be negative; syntax: %s\n", syntax);
		return -1;
	}
	const char* out_filename = argv[4];

	// read Gausslet data from disk
	struct gausslet_data gdata;
	{
		hid_t file = H5Fopen(gausslet_filename, H5F_ACC_RDONLY, H5P_DEFAULT);
		if (file < 0) {
			fprintf(stderr, "'H5Fopen' of Gausslet data file '%s' failed\n", gausslet_filename);
			return -1;
		}

		hsize_t dim[1];
		if (get_hdf5_attribute_dims(file, "gausslet_coeffs", dim) < 0) {
			fprintf(stderr, "obtaining dimensions of Gausslet coefficients failed\n");
			return -1;
		}
		gdata.indices.num = dim[0];
		if (gdata.indices.num % 2 != 1) {
			fprintf(stderr, "expecting an odd number of Gausslet coefficients\n");
			return -1;
		}
		gdata.indices.istart = -(gdata.indices.num - 1) / 2;

		gdata.coefficients = aligned_malloc(gdata.indices.num * sizeof(gdata.coefficients[0]));
		if (read_hdf5_attribute(file, "gausslet_coeffs", H5T_NATIVE_DOUBLE, gdata.coefficients) < 0) {
			fprintf(stderr, "reading Gausslet coefficients from disk failed\n");
			return -1;
		}

		H5Fclose(file);
	}

	// require a rotationally symmetric grid
	const struct cartesian_grid_3d grid = {
		.coord_range = {
			{ .istart = -(grid_length - 1) / 2, .num = grid_length },
			{ .istart = -(grid_length - 1) / 2, .num = grid_length },
			{ .istart = -(grid_length - 1) / 2, .num = grid_length },
		}
	};
	const glong num_points = cartesian_grid_3d_num_points(&grid);

	#ifdef _OPENMP
	printf("maximum number of OpenMP threads: %d\n", omp_get_max_threads());
	#else
	printf("OpenMP not available\n");
	#endif

	// get the tick resolution
	const double ticks_per_sec = (double)get_tick_resolution();

	printf("Enumerating symmetry-reduced electron repulsion integral (ERI) indices for a %li x %li x %li grid... ", grid.coord_range[0].num, grid.coord_range[1].num, grid.coord_range[2].num);
	const uint64_t tick_start_indices = get_time_ticks();
	struct sparse_eri_indices eri_indices;
	enumerate_symmetry_reduced_eri_indices(&grid, &eri_indices);
	const uint64_t tick_end_indices = get_time_ticks();
	printf("Done.\n");
	printf("number of indices (after translational, octahedral and permutational symmetry reductions): %li, reduction factor: %g\n",
		eri_indices.num, (double)eri_indices.num / (num_points * num_points * num_points * num_points));
	printf("wall clock time: %g seconds\n", (tick_end_indices - tick_start_indices) / ticks_per_sec);

	printf("Evaluating %li electron repulsion integrals (ERIs) with truncation tolerance %g... ", eri_indices.num, tol);
	const uint64_t tick_start_integrals = get_time_ticks();
	struct sparse_eri_gausslet_integrals eri_integrals;
	compute_sparse_eri_gausslet_integrals(&gdata, &grid, eri_indices.four_indices, eri_indices.num, tol, &eri_integrals);
	const uint64_t tick_end_integrals = get_time_ticks();
	delete_sparse_eri_indices(&eri_indices);
	printf("Done.\n");
	printf("wall clock time: %g seconds\n", (tick_end_integrals - tick_start_integrals) / ticks_per_sec);

	// save results to disk
	printf("Saving results to file \"%s\"... ", out_filename);
	{
		if (sizeof(long long) != 8) {
			printf("warning: 'long long' type is %li bytes on native platform, expecting 8 bytes; 64 bit integers might be incorrectly stored\n", sizeof(long long));
		}

		hid_t file = H5Fcreate(out_filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
		if (file < 0)
		{
			fprintf(stderr, "'H5Fcreate' failed for '%s'\n", out_filename);
			return -1;
		}

		// store grid length and tolerance as global attributes
		if (write_hdf5_scalar_attribute(file, "grid_length", H5T_STD_I64LE, H5T_NATIVE_LLONG, &grid_length) < 0)  // assuming that "long long" is 64 bit
		{
			fprintf(stderr, "writing 'grid_length' attribute to HDF5 file '%s' failed\n", out_filename);
			H5Fclose(file);
			return -1;
		}
		if (write_hdf5_scalar_attribute(file, "tol", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, &tol) < 0)
		{
			fprintf(stderr, "writing 'tol' attribute to HDF5 file '%s' failed\n", out_filename);
			H5Fclose(file);
			return -1;
		}

		// also save Gausslet coefficients, for completeness
		hsize_t dim_coeffs[1] = { gdata.indices.num };
		if (write_hdf5_dataset(file, "gausslet_coeffs", 1, dim_coeffs, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, gdata.coefficients) < 0)
		{
			fprintf(stderr, "writing Gausslet coefficients to HDF5 file '%s' failed\n", out_filename);
			H5Fclose(file);
			return -1;
		}

		hsize_t dim_integrals[1] = { eri_integrals.num_entries };

		if (write_hdf5_dataset(file, "integral_values", 1, dim_integrals, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, eri_integrals.integral_values) < 0)
		{
			fprintf(stderr, "writing integral values to HDF5 file '%s' failed\n", out_filename);
			H5Fclose(file);
			return -1;
		}

		if (write_hdf5_dataset(file, "four_indices", 1, dim_integrals, H5T_STD_I64LE, H5T_NATIVE_LLONG, eri_integrals.four_indices) < 0)  // assuming that "long long" is 64 bit
		{
			fprintf(stderr, "writing four-indices to HDF5 file '%s' failed\n", out_filename);
			H5Fclose(file);
			return -1;
		}

		if (H5Fclose(file) < 0)
		{
			fprintf(stderr, "'H5Fclose' failed for '%s'\n", out_filename);
			return -1;
		}
	}
	printf("Done.\n");

	delete_sparse_eri_gausslet_integrals(&eri_integrals);
	aligned_free(gdata.coefficients);

	return 0;
}
