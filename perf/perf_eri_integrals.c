#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "eri_integrals.h"
#include "aligned_memory.h"
#include "hdf5_util.h"
#include "timing.h"


int main()
{
	// read Gausslet data from disk
	struct gausslet_data gdata;
	{
		hid_t file = H5Fopen("../perf/perf_eri_integrals_gausslet_data.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
		if (file < 0) {
			fprintf(stderr, "'H5Fopen' failed\n");
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

	const struct cartesian_grid_3d grid = {
		.coord_range = {
			{ .istart = -1, .num = 3 },
			{ .istart = -1, .num = 3 },
			{ .istart = -2, .num = 5 },
		}
	};
	const long num_points = cartesian_grid_3d_num_points(&grid);

	const double tol = 1e-5;
	printf("truncation tolerance: %g\n", tol);

	#ifdef _OPENMP
	printf("maximum number of OpenMP threads: %d\n", omp_get_max_threads());
	#else
	printf("OpenMP not available\n");
	#endif

	// get the tick resolution
	const double ticks_per_sec = (double)get_tick_resolution();

	printf("Evaluating the electron repulsion integrals (ERIs) for Gausslet orbitals and %li grid points... ", num_points);
	const uint64_t tick_start = get_time_ticks();
	struct eri_gausslet_integrals eri;
	compute_eri_gausslet_integrals(&gdata, &grid, tol, &eri);
	const uint64_t tick_end = get_time_ticks();
	printf("Done.\n");

	const long num_points_trans = cartesian_grid_3d_num_points(&eri.grid_trans);
	printf("number of points of translational grid: %li\n", num_points_trans);

	// index of origin
	const long iorigin = cartesian_grid_3d_cartesian_to_linear_index(&eri.grid_trans,
		-eri.grid_trans.coord_range[0].istart,
		-eri.grid_trans.coord_range[1].istart,
		-eri.grid_trans.coord_range[2].istart);
	printf("ERI for all Gausslets at origin: %.17g\n", eri.integral_values[(iorigin * num_points_trans + iorigin) * num_points_trans + iorigin]);

	printf("wall clock time: %g seconds\n", (tick_end - tick_start) / ticks_per_sec);

	delete_eri_gausslet_integrals(&eri);
	aligned_free(gdata.coefficients);

	return 0;
}
