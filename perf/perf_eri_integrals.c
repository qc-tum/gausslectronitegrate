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

	// require a rotationally symmetric grid
	const struct cartesian_grid_3d grid = {
		.coord_range = {
			{ .istart = -2, .num = 5 },
			{ .istart = -2, .num = 5 },
			{ .istart = -2, .num = 5 },
		}
	};
	const glong num_points = cartesian_grid_3d_num_points(&grid);

	const double tol = 1e-5;
	printf("truncation tolerance: %g\n", tol);

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

	printf("Evaluating %li electron repulsion integrals (ERIs)... ", eri_indices.num);
	const uint64_t tick_start_integrals = get_time_ticks();
	struct sparse_eri_gausslet_integrals eri_integrals;
	compute_sparse_eri_gausslet_integrals(&gdata, &grid, eri_indices.four_indices, eri_indices.num, tol, &eri_integrals);
	const uint64_t tick_end_integrals = get_time_ticks();
	delete_sparse_eri_indices(&eri_indices);
	printf("Done.\n");
	printf("wall clock time: %g seconds\n", (tick_end_integrals - tick_start_integrals) / ticks_per_sec);

	// index of origin
	union cartesian_grid_point_3d pt_origin = { 0 };
	const glong iorigin = cartesian_grid_point_3d_to_linear_index(&grid, &pt_origin);
	const glong idx_tensor = ((iorigin * num_points + iorigin) * num_points + iorigin) * num_points + iorigin;
	printf("ERI for all Gausslets at origin: %.17g\n", sparse_eri_gausslet_integrals_get_value(&eri_integrals, idx_tensor));

	delete_sparse_eri_gausslet_integrals(&eri_integrals);
	aligned_free(gdata.coefficients);

	return 0;
}
