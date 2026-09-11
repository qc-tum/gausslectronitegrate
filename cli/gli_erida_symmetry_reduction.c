#include <stdlib.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "erida_integrals.h"
#include "aligned_memory.h"
#include "timing.h"


int main(int argc, char* argv[])
{
	const char* syntax = "gli_erida_symmetry_reduction <max grid length>";

	// parse command line arguments
	if (argc != 2)
	{
		fprintf(stderr, "expecting one input parameter; syntax: %s\n", syntax);
		return -1;
	}
	const glong max_grid_length = strtol(argv[1], NULL, 10);
	if (max_grid_length <= 0 || (max_grid_length % 2 == 0))
	{
		fprintf(stderr, "'max_grid_length' must be a positive odd integer; syntax: %s\n", syntax);
		return -1;
	}

	#ifdef _OPENMP
	printf("maximum number of OpenMP threads: %d\n", omp_get_max_threads());
	#else
	printf("OpenMP not available\n");
	#endif

	// get the tick resolution
	const double ticks_per_sec = (double)get_tick_resolution();

	printf("Enumerating symmetry-reduced electron repulsion integral (ERIDA) indices...\n");
	printf("grid dim   # points  # dense ERIDA  # symm. ERIDA  reduction     wall time [seconds]\n");

	for (glong grid_length = 1; grid_length <= max_grid_length; grid_length += 2)
	{
		// require a rotationally symmetric grid
		const struct cartesian_grid_3d grid = {
			.coord_range = {
				{ .istart = -(grid_length - 1) / 2, .num = grid_length },
				{ .istart = -(grid_length - 1) / 2, .num = grid_length },
				{ .istart = -(grid_length - 1) / 2, .num = grid_length },
			}
		};
		const glong num_points = cartesian_grid_3d_num_points(&grid);

		const uint64_t tick_start = get_time_ticks();
		struct sparse_erida_indices erida_indices;
		enumerate_symmetry_reduced_erida_indices(&grid, &erida_indices);
		const uint64_t tick_end = get_time_ticks();

		const double time = (tick_end - tick_start) / ticks_per_sec;
		const glong num_erida_dense = num_points * num_points;
		printf(grid_length < 10 ?
				"%li x %li x %li     %5li  %13li  %13li  %-12g  %10g\n" :
				"%li x %li x %li  %5li  %13li  %13li  %-12g  %10g\n",
			grid.coord_range[0].num, grid.coord_range[1].num, grid.coord_range[2].num,
			num_points,
			num_erida_dense,
			erida_indices.num,
			(double)erida_indices.num / num_erida_dense,
			time);

		delete_sparse_erida_indices(&erida_indices);
	}

	printf("Done.\n");

	return 0;
}
