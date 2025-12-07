#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "nuclear_integrals.h"
#include "gausslet_factors.h"
#include "aligned_memory.h"
#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the nuclear overlap integral of a normalizes Gaussians
/// `g(r) = exp(-||r / w||^2) / (sqrt(pi) * w)^3` with a nuclear point charge
/// in three dimensions with relative distance `dist`.
///
static double gaussian_nuclear_integral_3d(const double w, const double dist)
{
	if (dist > 0) {
		return erf(dist / w) / dist;
	}
	else {
		return 2 / (sqrt(M_PI) * w);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the nuclear overlap integrals for Gausslet orbitals and the specified nuclear positions.
///
void compute_nuclear_gausslet_integrals(
	const struct gausslet_data* gdata, const struct cartesian_grid_3d* grid,
	const struct atomic_nucleus* nuclei, const int num_nuclei,
	const double tol, struct nuclear_gausslet_integrals* ngi)
{
	// copy grid information
	ngi->grid = *grid;
	// copy nuclear information
	ngi->nuclei = aligned_malloc(num_nuclei * sizeof(ngi->nuclei[0]));
	memcpy(ngi->nuclei, nuclei, num_nuclei * sizeof(ngi->nuclei[0]));
	ngi->num_nuclei = num_nuclei;

	const double prefac = cubic_power(sqrt(M_PI) / 3.);

	struct gausslet_factors gf;
	{
		const long max_range = lmax(lmax(
			grid->coord_range[0].num,
			grid->coord_range[1].num),
			grid->coord_range[2].num);
		assert(max_range > 0);

		// unique shifts
		const struct range shifts = {
			.istart = -max_range + 1,
			.num    = 2 * max_range - 1,
		};

		compute_gausslet_factors(gdata, &shifts, tol, &gf);
	}

	const long num_points = cartesian_grid_3d_num_points(grid);
	ngi->integral_values = aligned_calloc(num_points * num_points * sizeof(ngi->integral_values[0]));

	#pragma omp parallel for schedule(dynamic) collapse(2)
	for (long icx = 0; icx < grid->coord_range[0].num; ++icx)
	{
		for (long jcx = 0; jcx < grid->coord_range[0].num; ++jcx)
		{
			const double center_x = grid->coord_range[0].istart + 0.5 * (icx + jcx);

			const long ishift_x     = (icx - jcx) - gf.shifts.istart;
			const double* factors_x = gf.factors[ishift_x];
			const long istart_x     = gf.indices[ishift_x].istart;
			const long num_x        = gf.indices[ishift_x].num;

			for (long icy = 0; icy < grid->coord_range[1].num; ++icy)
			{
				for (long jcy = 0; jcy < grid->coord_range[1].num; ++jcy)
				{
					const double center_y = grid->coord_range[1].istart + 0.5 * (icy + jcy);

					const long ishift_y     = (icy - jcy) - gf.shifts.istart;
					const double* factors_y = gf.factors[ishift_y];
					const long istart_y     = gf.indices[ishift_y].istart;
					const long num_y        = gf.indices[ishift_y].num;

					for (long icz = 0; icz < grid->coord_range[2].num; ++icz)
					{
						for (long jcz = 0; jcz < grid->coord_range[2].num; ++jcz)
						{
							const double center_z = grid->coord_range[2].istart + 0.5 * (icz + jcz);

							const long ishift_z     = (icz - jcz) - gf.shifts.istart;
							const double* factors_z = gf.factors[ishift_z];
							const long istart_z     = gf.indices[ishift_z].istart;
							const long num_z        = gf.indices[ishift_z].num;

							const long idx_i = cartesian_grid_3d_cartesian_to_linear_index(grid, icx, icy, icz);
							const long idx_j = cartesian_grid_3d_cartesian_to_linear_index(grid, jcx, jcy, jcz);

							// use symmetry to avoid redundant calculations
							if (idx_i > idx_j) {
								continue;
							}

							double val = 0;
							for (long sx = 0; sx < num_x; ++sx)
							{
								if (fabs(factors_x[sx]) <= tol) {
									continue;
								}

								const double pt_x = center_x + (istart_x + sx) / 6.0;

								for (long sy = 0; sy < num_y; ++sy)
								{
									if (fabs(factors_y[sy]) <= tol) {
										continue;
									}

									const double pt_y = center_y + (istart_y + sy) / 6.0;

									for (long sz = 0; sz < num_z; ++sz)
									{
										if (fabs(factors_z[sz]) <= tol) {
											continue;
										}

										const double pt_z = center_z + (istart_z + sz) / 6.0;

										const double factor = factors_x[sx] * factors_y[sy] * factors_z[sz];

										for (int k = 0; k < num_nuclei; ++k)
										{
											const double diff[3] = {
												pt_x - nuclei[k].pos[0],
												pt_y - nuclei[k].pos[1],
												pt_z - nuclei[k].pos[2],
											};
											const double dist = vec3_norm(diff);

											val += nuclei[k].charge * factor * gaussian_nuclear_integral_3d(1. / 3,  dist);
										}
									}
								}
							}
							val *= prefac;

							ngi->integral_values[idx_i * num_points + idx_j] = val;
						}
					}
				}
			}
		}
	}

	// fill entries in lower triangular part according to symmetry
	#pragma omp parallel for schedule(dynamic)
	for (long idx_i = 0; idx_i < num_points; ++idx_i) {
		for (long idx_j = 0; idx_j < idx_i; ++idx_j) {
			ngi->integral_values[idx_i * num_points + idx_j] = ngi->integral_values[idx_j * num_points + idx_i];
		}
	}

	delete_gausslet_factors(&gf);
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete nuclear overlap integral storage structure (free memory).
///
void delete_nuclear_gausslet_integrals(struct nuclear_gausslet_integrals* ngi)
{
	aligned_free(ngi->integral_values);
	aligned_free(ngi->nuclei);
	ngi->num_nuclei = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate a single nuclear overlap integral for Gausslet orbitals and the specified nuclear positions.
///
double compute_nuclear_gausslet_integral(
	const struct gausslet_data* gdata, const union cartesian_grid_point_3d points[2],
	const struct atomic_nucleus* nuclei, const int num_nuclei, const double tol)
{
	const double prefac = cubic_power(sqrt(M_PI) / 3.);

	struct gausslet_factors gf;
	{
		const long max_range = lmax(lmax(
			labs(points[0].x - points[1].x),
			labs(points[0].y - points[1].y)),
			labs(points[0].z - points[1].z)) + 1;

		// unique shifts
		const struct range shifts = {
			.istart = -max_range + 1,
			.num    = 2 * max_range - 1,
		};

		compute_gausslet_factors(gdata, &shifts, tol, &gf);
	}

	const double center[3] = {
		0.5 * (points[0].x + points[1].x),
		0.5 * (points[0].y + points[1].y),
		0.5 * (points[0].z + points[1].z),
	};

	const union cartesian_grid_point_3d ishift = {
		(points[0].x - points[1].x) - gf.shifts.istart,
		(points[0].y - points[1].y) - gf.shifts.istart,
		(points[0].z - points[1].z) - gf.shifts.istart,
	};

	const double* factors[3] = {
		gf.factors[ishift.x],
		gf.factors[ishift.y],
		gf.factors[ishift.z],
	};

	const union cartesian_grid_point_3d istart = {
		gf.indices[ishift.x].istart,
		gf.indices[ishift.y].istart,
		gf.indices[ishift.z].istart,
	};

	const union cartesian_grid_point_3d num = {
		gf.indices[ishift.x].num,
		gf.indices[ishift.y].num,
		gf.indices[ishift.z].num,
	};

	double val = 0;
	for (long sx = 0; sx < num.x; ++sx)
	{
		if (fabs(factors[0][sx]) <= tol) {
			continue;
		}

		const double pt_x = center[0] + (istart.x + sx) / 6.0;

		for (long sy = 0; sy < num.y; ++sy)
		{
			if (fabs(factors[1][sy]) <= tol) {
				continue;
			}

			const double pt_y = center[1] + (istart.y + sy) / 6.0;

			for (long sz = 0; sz < num.z; ++sz)
			{
				if (fabs(factors[2][sz]) <= tol) {
					continue;
				}

				const double pt_z = center[2] + (istart.z + sz) / 6.0;

				const double factor = factors[0][sx] * factors[1][sy] * factors[2][sz];

				for (int k = 0; k < num_nuclei; ++k)
				{
					const double diff[3] = {
						pt_x - nuclei[k].pos[0],
						pt_y - nuclei[k].pos[1],
						pt_z - nuclei[k].pos[2],
					};
					const double dist = vec3_norm(diff);

					val += nuclei[k].charge * factor * gaussian_nuclear_integral_3d(1. / 3,  dist);
				}
			}
		}
	}
	val *= prefac;

	delete_gausslet_factors(&gf);

	return val;
}
