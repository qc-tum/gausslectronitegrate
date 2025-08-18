#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>
#include "nuclear_integrals.h"
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
/// \brief Temporary structure storing pre-computed products of Gausslet coefficients
/// and Gaussian factors for evaluating nuclear overlap integrals.
///
struct nuclear_gausslet_factors
{
	double** factors;       //!< Gaussian factors: factors[i] stores the factors for shift index 'i'
	struct range* indices;  //!< logical indices, for each shift
	struct range shifts;    //!< shifts
};


//________________________________________________________________________________________________________________________
///
/// \brief Pre-compute products of Gausslet coefficients and Gaussian factors for evaluating nuclear overlap integrals.
///
static void compute_nuclear_gausslet_factors(const struct gausslet_data* gdata, const struct range* shifts, const double tol, struct nuclear_gausslet_factors* ngf)
{
	// copy shifts
	ngf->shifts = *shifts;

	// factor indices: [2 * "smallest Gausslet index", 2 * "largest Gausslet index"]
	assert(gdata->indices.istart < 0);
	assert(gdata->indices.istart + gdata->indices.num - 1 > 0);
	const struct range all_indices = {
		.istart = 2 * gdata->indices.istart,
		.num    = 2 * gdata->indices.num - 1,
	};

	ngf->factors = aligned_calloc(ngf->shifts.num * sizeof(ngf->factors[0]));
	ngf->indices = aligned_calloc(ngf->shifts.num * sizeof(ngf->indices[0]));

	for (long k = 0; k < ngf->shifts.num; ++k)
	{
		const long shift = ngf->shifts.istart + k;

		double* all_factors = aligned_calloc(all_indices.num * sizeof(all_factors[0]));

		for (long i = 0; i < gdata->indices.num; ++i)
		{
			const double coeff_a = gdata->coefficients[i];

			for (long j = 0; j < gdata->indices.num; ++j)
			{
				const double coeff_b = gdata->coefficients[j];

				// using that all_indices.istart == 2 * gdata->indices.istart
				const long idx = i + j;
				assert(idx < all_indices.num);
				all_factors[idx] += coeff_a * coeff_b * exp(-square(0.5 * (3*shift + (i - j))));
			}
		}

		// filter out small factors
		long min_index = -1;
		for (long m = 0; m < all_indices.num; ++m) {
			if (fabs(all_factors[m]) > tol) {
				min_index = m;
				break;
			}
		}
		long max_index = -1;
		for (long m = all_indices.num - 1; m >= 0; --m) {
			if (fabs(all_factors[m]) > tol) {
				max_index = m;
				break;
			}
		}
		if (min_index != -1)
		{
			assert(max_index != -1);
			assert(min_index <= max_index);

			ngf->indices[k].istart = all_indices.istart + min_index;
			ngf->indices[k].num = max_index - min_index + 1;

			ngf->factors[k] = aligned_malloc(ngf->indices[k].num * sizeof(ngf->factors[k][0]));
			memcpy(ngf->factors[k], &all_factors[min_index], ngf->indices[k].num * sizeof(ngf->factors[k][0]));
		}

		aligned_free(all_factors);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete Gausslet coefficients and Gaussian factors structure (free memory).
///
static void delete_nuclear_gausslet_factors(struct nuclear_gausslet_factors* ngf)
{
	for (long i = 0; i < ngf->shifts.num; ++i) {
		if (ngf->factors[i] != NULL) {
			aligned_free(ngf->factors[i]);
		}
	}
	aligned_free(ngf->factors);
	aligned_free(ngf->indices);
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the nuclear overlap integrals for two Gausslet orbitals
/// with coordinates at integer `centers`.
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

	struct nuclear_gausslet_factors ngf;
	{
		const long max_range = lmax(lmax(
			grid->coord_range[0].num,
			grid->coord_range[1].num),
			grid->coord_range[2].num);
		assert(max_range > 0);

		// unique shifts
		const struct range shifts = {
			.istart = -max_range,
			.num    = 2 * max_range + 1,
		};

		compute_nuclear_gausslet_factors(gdata, &shifts, tol, &ngf);
	}

	const long num_points = cartesian_grid_3d_num_points(grid);
	ngi->integral_values = aligned_calloc(num_points * num_points * sizeof(ngi->integral_values[0]));

	for (long icx = 0; icx < grid->coord_range[0].num; ++icx)
	{
		for (long jcx = 0; jcx < grid->coord_range[0].num; ++jcx)
		{
			const double center_x = grid->coord_range[0].istart + 0.5 * (icx + jcx);

			const long idx_shift_x  = (icx - jcx) - ngf.shifts.istart;
			const double* factors_x = ngf.factors[idx_shift_x];
			const long istart_x     = ngf.indices[idx_shift_x].istart;
			const long num_x        = ngf.indices[idx_shift_x].num;

			for (long icy = 0; icy < grid->coord_range[1].num; ++icy)
			{
				for (long jcy = 0; jcy < grid->coord_range[1].num; ++jcy)
				{
					const double center_y = grid->coord_range[1].istart + 0.5 * (icy + jcy);

					const long idx_shift_y  = (icy - jcy) - ngf.shifts.istart;
					const double* factors_y = ngf.factors[idx_shift_y];
					const long istart_y     = ngf.indices[idx_shift_y].istart;
					const long num_y        = ngf.indices[idx_shift_y].num;

					for (long icz = 0; icz < grid->coord_range[2].num; ++icz)
					{
						for (long jcz = 0; jcz < grid->coord_range[2].num; ++jcz)
						{
							const double center_z = grid->coord_range[2].istart + 0.5 * (icz + jcz);

							const long idx_shift_z  = (icz - jcz) - ngf.shifts.istart;
							const double* factors_z = ngf.factors[idx_shift_z];
							const long istart_z     = ngf.indices[idx_shift_z].istart;
							const long num_z        = ngf.indices[idx_shift_z].num;

							const long idx_i = cartesian_grid_3d_cartesian_to_linear_index(grid, icx, icy, icz);
							const long idx_j = cartesian_grid_3d_cartesian_to_linear_index(grid, jcx, jcy, jcz);

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

	delete_nuclear_gausslet_factors(&ngf);
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
