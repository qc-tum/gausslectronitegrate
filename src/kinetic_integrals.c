#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "kinetic_integrals.h"
#include "aligned_memory.h"
#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the one-dimensional overlap integral of the derivative functions
/// of two elementary Gaussians with relative index shift `d`.
///
static inline double elementary_gaussian_derivative_integral(const long d)
{
	const double d2 = d * d;
	return 1.5 * sqrt(M_PI) * (1 - 0.5 * d2) * exp(-0.25 * d2);
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the kinetic overlap integral of two Gausslet functions with relative index shift `shift`.
///
static double kinetic_gausslet_integral(const struct gausslet_data* gdata, const long shift)
{
	double val = 0;
	for (long i = 0; i < gdata->indices.num; ++i)
	{
		for (long j = 0; j < gdata->indices.num; ++j)
		{
			val += 0.5 * gdata->coefficients[i] * gdata->coefficients[j] * elementary_gaussian_derivative_integral(3 * shift + (i - j));
		}
	}

	return val;
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the kinetic overlap integrals for Gausslet orbitals located at the specified grid points.
///
void compute_kinetic_gausslet_integrals(const struct gausslet_data* gdata, const struct cartesian_grid_3d* grid, struct kinetic_gausslet_integrals* kgi)
{
	// copy grid information
	kgi->grid = *grid;

	const long max_range = lmax(lmax(
		grid->coord_range[0].num,
		grid->coord_range[1].num),
		grid->coord_range[2].num);
	assert(max_range > 0);

	double* kint1d = aligned_malloc(max_range * sizeof(kint1d[0]));
	for (long shift = 0; shift < max_range; ++shift) {
		kint1d[shift] = kinetic_gausslet_integral(gdata, shift);
	}

	const long num_points = cartesian_grid_3d_num_points(grid);
	kgi->integral_values = aligned_calloc(num_points * num_points * sizeof(kgi->integral_values[0]));

	#pragma omp parallel for schedule(dynamic) collapse(6)
	for (long icx = 0; icx < grid->coord_range[0].num; ++icx)
	{
		for (long jcx = 0; jcx < grid->coord_range[0].num; ++jcx)
		{
			for (long icy = 0; icy < grid->coord_range[1].num; ++icy)
			{
				for (long jcy = 0; jcy < grid->coord_range[1].num; ++jcy)
				{
					for (long icz = 0; icz < grid->coord_range[2].num; ++icz)
					{
						for (long jcz = 0; jcz < grid->coord_range[2].num; ++jcz)
						{
							const long idx_i = cartesian_grid_3d_cartesian_to_linear_index(grid, icx, icy, icz);
							const long idx_j = cartesian_grid_3d_cartesian_to_linear_index(grid, jcx, jcy, jcz);

							// structurally K x I x I + I x K x I + I x I x K
							kgi->integral_values[idx_i * num_points + idx_j] =
								kint1d[labs(icx - jcx)] * (icy == jcy ? 1 : 0)    * (icz == jcz ? 1 : 0) +
								(icx == jcx ? 1 : 0)    * kint1d[labs(icy - jcy)] * (icz == jcz ? 1 : 0) +
								(icx == jcx ? 1 : 0)    * (icy == jcy ? 1 : 0)    * kint1d[labs(icz - jcz)];
						}
					}
				}
			}
		}
	}

	aligned_free(kint1d);
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete kinetic overlap integral storage structure (free memory).
///
void delete_kinetic_gausslet_integrals(struct kinetic_gausslet_integrals* kgi)
{
	aligned_free(kgi->integral_values);
}
