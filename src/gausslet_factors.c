#include <math.h>
#include <assert.h>
#include "gausslet_factors.h"
#include "aligned_memory.h"
#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Pre-compute products of Gausslet coefficients and Gaussian factors.
///
void compute_gausslet_factors(const struct gausslet_data* gdata, const struct range* shifts, const double tol, struct gausslet_factors* gf)
{
	// copy shifts
	gf->shifts = *shifts;

	// factor indices: [2 * "smallest Gausslet index", 2 * "largest Gausslet index"]
	assert(gdata->indices.istart < 0);
	assert(gdata->indices.istart + gdata->indices.num - 1 > 0);
	const struct range all_indices = {
		.istart = 2 * gdata->indices.istart,
		.num    = 2 * gdata->indices.num - 1,
	};

	gf->factors = aligned_calloc(gf->shifts.num * sizeof(gf->factors[0]));
	gf->indices = aligned_calloc(gf->shifts.num * sizeof(gf->indices[0]));

	// sqrt(pi) / 3
	const double prefac = 0.590817950301838675766;

	for (glong k = 0; k < gf->shifts.num; ++k)
	{
		const glong shift = gf->shifts.istart + k;

		double* all_factors = aligned_calloc(all_indices.num * sizeof(all_factors[0]));

		for (glong i = 0; i < gdata->indices.num; ++i)
		{
			const double coeff_a = gdata->coefficients[i];

			for (glong j = 0; j < gdata->indices.num; ++j)
			{
				const double coeff_b = gdata->coefficients[j];

				// using that all_indices.istart == 2 * gdata->indices.istart
				const glong idx = i + j;
				assert(idx < all_indices.num);
				all_factors[idx] += prefac * coeff_a * coeff_b * exp(-square(0.5 * (3 * shift + (i - j))));
			}
		}

		// filter out small factors
		glong min_index = -1;
		for (glong m = 0; m < all_indices.num; ++m) {
			if (fabs(all_factors[m]) > tol) {
				min_index = m;
				break;
			}
		}
		glong max_index = -1;
		for (glong m = all_indices.num - 1; m >= 0; --m) {
			if (fabs(all_factors[m]) > tol) {
				max_index = m;
				break;
			}
		}
		if (min_index != -1)
		{
			assert(max_index != -1);
			assert(min_index <= max_index);

			gf->indices[k].istart = all_indices.istart + min_index;
			gf->indices[k].num = max_index - min_index + 1;

			gf->factors[k] = aligned_malloc(gf->indices[k].num * sizeof(gf->factors[k][0]));
			memcpy(gf->factors[k], &all_factors[min_index], gf->indices[k].num * sizeof(gf->factors[k][0]));
		}

		aligned_free(all_factors);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete Gausslet coefficients and Gaussian factors structure (free memory).
///
void delete_gausslet_factors(struct gausslet_factors* gf)
{
	for (glong i = 0; i < gf->shifts.num; ++i) {
		if (gf->factors[i] != NULL) {
			aligned_free(gf->factors[i]);
		}
	}
	aligned_free(gf->factors);
	aligned_free(gf->indices);
}
