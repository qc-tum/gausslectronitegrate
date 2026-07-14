#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>
#include "erida_integrals.h"
#include "symmetry.h"
#include "index_list.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the Coulomb overlap integral of two normalizes Gaussians
/// `g(r) = exp(-||r / w||^2) / (sqrt(pi) * w)^3` in three dimensions
/// with relative distance `dist`.
///
static double gaussian_coulomb_integral_3d(const double w, const double dist)
{
	if (dist > 0) {
		return erf(dist / (sqrt(2.) * w)) / dist;
	}
	else {
		return sqrt(2. / M_PI) / w;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Temporary structure storing pre-computed products of Gausslet factors.
///
struct gausslet_factor_products
{
	double* products;      //!< product values
	struct range indices;  //!< logical indices
};


//________________________________________________________________________________________________________________________
///
/// \brief Pre-compute products of Gausslet factors for evaluating electron repulsion integrals using the integral diagonal approximation (ERIDA).
///
static void compute_gausslet_factor_products(const struct gausslet_data* gdata, const double tol, struct gausslet_factor_products* gfp)
{
	// initialize with zero (for the case that all products are filtered out based on the tolerance)
	gfp->indices.istart = 0;
	gfp->indices.num = 0;

	// 2 pi / 9
	const double prefac = 0.69813170079773183077;

	// convolution indices
	assert(gdata->indices.num > 0);
	const struct range all_indices = {
		.istart = -(gdata->indices.num - 1),
		.num    = 2 * gdata->indices.num - 1,
	};

	// convolution
	double* all_products = aligned_calloc(all_indices.num * sizeof(all_products[0]));
	for (glong i = 0; i < gdata->indices.num; ++i)
	{
		for (glong j = 0; j < gdata->indices.num; ++j)
		{
			const glong idx = i - j - all_indices.istart;
			assert(0 <= idx && idx < all_indices.num);
			all_products[idx] += prefac * gdata->coefficients[i] * gdata->coefficients[j];
		}
	}

	// filter out small numbers
	glong min_index = -1;
	for (glong m = 0; m < all_indices.num; ++m) {
		if (fabs(all_products[m]) > tol) {
			min_index = m;
			break;
		}
	}
	glong max_index = -1;
	for (glong m = all_indices.num - 1; m >= 0; --m) {
		if (fabs(all_products[m]) > tol) {
			max_index = m;
			break;
		}
	}
	if (min_index != -1)
	{
		assert(max_index != -1);
		assert(min_index <= max_index);

		gfp->indices.istart = all_indices.istart + min_index;
		gfp->indices.num = max_index - min_index + 1;

		gfp->products = aligned_malloc(gfp->indices.num * sizeof(gfp->products[0]));
		memcpy(gfp->products, &all_products[min_index], gfp->indices.num * sizeof(gfp->products[0]));
	}

	aligned_free(all_products);
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete pre-computed products of Gausslet factors structure (free memory).
///
static void delete_gausslet_factor_products(struct gausslet_factor_products* gfp)
{
	if (gfp->products != NULL) {
		aligned_free(gfp->products);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate a single electron repulsion integral using the integral diagonal approximation (ERIDA) for Gausslet orbitals.
///
double compute_erida_gausslet_integral(const struct gausslet_data* gdata, const union cartesian_grid_point_3d points[2], const double tol)
{
	// sqrt(2) / 3
	const double sqrt2_3 = 0.47140452079103168293;

	struct gausslet_factor_products gfp;
	compute_gausslet_factor_products(gdata, tol, &gfp);

	const double center_diff[3] = {
		points[0].x - points[1].x,
		points[0].y - points[1].y,
		points[0].z - points[1].z,
	};

	double val = 0;
	double pt[3];
	for (glong sx = 0; sx < gfp.indices.num; ++sx)
	{
		if (fabs(gfp.products[sx]) <= tol) {
			continue;
		}

		pt[0] = center_diff[0] + (gfp.indices.istart + sx) / 3.0;

		for (glong sy = 0; sy < gfp.indices.num; ++sy)
		{
			if (fabs(gfp.products[sy]) <= tol) {
				continue;
			}

			pt[1] = center_diff[1] + (gfp.indices.istart + sy) / 3.0;

			for (glong sz = 0; sz < gfp.indices.num; ++sz)
			{
				if (fabs(gfp.products[sz]) <= tol) {
					continue;
				}

				pt[2] = center_diff[2] + (gfp.indices.istart + sz) / 3.0;

				val += gfp.products[sx] * gfp.products[sy] * gfp.products[sz] * gaussian_coulomb_integral_3d(sqrt2_3, vec3_norm(pt));
			}
		}
	}

	delete_gausslet_factor_products(&gfp);

	return val;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the minimal combined index tuple (i, j) appearing within the octahedral group orbit
/// and i <-> j permutation symmetry.
///
glong minimum_octahedral_orbit_erida_tensor_index(const glong num_points, const glong* octahedral_perm[48], const glong i, const glong j)
{
	assert(0 <= i && i < num_points);
	assert(0 <= j && j < num_points);

	glong min_index = GLONG_MAX;

	for (int n = 0; n < 48; ++n)
	{
		glong a = octahedral_perm[n][i];
		glong b = octahedral_perm[n][j];

		// permutation symmetry
		if (a > b)
		{
			// swap
			glong tmp = a;
			a = b;
			b = tmp;
		}

		min_index = lmin(min_index, a * num_points + b);
	}

	return min_index;
}


//________________________________________________________________________________________________________________________
///
/// \brief Test whether the provided 'index' is smaller than or equal to
/// the minimal combined index tuple (i, j) appearing within the octahedral group orbit
/// and i <-> j permutation symmetry.
///
static inline bool is_minimum_octahedral_orbit_erida_tensor_index(
	const glong num_points, const glong* octahedral_perm[48],
	const glong i, const glong j, const glong index)
{
	assert(0 <= i && i < num_points);
	assert(0 <= j && j < num_points);

	for (int n = 0; n < 48; ++n)
	{
		glong a = octahedral_perm[n][i];
		glong b = octahedral_perm[n][j];

		// permutation symmetry
		if (a > b)
		{
			// swap
			glong tmp = a;
			a = b;
			b = tmp;
		}

		if (a * num_points + b < index) {
			return false;  // fast return
		}
	}

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Comparison function for sorting.
///
static int compare_indices(const void* a, const void* b)
{
	const glong x = *((glong*)a);
	const glong y = *((glong*)b);

	if (x < y) {
		return -1;
	}
	if (x > y) {
		return 1;
	}
	// x == y
	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Enumerate the linearized "diagonal" electron repulsion integral indices (i, j) after symmetry reduction,
/// using translational invariance by only retaining integrals
/// with the center of the enclosing box of the orbital grid points at the origin,
/// and exploiting octahedral point group symmetry and i <-> j permutation symmetry.
///
void enumerate_symmetry_reduced_erida_indices(const struct cartesian_grid_3d* grid, struct sparse_erida_indices* ret)
{
	// copy grid information
	ret->grid = *grid;

	glong* octahedral_perm[48];
	evaluate_octahedral_grid_permutations(grid, octahedral_perm);

	const glong num_points = cartesian_grid_3d_num_points(grid);

	struct index_list index_list = { 0 };

	#pragma omp parallel for schedule(dynamic) collapse(2)
	for (glong icx = 0; icx < grid->coord_range[0].num; ++icx)
	{
		for (glong jcx = 0; jcx < grid->coord_range[0].num; ++jcx)
		{
			// order i <= j implies icx <= jcx
			if (icx > jcx) {
				continue;
			}

			// exploit translational invariance
			{
				// x-coordinate of orbital box center times 2
				const glong center_x = 2 * grid->coord_range[0].istart + lmin(icx, jcx) + lmax(icx, jcx);
				if (center_x < -1 || 1 < center_x) {
					continue;
				}
			}

			for (glong icy = 0; icy < grid->coord_range[1].num; ++icy)
			{
				for (glong jcy = 0; jcy < grid->coord_range[1].num; ++jcy)
				{
					// exploit translational invariance
					{
						// y-coordinate of orbital box center times 2
						const glong center_y = 2 * grid->coord_range[1].istart + lmin(icy, jcy) + lmax(icy, jcy);
						if (center_y < -1 || 1 < center_y) {
							continue;
						}
					}

					for (glong icz = 0; icz < grid->coord_range[2].num; ++icz)
					{
						for (glong jcz = 0; jcz < grid->coord_range[2].num; ++jcz)
						{
							// exploit translational invariance
							{
								// z-coordinate of orbital box center times 2
								const glong center_z = 2 * grid->coord_range[2].istart + lmin(icz, jcz) + lmax(icz, jcz);
								if (center_z < -1 || 1 < center_z) {
									continue;
								}
							}

							const glong idx_i = cartesian_grid_3d_cartesian_to_linear_index(grid, icx, icy, icz);
							const glong idx_j = cartesian_grid_3d_cartesian_to_linear_index(grid, jcx, jcy, jcz);

							const glong idx_tensor = idx_i * num_points + idx_j;

							// use point group and permutation symmetries to avoid redundant calculations
							if (is_minimum_octahedral_orbit_erida_tensor_index(num_points, (const glong**)octahedral_perm, idx_i, idx_j, idx_tensor))
							{
								#pragma omp critical
								{
									index_list_add_entry(&index_list, idx_tensor);
								}
							}
						}
					}
				}
			}
		}
	}

	for (int i = 0; i < 48; ++i) {
		aligned_free(octahedral_perm[i]);
	}

	// transfer list entries to output structure
	ret->num = index_list.size;
	ret->two_indices = aligned_malloc(ret->num * sizeof(ret->two_indices[0]));
	index_list_to_array(&index_list, ret->two_indices);
	delete_index_list(&index_list);

	// sort entries
	qsort(ret->two_indices, ret->num, sizeof(ret->two_indices[0]), compare_indices);
	// indices must be unique
	#ifndef NDEBUG
	for (glong i = 0; i < ret->num - 1; ++i) {
		assert(ret->two_indices[i] < ret->two_indices[i + 1]);
	}
	#endif
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete "diagonal" electron repulsion integral indices structure (free memory).
///
void delete_sparse_erida_indices(struct sparse_erida_indices* indices)
{
	aligned_free(indices->two_indices);
	indices->num = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the electron repulsion integrals using the integral diagonal approximation (ERIDA) for Gausslet orbitals
/// and the provided linearized indices (i, j).
///
void compute_sparse_erida_gausslet_integrals(const struct gausslet_data* gdata,
	const struct cartesian_grid_3d* grid, const glong* two_indices, const glong num_indices,
	const double tol, struct sparse_erida_gausslet_integrals* erida)
{
	// sqrt(2) / 3
	const double sqrt2_3 = 0.47140452079103168293;

	struct gausslet_factor_products gfp;
	compute_gausslet_factor_products(gdata, tol, &gfp);

	// copy grid information
	erida->grid = *grid;

	// copy indices
	erida->num_entries = num_indices;
	erida->two_indices = aligned_malloc(num_indices * sizeof(erida->two_indices[0]));
	memcpy(erida->two_indices, two_indices, num_indices * sizeof(erida->two_indices[0]));

	erida->integral_values = aligned_malloc(erida->num_entries * sizeof(erida->integral_values[0]));

	const glong num_points = cartesian_grid_3d_num_points(grid);

	#pragma omp parallel for
	for (glong n = 0; n < num_indices; ++n)
	{
		// decode (i, j) indices
		glong idx_i, idx_j;
		{
			glong rem = two_indices[n];
			idx_j = rem % num_points;
			rem  /= num_points;
			idx_i = rem;
		}

		// decode grid points
		union cartesian_grid_point_3d pt_i, pt_j;
		linear_index_to_cartesian_grid_point_3d(grid, idx_i, &pt_i);
		linear_index_to_cartesian_grid_point_3d(grid, idx_j, &pt_j);

		const double center_diff[3] = {
			pt_i.x - pt_j.x,
			pt_i.y - pt_j.y,
			pt_i.z - pt_j.z,
		};

		double val = 0;
		double pt[3];
		for (glong sx = 0; sx < gfp.indices.num; ++sx)
		{
			if (fabs(gfp.products[sx]) <= tol) {
				continue;
			}

			pt[0] = center_diff[0] + (gfp.indices.istart + sx) / 3.0;

			for (glong sy = 0; sy < gfp.indices.num; ++sy)
			{
				if (fabs(gfp.products[sy]) <= tol) {
					continue;
				}

				pt[1] = center_diff[1] + (gfp.indices.istart + sy) / 3.0;

				for (glong sz = 0; sz < gfp.indices.num; ++sz)
				{
					if (fabs(gfp.products[sz]) <= tol) {
						continue;
					}

					pt[2] = center_diff[2] + (gfp.indices.istart + sz) / 3.0;

					val += gfp.products[sx] * gfp.products[sy] * gfp.products[sz] * gaussian_coulomb_integral_3d(sqrt2_3, vec3_norm(pt));
				}
			}
		}

		erida->integral_values[n] = val;
	}

	delete_gausslet_factor_products(&gfp);
}


//________________________________________________________________________________________________________________________
///
/// \brief Retrieve the value corresponding to 'index', or 0 if 'index' cannot be found.
///
double sparse_erida_gausslet_integrals_get_value(const struct sparse_erida_gausslet_integrals* erida, const glong index)
{
	// search interval: [lower, upper)
	glong lower = 0;
	glong upper = erida->num_entries;
	while (true)
	{
		if (lower >= upper) {
			return 0;  // 'index' not found
		}
		const glong i = (lower + upper) / 2;
		if (index < erida->two_indices[i]) {
			upper = i;
		}
		else if (index > erida->two_indices[i]) {
			lower = i + 1;
		}
		else {
			// found it
			return erida->integral_values[i];
		}
	}
}



//________________________________________________________________________________________________________________________
///
/// \brief Project the electron repulsion integrals using the integral diagonal approximation (ERIDA)
/// onto the specified basis along each axis. The output tensor has degree four.
///
void project_sparse_erida_gausslet_integrals(const struct sparse_erida_gausslet_integrals* erida, const double* restrict basis, const glong num_states, double* restrict eri_proj)
{
	const glong num_points = cartesian_grid_3d_num_points(&erida->grid);
	assert(num_points > 0);

	glong* octahedral_perm[48];
	evaluate_octahedral_grid_permutations(&erida->grid, octahedral_perm);

	memset(eri_proj, 0, num_states * num_states * num_states * num_states * sizeof(eri_proj[0]));

	#pragma omp parallel for collapse(2)
	for (glong icx = 0; icx < erida->grid.coord_range[0].num; ++icx)
	{
		for (glong jcx = 0; jcx < erida->grid.coord_range[0].num; ++jcx)
		{
			// x-coordinate of orbital box center times 2
			const glong center_x = 2 * erida->grid.coord_range[0].istart + lmin(icx, jcx) + lmax(icx, jcx);
			const glong trans_x = center_x / 2;
			assert(-1 <= center_x - 2 * trans_x && center_x - 2 * trans_x <= 1);

			for (glong icy = 0; icy < erida->grid.coord_range[1].num; ++icy)
			{
				for (glong jcy = 0; jcy < erida->grid.coord_range[1].num; ++jcy)
				{
					// y-coordinate of orbital box center times 2
					const glong center_y = 2 * erida->grid.coord_range[1].istart + lmin(icy, jcy) + lmax(icy, jcy);
					const glong trans_y = center_y / 2;
					assert(-1 <= center_y - 2 * trans_y && center_y - 2 * trans_y <= 1);

					for (glong icz = 0; icz < erida->grid.coord_range[2].num; ++icz)
					{
						for (glong jcz = 0; jcz < erida->grid.coord_range[2].num; ++jcz)
						{
							// z-coordinate of orbital box center times 2
							const glong center_z = 2 * erida->grid.coord_range[2].istart + lmin(icz, jcz) + lmax(icz, jcz);
							const glong trans_z = center_z / 2;
							assert(-1 <= center_z - 2 * trans_z && center_z - 2 * trans_z <= 1);

							const glong idx_i = cartesian_grid_3d_cartesian_to_linear_index(&erida->grid, icx, icy, icz);
							const glong idx_j = cartesian_grid_3d_cartesian_to_linear_index(&erida->grid, jcx, jcy, jcz);

							const glong idx_i_p = cartesian_grid_3d_cartesian_to_linear_index(&erida->grid, icx - trans_x, icy - trans_y, icz - trans_z);
							const glong idx_j_p = cartesian_grid_3d_cartesian_to_linear_index(&erida->grid, jcx - trans_x, jcy - trans_y, jcz - trans_z);
							const glong idx_tensor_sparse = minimum_octahedral_orbit_erida_tensor_index(
								num_points, (const glong**)octahedral_perm, idx_i_p, idx_j_p);

							const double val = sparse_erida_gausslet_integrals_get_value(erida, idx_tensor_sparse);

							for (glong p = 0; p < num_states; ++p)
							{
								for (glong q = 0; q < num_states; ++q)
								{
									for (glong r = 0; r < num_states; ++r)
									{
										for (glong s = 0; s < num_states; ++s)
										{
											#pragma omp atomic
											eri_proj[((p * num_states + q) * num_states + r) * num_states + s] +=
												  basis[idx_i * num_states + p]
												* basis[idx_i * num_states + q]
												* basis[idx_j * num_states + r]
												* basis[idx_j * num_states + s]
												* val;
										}
									}
								}
							}

						}
					}
				}
			}
		}
	}

	for (int i = 0; i < 48; ++i) {
		aligned_free(octahedral_perm[i]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Fill the dense matrix entries with the electron repulsion integrals using the integral diagonal approximation (ERIDA).
///
void fill_dense_erida_matrix(const struct sparse_erida_gausslet_integrals* erida_sparse, const struct cartesian_grid_3d* grid_dense, double* erida_matrix)
{
	const glong num_points_sparse = cartesian_grid_3d_num_points(&erida_sparse->grid);
	const glong num_points_dense  = cartesian_grid_3d_num_points(grid_dense);
	assert(num_points_sparse > 0);
	assert(num_points_dense  > 0);

	glong* octahedral_perm[48];
	evaluate_octahedral_grid_permutations(&erida_sparse->grid, octahedral_perm);

	#pragma omp parallel for collapse(2)
	for (glong icx = 0; icx < grid_dense->coord_range[0].num; ++icx)
	{
		for (glong jcx = 0; jcx < grid_dense->coord_range[0].num; ++jcx)
		{
			// x-coordinate of orbital box center times 2
			const glong center_x = 2 * grid_dense->coord_range[0].istart + lmin(icx, jcx) + lmax(icx, jcx);
			const glong trans_x = center_x / 2;
			assert(-1 <= center_x - 2 * trans_x && center_x - 2 * trans_x <= 1);

			for (glong icy = 0; icy < grid_dense->coord_range[1].num; ++icy)
			{
				for (glong jcy = 0; jcy < grid_dense->coord_range[1].num; ++jcy)
				{
					// y-coordinate of orbital box center times 2
					const glong center_y = 2 * grid_dense->coord_range[1].istart + lmin(icy, jcy) + lmax(icy, jcy);
					const glong trans_y = center_y / 2;
					assert(-1 <= center_y - 2 * trans_y && center_y - 2 * trans_y <= 1);

					for (glong icz = 0; icz < grid_dense->coord_range[2].num; ++icz)
					{
						for (glong jcz = 0; jcz < grid_dense->coord_range[2].num; ++jcz)
						{
							// z-coordinate of orbital box center times 2
							const glong center_z = 2 * grid_dense->coord_range[2].istart + lmin(icz, jcz) + lmax(icz, jcz);
							const glong trans_z = center_z / 2;
							assert(-1 <= center_z - 2 * trans_z && center_z - 2 * trans_z <= 1);

							const glong idx_i = cartesian_grid_3d_cartesian_to_linear_index(grid_dense, icx, icy, icz);
							const glong idx_j = cartesian_grid_3d_cartesian_to_linear_index(grid_dense, jcx, jcy, jcz);
							const glong idx_tensor_dense  = idx_i * num_points_dense + idx_j;

							const union cartesian_grid_point_3d pt_i_p = {
								.x = grid_dense->coord_range[0].istart + icx - trans_x,
								.y = grid_dense->coord_range[1].istart + icy - trans_y,
								.z = grid_dense->coord_range[2].istart + icz - trans_z,
							};
							const union cartesian_grid_point_3d pt_j_p = {
								.x = grid_dense->coord_range[0].istart + jcx - trans_x,
								.y = grid_dense->coord_range[1].istart + jcy - trans_y,
								.z = grid_dense->coord_range[2].istart + jcz - trans_z,
							};
							const glong idx_i_p = cartesian_grid_point_3d_to_linear_index(&erida_sparse->grid, &pt_i_p);
							const glong idx_j_p = cartesian_grid_point_3d_to_linear_index(&erida_sparse->grid, &pt_j_p);
							const glong idx_tensor_sparse = minimum_octahedral_orbit_erida_tensor_index(num_points_sparse, (const glong**)octahedral_perm, idx_i_p, idx_j_p);

							erida_matrix[idx_tensor_dense] = sparse_erida_gausslet_integrals_get_value(erida_sparse, idx_tensor_sparse);
						}
					}
				}
			}
		}
	}

	for (int i = 0; i < 48; ++i) {
		aligned_free(octahedral_perm[i]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete the sparse electron repulsion integral storage structure (free memory).
///
void delete_sparse_erida_gausslet_integrals(struct sparse_erida_gausslet_integrals* erida)
{
	aligned_free(erida->two_indices);
	aligned_free(erida->integral_values);
	erida->num_entries = 0;
}
