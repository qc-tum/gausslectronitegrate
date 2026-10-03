#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>
#include "eri_integrals.h"
#include "gausslet_factors.h"
#include "symmetry.h"
#include "index_list.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the Coulomb overlap integral of two normalizes Gaussians
/// `g(r) = exp(-||r / w||^2) / (sqrt(pi) * w)^3` in three dimensions
/// with relative distance `dist`.
///
static long double gaussian_coulomb_integral_3d(const long double w, const long double dist)
{
	if (dist > 0) {
		return erfl(dist / (sqrtl(2) * w)) / dist;
	}
	else {
		// sqrt(2 / pi)
		return 0.7978845608028653558798921198687637L / w;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Temporary structure storing pre-computed products of Gausslet factors.
///
struct eri_gausslet_factor_products
{
	double** products;      //!< products[i * shifts.num + j] stores the products for shift index pair '(i, j)'
	struct range* indices;  //!< logical indices, for each shift pair
	struct range shifts;    //!< shifts
};


//________________________________________________________________________________________________________________________
///
/// \brief Pre-compute products of Gausslet factors for evaluating electron repulsion integrals.
///
static void compute_eri_gausslet_factor_products(const struct gausslet_data* gdata, const struct range* shifts, const double tol, struct eri_gausslet_factor_products* gfp)
{
	assert(shifts->num > 0);

	// copy shifts
	gfp->shifts = *shifts;

	gfp->products = aligned_calloc(shifts->num * shifts->num * sizeof(gfp->products[0]));
	gfp->indices  = aligned_calloc(shifts->num * shifts->num * sizeof(gfp->indices[0]));

	struct gausslet_factors gf;
	compute_gausslet_factors(gdata, shifts, 0., &gf);

	for (glong k = 0; k < shifts->num; ++k)
	{
		for (glong l = 0; l < shifts->num; ++l)
		{
			const struct range all_indices = range_minus(gf.indices[k], gf.indices[l]);

			// convolution
			double* all_products = aligned_calloc(all_indices.num * sizeof(all_products[0]));
			for (glong i = 0; i < gf.indices[k].num; ++i)
			{
				for (glong j = 0; j < gf.indices[l].num; ++j)
				{
					const glong idx = (gf.indices[k].istart + i) - (gf.indices[l].istart + j) - all_indices.istart;
					assert(0 <= idx && idx < all_indices.num);
					all_products[idx] += gf.factors[k][i] * gf.factors[l][j];
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

				const glong kl = k * shifts->num + l;
				gfp->indices[kl].istart = all_indices.istart + min_index;
				gfp->indices[kl].num = max_index - min_index + 1;

				gfp->products[kl] = aligned_malloc(gfp->indices[kl].num * sizeof(gfp->products[kl][0]));
				memcpy(gfp->products[kl], &all_products[min_index], gfp->indices[kl].num * sizeof(gfp->products[kl][0]));
			}

			aligned_free(all_products);
		}
	}

	delete_gausslet_factors(&gf);
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete pre-computed products of Gausslet factors structure (free memory).
///
static void delete_eri_gausslet_factor_products(struct eri_gausslet_factor_products* gfp)
{
	for (glong i = 0; i < gfp->shifts.num * gfp->shifts.num; ++i) {
		if (gfp->products[i] != NULL) {
			aligned_free(gfp->products[i]);
		}
	}
	aligned_free(gfp->products);
	aligned_free(gfp->indices);
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate a single electron repulsion integral (ERI) for Gausslet orbitals.
///
double compute_eri_gausslet_integral(const struct gausslet_data* gdata, const union cartesian_grid_point_3d points[4], const double tol)
{
	const glong max_range_01 = lmax(lmax(
		labs(points[0].x - points[1].x),
		labs(points[0].y - points[1].y)),
		labs(points[0].z - points[1].z)) + 1;
	const glong max_range_23 = lmax(lmax(
		labs(points[2].x - points[3].x),
		labs(points[2].y - points[3].y)),
		labs(points[2].z - points[3].z)) + 1;
	const glong max_grid_range = lmax(max_range_01, max_range_23);

	struct eri_gausslet_factor_products gfp;
	{
		// unique shifts
		const struct range shifts = {
			.istart = -max_grid_range + 1,
			.num    = 2 * max_grid_range - 1,
		};

		compute_eri_gausslet_factor_products(gdata, &shifts, tol, &gfp);
	}

	// look-up table for Coulomb integrals
	const glong num_table_entries = 3 * lsquare(6 * (max_grid_range - 1) + 2 * (gdata->indices.num - 1)) + 1;
	double* coulomb_integral_table = aligned_calloc(num_table_entries * sizeof(coulomb_integral_table[0]));
	for (glong i = 0; i < num_table_entries; ++i)
	{
		coulomb_integral_table[i] = (double)gaussian_coulomb_integral_3d(1.L / 3, sqrtl(i) / 6);
	}

	glong center_diff_6[3];
	const double* products[3];
	union cartesian_grid_point_3d istart, num;
	for (int i = 0; i < 3; ++i)
	{
		center_diff_6[i] = 3 * ((points[0].c[i] + points[1].c[i]) - (points[2].c[i] + points[3].c[i]));
		const glong ishift01  = (points[0].c[i] - points[1].c[i]) - gfp.shifts.istart;
		const glong ishift23  = (points[2].c[i] - points[3].c[i]) - gfp.shifts.istart;
		assert(0 <= ishift01 && ishift01 < gfp.shifts.num);
		assert(0 <= ishift23 && ishift23 < gfp.shifts.num);
		const glong ishift0123 = ishift01 * gfp.shifts.num + ishift23;
		products[i] = gfp.products[ishift0123];
		istart.c[i] = gfp.indices [ishift0123].istart;
		num.c[i]    = gfp.indices [ishift0123].num;
	}

	const glong num_circular_products = lmax(lsquare(center_diff_6[0] + istart.x), lsquare(center_diff_6[0] + istart.x + num.x - 1))
	                                  + lmax(lsquare(center_diff_6[1] + istart.y), lsquare(center_diff_6[1] + istart.y + num.y - 1)) + 1;
	double* circular_products = aligned_calloc(num_circular_products * sizeof(circular_products[0]));
	for (glong sx = 0; sx < num.x; ++sx)
	{
		if (fabs(products[0][sx]) <= tol) {
			continue;
		}

		const glong sq_x = lsquare(center_diff_6[0] + (istart.x + sx));

		for (glong sy = 0; sy < num.y; ++sy)
		{
			if (fabs(products[1][sy]) <= tol) {
				continue;
			}

			const glong sq_y = lsquare(center_diff_6[1] + (istart.y + sy));

			assert(sq_x + sq_y < num_circular_products);
			circular_products[sq_x + sq_y] += products[0][sx] * products[1][sy];
		}
	}

	double val = 0;
	for (glong rho = 0; rho < num_circular_products; ++rho)
	{
		if (circular_products[rho] == 0) {
			continue;
		}

		double t = 0;
		for (glong sz = 0; sz < num.z; ++sz)
		{
			if (fabs(products[2][sz]) <= tol) {
				continue;
			}

			const glong sq_z = lsquare(center_diff_6[2] + (istart.z + sz));

			t += products[2][sz] * coulomb_integral_table[rho + sq_z];
		}
		t *= circular_products[rho];

		val += t;
	}

	aligned_free(circular_products);
	aligned_free(coulomb_integral_table);
	delete_eri_gausslet_factor_products(&gfp);

	return val;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the minimal combined 4-index appearing within the octahedral group orbit
/// and i <-> j, k <-> l, (i, j) <-> (k, l) permutation symmetry.
///
glong minimum_octahedral_orbit_eri_tensor_index(
	const glong num_points, const glong* octahedral_perm[48],
	const glong i, const glong j, const glong k, const glong l)
{
	assert(0 <= i && i < num_points);
	assert(0 <= j && j < num_points);
	assert(0 <= k && k < num_points);
	assert(0 <= l && l < num_points);

	glong min_index = GLONG_MAX;

	for (int n = 0; n < 48; ++n)
	{
		glong a = octahedral_perm[n][i];
		glong b = octahedral_perm[n][j];
		glong c = octahedral_perm[n][k];
		glong d = octahedral_perm[n][l];

		// permutation symmetries

		if (a > b)
		{
			// swap
			glong tmp = a;
			a = b;
			b = tmp;
		}

		if (c > d)
		{
			// swap
			glong tmp = c;
			c = d;
			d = tmp;
		}

		glong ab = a * num_points + b;
		glong cd = c * num_points + d;

		if (ab > cd)
		{
			// swap
			glong tmp = ab;
			ab = cd;
			cd = tmp;
		}

		min_index = lmin(min_index, ab * (num_points * num_points) + cd);
	}

	return min_index;
}


//________________________________________________________________________________________________________________________
///
/// \brief Test whether the provided 'index' is smaller than or equal to
/// the minimal combined 4-index appearing within the octahedral group orbit
/// and i <-> j, k <-> l, (i, j) <-> (k, l) permutation symmetry.
///
static inline bool is_lower_bound_octahedral_orbit_eri_tensor_index(
	const glong num_points, const glong* octahedral_perm[48],
	const glong i, const glong j, const glong k, const glong l,
	const glong index)
{
	assert(0 <= i && i < num_points);
	assert(0 <= j && j < num_points);
	assert(0 <= k && k < num_points);
	assert(0 <= l && l < num_points);

	for (int n = 0; n < 48; ++n)
	{
		glong a = octahedral_perm[n][i];
		glong b = octahedral_perm[n][j];
		glong c = octahedral_perm[n][k];
		glong d = octahedral_perm[n][l];

		// permutation symmetries

		if (a > b)
		{
			// swap
			glong tmp = a;
			a = b;
			b = tmp;
		}

		if (c > d)
		{
			// swap
			glong tmp = c;
			c = d;
			d = tmp;
		}

		glong ab = a * num_points + b;
		glong cd = c * num_points + d;

		if (ab > cd)
		{
			// swap
			glong tmp = ab;
			ab = cd;
			cd = tmp;
		}

		if (ab * (num_points * num_points) + cd < index) {
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
/// \brief Enumerate the linearized electron repulsion integral (ERI) indices (ij|kl) after symmetry reduction,
/// using translational invariance by only retaining integrals
/// with the center of the enclosing box of the orbital grid points at the origin,
/// and exploiting octahedral point group symmetry and i <-> j, k <-> l, (i, j) <-> (k, l) permutation symmetry.
///
void enumerate_symmetry_reduced_eri_indices(const struct cartesian_grid_3d* grid, struct sparse_eri_indices* ret)
{
	// copy grid information
	ret->grid = *grid;

	glong* octahedral_perm[48];
	evaluate_octahedral_grid_permutations(grid, octahedral_perm);

	const glong num_points = cartesian_grid_3d_num_points(grid);

	struct index_list index_list = { 0 };

	#pragma omp parallel for schedule(dynamic) collapse(4)
	for (glong icx = 0; icx < grid->coord_range[0].num; ++icx)
	{
		for (glong jcx = 0; jcx < grid->coord_range[0].num; ++jcx)
		{
			for (glong kcx = 0; kcx < grid->coord_range[0].num; ++kcx)
			{
				for (glong lcx = 0; lcx < grid->coord_range[0].num; ++lcx)
				{
					// deferring these checks to allow for OpenMP "collapse" of four nested loops
					// order i <= j implies icx <= jcx
					if (icx > jcx) {
						continue;
					}
					// order k <= l implies kcx <= lcx
					if (kcx > lcx) {
						continue;
					}
					// order (i, j) <= (k, l) implies icx <= kcx
					if (icx > kcx) {
						continue;
					}

					// exploit translational invariance
					// x-coordinate of orbital box center times 2
					const glong center_x = 2 * grid->coord_range[0].istart + lmin(lmin(lmin(icx, jcx), kcx), lcx) +
					                                                         lmax(lmax(lmax(icx, jcx), kcx), lcx);
					if (center_x < -1 || 1 < center_x) {
						continue;
					}

					for (glong icy = 0; icy < grid->coord_range[1].num; ++icy)
					{
						for (glong jcy = 0; jcy < grid->coord_range[1].num; ++jcy)
						{
							for (glong kcy = 0; kcy < grid->coord_range[1].num; ++kcy)
							{
								for (glong lcy = 0; lcy < grid->coord_range[1].num; ++lcy)
								{
									// exploit translational invariance
									// y-coordinate of orbital box center times 2
									const glong center_y = 2 * grid->coord_range[1].istart + lmin(lmin(lmin(icy, jcy), kcy), lcy) +
									                                                         lmax(lmax(lmax(icy, jcy), kcy), lcy);
									if (center_y < -1 || 1 < center_y) {
										continue;
									}

									for (glong icz = 0; icz < grid->coord_range[2].num; ++icz)
									{
										for (glong jcz = 0; jcz < grid->coord_range[2].num; ++jcz)
										{
											for (glong kcz = 0; kcz < grid->coord_range[2].num; ++kcz)
											{
												for (glong lcz = 0; lcz < grid->coord_range[2].num; ++lcz)
												{
													// exploit translational invariance
													// z-coordinate of orbital box center times 2
													const glong center_z = 2 * grid->coord_range[2].istart + lmin(lmin(lmin(icz, jcz), kcz), lcz) +
													                                                         lmax(lmax(lmax(icz, jcz), kcz), lcz);
													if (center_z < -1 || 1 < center_z) {
														continue;
													}

													const glong idx_i = cartesian_grid_3d_cartesian_to_linear_index(grid, icx, icy, icz);
													const glong idx_j = cartesian_grid_3d_cartesian_to_linear_index(grid, jcx, jcy, jcz);
													const glong idx_k = cartesian_grid_3d_cartesian_to_linear_index(grid, kcx, kcy, kcz);
													const glong idx_l = cartesian_grid_3d_cartesian_to_linear_index(grid, lcx, lcy, lcz);
													const glong idx_tensor = ((idx_i * num_points + idx_j) * num_points + idx_k) * num_points + idx_l;

													bool is_minimal_index = true;
													// bias shifts of bounding box center
													for (glong bx = 0; bx <= labs(center_x); ++bx)
													{
														for (glong by = 0; by <= labs(center_y); ++by)
														{
															for (glong bz = 0; bz <= labs(center_z); ++bz)
															{
																if (!is_minimal_index) {
																	continue;
																}

																const glong idx_biased_i = cartesian_grid_3d_cartesian_to_linear_index(grid, icx - bx*center_x, icy - by*center_y, icz - bz*center_z);
																const glong idx_biased_j = cartesian_grid_3d_cartesian_to_linear_index(grid, jcx - bx*center_x, jcy - by*center_y, jcz - bz*center_z);
																const glong idx_biased_k = cartesian_grid_3d_cartesian_to_linear_index(grid, kcx - bx*center_x, kcy - by*center_y, kcz - bz*center_z);
																const glong idx_biased_l = cartesian_grid_3d_cartesian_to_linear_index(grid, lcx - bx*center_x, lcy - by*center_y, lcz - bz*center_z);

																// use point group and permutation symmetries to avoid redundant calculations
																if (!is_lower_bound_octahedral_orbit_eri_tensor_index(num_points, (const glong**)octahedral_perm, idx_biased_i, idx_biased_j, idx_biased_k, idx_biased_l, idx_tensor))
																{
																	is_minimal_index = false;
																}
															}
														}
													}
													if (is_minimal_index)
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
	ret->four_indices = aligned_malloc(ret->num * sizeof(ret->four_indices[0]));
	index_list_to_array(&index_list, ret->four_indices);
	delete_index_list(&index_list);

	// sort entries
	qsort(ret->four_indices, ret->num, sizeof(ret->four_indices[0]), compare_indices);
	// indices must be unique
	#ifndef NDEBUG
	for (glong i = 0; i < ret->num - 1; ++i) {
		assert(ret->four_indices[i] < ret->four_indices[i + 1]);
	}
	#endif
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete electron repulsion integral indices structure (free memory).
///
void delete_sparse_eri_indices(struct sparse_eri_indices* indices)
{
	aligned_free(indices->four_indices);
	indices->num = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the electron repulsion integrals (ERIs) for Gausslet orbitals and the provided linearized indices (ij|kl).
///
void compute_sparse_eri_gausslet_integrals(const struct gausslet_data* gdata,
	const struct cartesian_grid_3d* grid, const glong* four_indices, const glong num_indices,
	const double tol, struct sparse_eri_gausslet_integrals* eri)
{
	// copy grid information
	eri->grid = *grid;

	const glong max_grid_range = lmax(lmax(
		grid->coord_range[0].num,
		grid->coord_range[1].num),
		grid->coord_range[2].num);
	assert(max_grid_range > 0);

	struct eri_gausslet_factor_products gfp;
	{
		// unique shifts
		const struct range shifts = {
			.istart = -max_grid_range + 1,
			.num    = 2 * max_grid_range - 1,
		};

		compute_eri_gausslet_factor_products(gdata, &shifts, tol, &gfp);
	}

	// look-up table for Coulomb integrals
	const glong num_table_entries = 3 * lsquare(6 * (max_grid_range - 1) + 2 * (gdata->indices.num - 1)) + 1;
	double* coulomb_integral_table = aligned_calloc(num_table_entries * sizeof(coulomb_integral_table[0]));
	for (glong i = 0; i < num_table_entries; ++i)
	{
		coulomb_integral_table[i] = (double)gaussian_coulomb_integral_3d(1.L / 3, sqrtl(i) / 6);
	}

	// copy indices
	eri->num_entries = num_indices;
	eri->four_indices = aligned_malloc(num_indices * sizeof(eri->four_indices[0]));
	memcpy(eri->four_indices, four_indices, num_indices * sizeof(eri->four_indices[0]));

	eri->integral_values = aligned_malloc(eri->num_entries * sizeof(eri->integral_values[0]));

	const glong num_points = cartesian_grid_3d_num_points(grid);

	#pragma omp parallel for
	for (glong n = 0; n < num_indices; ++n)
	{
		// decode (ij|kl) indices
		glong idx_i, idx_j, idx_k, idx_l;
		{
			glong rem = four_indices[n];
			idx_l = rem % num_points;
			rem  /= num_points;
			idx_k = rem % num_points;
			rem  /= num_points;
			idx_j = rem % num_points;
			rem  /= num_points;
			idx_i = rem;
		}

		// decode grid points
		union cartesian_grid_point_3d pt_i, pt_j, pt_k, pt_l;
		linear_index_to_cartesian_grid_point_3d(grid, idx_i, &pt_i);
		linear_index_to_cartesian_grid_point_3d(grid, idx_j, &pt_j);
		linear_index_to_cartesian_grid_point_3d(grid, idx_k, &pt_k);
		linear_index_to_cartesian_grid_point_3d(grid, idx_l, &pt_l);

		glong center_diff_6[3];
		const double* products[3];
		union cartesian_grid_point_3d istart, num;
		for (int i = 0; i < 3; ++i)
		{
			center_diff_6[i] = 3 * ((pt_i.c[i] + pt_j.c[i]) - (pt_k.c[i] + pt_l.c[i]));
			const glong ishift01  = (pt_i.c[i] - pt_j.c[i]) - gfp.shifts.istart;
			const glong ishift23  = (pt_k.c[i] - pt_l.c[i]) - gfp.shifts.istart;
			assert(0 <= ishift01 && ishift01 < gfp.shifts.num);
			assert(0 <= ishift23 && ishift23 < gfp.shifts.num);
			const glong ishift0123 = ishift01 * gfp.shifts.num + ishift23;
			products[i] = gfp.products[ishift0123];
			istart.c[i] = gfp.indices [ishift0123].istart;
			num.c[i]    = gfp.indices [ishift0123].num;
		}

		const glong num_circular_products = lmax(lsquare(center_diff_6[0] + istart.x), lsquare(center_diff_6[0] + istart.x + num.x - 1))
		                                  + lmax(lsquare(center_diff_6[1] + istart.y), lsquare(center_diff_6[1] + istart.y + num.y - 1)) + 1;
		double* circular_products = aligned_calloc(num_circular_products * sizeof(circular_products[0]));
		for (glong sx = 0; sx < num.x; ++sx)
		{
			if (fabs(products[0][sx]) <= tol) {
				continue;
			}

			const glong sq_x = lsquare(center_diff_6[0] + (istart.x + sx));

			for (glong sy = 0; sy < num.y; ++sy)
			{
				if (fabs(products[1][sy]) <= tol) {
					continue;
				}

				const glong sq_y = lsquare(center_diff_6[1] + (istart.y + sy));

				assert(sq_x + sq_y < num_circular_products);
				circular_products[sq_x + sq_y] += products[0][sx] * products[1][sy];
			}
		}

		double val = 0;
		for (glong rho = 0; rho < num_circular_products; ++rho)
		{
			if (circular_products[rho] == 0) {
				continue;
			}

			double t = 0;
			for (glong sz = 0; sz < num.z; ++sz)
			{
				if (fabs(products[2][sz]) <= tol) {
					continue;
				}

				const glong sq_z = lsquare(center_diff_6[2] + (istart.z + sz));

				t += products[2][sz] * coulomb_integral_table[rho + sq_z];
			}
			t *= circular_products[rho];

			val += t;
		}

		eri->integral_values[n] = val;

		aligned_free(circular_products);
	}

	aligned_free(coulomb_integral_table);
	delete_eri_gausslet_factor_products(&gfp);
}


//________________________________________________________________________________________________________________________
///
/// \brief Retrieve the value corresponding to 'index', or 0 if 'index' cannot be found.
///
double sparse_eri_gausslet_integrals_get_value(const struct sparse_eri_gausslet_integrals* eri, const glong index)
{
	// search interval: [lower, upper)
	glong lower = 0;
	glong upper = eri->num_entries;
	while (true)
	{
		if (lower >= upper) {
			return 0;  // 'index' not found
		}
		const glong i = (lower + upper) / 2;
		if (index < eri->four_indices[i]) {
			upper = i;
		}
		else if (index > eri->four_indices[i]) {
			lower = i + 1;
		}
		else {
			// found it
			return eri->integral_values[i];
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Project the electron repulsion integral (ERI) tensor onto the specified basis along each axis.
///
void project_sparse_eri_gausslet_integrals(const struct sparse_eri_gausslet_integrals* eri, const double* restrict basis, const glong num_states, double* restrict eri_proj)
{
	const glong num_points = cartesian_grid_3d_num_points(&eri->grid);
	assert(num_points > 0);

	glong* octahedral_perm[48];
	evaluate_octahedral_grid_permutations(&eri->grid, octahedral_perm);

	memset(eri_proj, 0, num_states * num_states * num_states * num_states * sizeof(eri_proj[0]));

	#pragma omp parallel for collapse(4)
	for (glong icx = 0; icx < eri->grid.coord_range[0].num; ++icx)
	{
		for (glong jcx = 0; jcx < eri->grid.coord_range[0].num; ++jcx)
		{
			for (glong kcx = 0; kcx < eri->grid.coord_range[0].num; ++kcx)
			{
				for (glong lcx = 0; lcx < eri->grid.coord_range[0].num; ++lcx)
				{
					// x-coordinate of orbital box center times 2
					const glong center_x = 2 * eri->grid.coord_range[0].istart + lmin(lmin(lmin(icx, jcx), kcx), lcx) +
					                                                             lmax(lmax(lmax(icx, jcx), kcx), lcx);
					const glong trans_x  = center_x / 2;
					const glong bias_x   = center_x - 2 * trans_x;
					assert(-1 <= bias_x && bias_x <= 1);

					for (glong icy = 0; icy < eri->grid.coord_range[1].num; ++icy)
					{
						for (glong jcy = 0; jcy < eri->grid.coord_range[1].num; ++jcy)
						{
							for (glong kcy = 0; kcy < eri->grid.coord_range[1].num; ++kcy)
							{
								for (glong lcy = 0; lcy < eri->grid.coord_range[1].num; ++lcy)
								{
									// y-coordinate of orbital box center times 2
									const glong center_y = 2 * eri->grid.coord_range[1].istart + lmin(lmin(lmin(icy, jcy), kcy), lcy) +
									                                                             lmax(lmax(lmax(icy, jcy), kcy), lcy);
									const glong trans_y  = center_y / 2;
									const glong bias_y   = center_y - 2 * trans_y;
									assert(-1 <= bias_y && bias_y <= 1);

									for (glong icz = 0; icz < eri->grid.coord_range[2].num; ++icz)
									{
										for (glong jcz = 0; jcz < eri->grid.coord_range[2].num; ++jcz)
										{
											for (glong kcz = 0; kcz < eri->grid.coord_range[2].num; ++kcz)
											{
												for (glong lcz = 0; lcz < eri->grid.coord_range[2].num; ++lcz)
												{
													// z-coordinate of orbital box center times 2
													const glong center_z = 2 * eri->grid.coord_range[2].istart + lmin(lmin(lmin(icz, jcz), kcz), lcz) +
													                                                             lmax(lmax(lmax(icz, jcz), kcz), lcz);
													const glong trans_z  = center_z / 2;
													const glong bias_z   = center_z - 2 * trans_z;
													assert(-1 <= bias_z && bias_z <= 1);

													const glong idx_i = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid, icx, icy, icz);
													const glong idx_j = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid, jcx, jcy, jcz);
													const glong idx_k = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid, kcx, kcy, kcz);
													const glong idx_l = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid, lcx, lcy, lcz);

													glong idx_tensor_sparse = GLONG_MAX;
													for (glong bx = 0; bx <= labs(bias_x); ++bx)
													{
														for (glong by = 0; by <= labs(bias_y); ++by)
														{
															for (glong bz = 0; bz <= labs(bias_z); ++bz)
															{
																const glong idx_i_p = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid,
																	icx - trans_x - bx*bias_x,
																	icy - trans_y - by*bias_y,
																	icz - trans_z - bz*bias_z);
																const glong idx_j_p = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid,
																	jcx - trans_x - bx*bias_x,
																	jcy - trans_y - by*bias_y,
																	jcz - trans_z - bz*bias_z);
																const glong idx_k_p = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid,
																	kcx - trans_x - bx*bias_x,
																	kcy - trans_y - by*bias_y,
																	kcz - trans_z - bz*bias_z);
																const glong idx_l_p = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid,
																	lcx - trans_x - bx*bias_x,
																	lcy - trans_y - by*bias_y,
																	lcz - trans_z - bz*bias_z);

																idx_tensor_sparse = lmin(idx_tensor_sparse, minimum_octahedral_orbit_eri_tensor_index(
																	num_points, (const glong**)octahedral_perm, idx_i_p, idx_j_p, idx_k_p, idx_l_p));
															}
														}
													}

													const double val = sparse_eri_gausslet_integrals_get_value(eri, idx_tensor_sparse);

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
																		* basis[idx_j * num_states + q]
																		* basis[idx_k * num_states + r]
																		* basis[idx_l * num_states + s]
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
/// \brief Fill all entries of the dense electron repulsion integral (ERI) tensor of degree four.
///
void fill_dense_eri_tensor(const struct sparse_eri_gausslet_integrals* eri_sparse, const struct cartesian_grid_3d* grid_dense, double* eri_tensor)
{
	const glong num_points_sparse = cartesian_grid_3d_num_points(&eri_sparse->grid);
	const glong num_points_dense  = cartesian_grid_3d_num_points(grid_dense);
	assert(num_points_sparse > 0);
	assert(num_points_dense  > 0);

	glong* octahedral_perm[48];
	evaluate_octahedral_grid_permutations(&eri_sparse->grid, octahedral_perm);

	#pragma omp parallel for collapse(4)
	for (glong icx = 0; icx < grid_dense->coord_range[0].num; ++icx)
	{
		for (glong jcx = 0; jcx < grid_dense->coord_range[0].num; ++jcx)
		{
			for (glong kcx = 0; kcx < grid_dense->coord_range[0].num; ++kcx)
			{
				for (glong lcx = 0; lcx < grid_dense->coord_range[0].num; ++lcx)
				{
					// x-coordinate of orbital box center times 2
					const glong center_x = 2 * grid_dense->coord_range[0].istart + lmin(lmin(lmin(icx, jcx), kcx), lcx) +
					                                                               lmax(lmax(lmax(icx, jcx), kcx), lcx);
					const glong trans_x  = center_x / 2;
					const glong bias_x   = center_x - 2 * trans_x;
					assert(-1 <= bias_x && bias_x <= 1);

					for (glong icy = 0; icy < grid_dense->coord_range[1].num; ++icy)
					{
						for (glong jcy = 0; jcy < grid_dense->coord_range[1].num; ++jcy)
						{
							for (glong kcy = 0; kcy < grid_dense->coord_range[1].num; ++kcy)
							{
								for (glong lcy = 0; lcy < grid_dense->coord_range[1].num; ++lcy)
								{
									// y-coordinate of orbital box center times 2
									const glong center_y = 2 * grid_dense->coord_range[1].istart + lmin(lmin(lmin(icy, jcy), kcy), lcy) +
									                                                               lmax(lmax(lmax(icy, jcy), kcy), lcy);
									const glong trans_y  = center_y / 2;
									const glong bias_y   = center_y - 2 * trans_y;
									assert(-1 <= bias_y && bias_y <= 1);

									for (glong icz = 0; icz < grid_dense->coord_range[2].num; ++icz)
									{
										for (glong jcz = 0; jcz < grid_dense->coord_range[2].num; ++jcz)
										{
											for (glong kcz = 0; kcz < grid_dense->coord_range[2].num; ++kcz)
											{
												for (glong lcz = 0; lcz < grid_dense->coord_range[2].num; ++lcz)
												{
													// z-coordinate of orbital box center times 2
													const glong center_z = 2 * grid_dense->coord_range[2].istart + lmin(lmin(lmin(icz, jcz), kcz), lcz) +
													                                                               lmax(lmax(lmax(icz, jcz), kcz), lcz);
													const glong trans_z  = center_z / 2;
													const glong bias_z   = center_z - 2 * trans_z;
													assert(-1 <= bias_z && bias_z <= 1);

													const glong idx_i = cartesian_grid_3d_cartesian_to_linear_index(grid_dense, icx, icy, icz);
													const glong idx_j = cartesian_grid_3d_cartesian_to_linear_index(grid_dense, jcx, jcy, jcz);
													const glong idx_k = cartesian_grid_3d_cartesian_to_linear_index(grid_dense, kcx, kcy, kcz);
													const glong idx_l = cartesian_grid_3d_cartesian_to_linear_index(grid_dense, lcx, lcy, lcz);
													const glong idx_tensor_dense  = ((idx_i * num_points_dense + idx_j) * num_points_dense + idx_k) * num_points_dense + idx_l;

													glong idx_tensor_sparse = GLONG_MAX;
													for (glong bx = 0; bx <= labs(bias_x); ++bx)
													{
														for (glong by = 0; by <= labs(bias_y); ++by)
														{
															for (glong bz = 0; bz <= labs(bias_z); ++bz)
															{
																const union cartesian_grid_point_3d pt_i_p = {
																	.x = grid_dense->coord_range[0].istart + icx - trans_x - bx*bias_x,
																	.y = grid_dense->coord_range[1].istart + icy - trans_y - by*bias_y,
																	.z = grid_dense->coord_range[2].istart + icz - trans_z - bz*bias_z,
																};
																const union cartesian_grid_point_3d pt_j_p = {
																	.x = grid_dense->coord_range[0].istart + jcx - trans_x - bx*bias_x,
																	.y = grid_dense->coord_range[1].istart + jcy - trans_y - by*bias_y,
																	.z = grid_dense->coord_range[2].istart + jcz - trans_z - bz*bias_z,
																};
																const union cartesian_grid_point_3d pt_k_p = {
																	.x = grid_dense->coord_range[0].istart + kcx - trans_x - bx*bias_x,
																	.y = grid_dense->coord_range[1].istart + kcy - trans_y - by*bias_y,
																	.z = grid_dense->coord_range[2].istart + kcz - trans_z - bz*bias_z,
																};
																const union cartesian_grid_point_3d pt_l_p = {
																	.x = grid_dense->coord_range[0].istart + lcx - trans_x - bx*bias_x,
																	.y = grid_dense->coord_range[1].istart + lcy - trans_y - by*bias_y,
																	.z = grid_dense->coord_range[2].istart + lcz - trans_z - bz*bias_z,
																};

																const glong idx_i_p = cartesian_grid_point_3d_to_linear_index(&eri_sparse->grid, &pt_i_p);
																const glong idx_j_p = cartesian_grid_point_3d_to_linear_index(&eri_sparse->grid, &pt_j_p);
																const glong idx_k_p = cartesian_grid_point_3d_to_linear_index(&eri_sparse->grid, &pt_k_p);
																const glong idx_l_p = cartesian_grid_point_3d_to_linear_index(&eri_sparse->grid, &pt_l_p);

																idx_tensor_sparse = lmin(idx_tensor_sparse, minimum_octahedral_orbit_eri_tensor_index(num_points_sparse, (const glong**)octahedral_perm, idx_i_p, idx_j_p, idx_k_p, idx_l_p));
															}
														}
													}

													eri_tensor[idx_tensor_dense] = sparse_eri_gausslet_integrals_get_value(eri_sparse, idx_tensor_sparse);
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
		}
	}

	for (int i = 0; i < 48; ++i) {
		aligned_free(octahedral_perm[i]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply the electron repulsion integral (ERI) tensor interpreted as a matrix in physicist's convention to a list of states.
///
void apply_sparse_eri_tensor(const struct sparse_eri_gausslet_integrals* eri, const double* restrict states, const glong num_states, double* restrict ret)
{
	const glong num_points = cartesian_grid_3d_num_points(&eri->grid);
	assert(num_points > 0);

	glong* octahedral_perm[48];
	evaluate_octahedral_grid_permutations(&eri->grid, octahedral_perm);

	const glong dim_state = num_points * num_points;

	memset(ret, 0, dim_state * num_states * sizeof(ret[0]));

	// re-ordering j <-> k loops, and parallelizing only outer two i, k loops to avoid race conditions
	#pragma omp parallel for collapse(2)
	for (glong icx = 0; icx < eri->grid.coord_range[0].num; ++icx)
	{
		for (glong kcx = 0; kcx < eri->grid.coord_range[0].num; ++kcx)
		{
			for (glong jcx = 0; jcx < eri->grid.coord_range[0].num; ++jcx)
			{
				for (glong lcx = 0; lcx < eri->grid.coord_range[0].num; ++lcx)
				{
					// x-coordinate of orbital box center times 2
					const glong center_x = 2 * eri->grid.coord_range[0].istart + lmin(lmin(lmin(icx, jcx), kcx), lcx) +
					                                                             lmax(lmax(lmax(icx, jcx), kcx), lcx);
					const glong trans_x  = center_x / 2;
					const glong bias_x   = center_x - 2 * trans_x;
					assert(-1 <= bias_x && bias_x <= 1);

					for (glong icy = 0; icy < eri->grid.coord_range[1].num; ++icy)
					{
						for (glong kcy = 0; kcy < eri->grid.coord_range[1].num; ++kcy)
						{
							for (glong jcy = 0; jcy < eri->grid.coord_range[1].num; ++jcy)
							{
								for (glong lcy = 0; lcy < eri->grid.coord_range[1].num; ++lcy)
								{
									// y-coordinate of orbital box center times 2
									const glong center_y = 2 * eri->grid.coord_range[1].istart + lmin(lmin(lmin(icy, jcy), kcy), lcy) +
									                                                             lmax(lmax(lmax(icy, jcy), kcy), lcy);
									const glong trans_y  = center_y / 2;
									const glong bias_y   = center_y - 2 * trans_y;
									assert(-1 <= bias_y && bias_y <= 1);

									for (glong icz = 0; icz < eri->grid.coord_range[2].num; ++icz)
									{
										for (glong kcz = 0; kcz < eri->grid.coord_range[2].num; ++kcz)
										{
											for (glong jcz = 0; jcz < eri->grid.coord_range[2].num; ++jcz)
											{
												for (glong lcz = 0; lcz < eri->grid.coord_range[2].num; ++lcz)
												{
													// z-coordinate of orbital box center times 2
													const glong center_z = 2 * eri->grid.coord_range[2].istart + lmin(lmin(lmin(icz, jcz), kcz), lcz) +
													                                                             lmax(lmax(lmax(icz, jcz), kcz), lcz);
													const glong trans_z  = center_z / 2;
													const glong bias_z   = center_z - 2 * trans_z;
													assert(-1 <= bias_z && bias_z <= 1);

													const glong idx_i = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid, icx, icy, icz);
													const glong idx_j = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid, jcx, jcy, jcz);
													const glong idx_k = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid, kcx, kcy, kcz);
													const glong idx_l = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid, lcx, lcy, lcz);
													// group (i, k) and (j, l) in accordance with physicist's convention
													const glong idx_ik = idx_i * num_points + idx_k;
													const glong idx_jl = idx_j * num_points + idx_l;

													glong idx_tensor_sparse = GLONG_MAX;
													for (glong bx = 0; bx <= labs(bias_x); ++bx)
													{
														for (glong by = 0; by <= labs(bias_y); ++by)
														{
															for (glong bz = 0; bz <= labs(bias_z); ++bz)
															{
																const union cartesian_grid_point_3d pt_i_p = {
																	.x = eri->grid.coord_range[0].istart + icx - trans_x - bx*bias_x,
																	.y = eri->grid.coord_range[1].istart + icy - trans_y - by*bias_y,
																	.z = eri->grid.coord_range[2].istart + icz - trans_z - bz*bias_z,
																};
																const union cartesian_grid_point_3d pt_j_p = {
																	.x = eri->grid.coord_range[0].istart + jcx - trans_x - bx*bias_x,
																	.y = eri->grid.coord_range[1].istart + jcy - trans_y - by*bias_y,
																	.z = eri->grid.coord_range[2].istart + jcz - trans_z - bz*bias_z,
																};
																const union cartesian_grid_point_3d pt_k_p = {
																	.x = eri->grid.coord_range[0].istart + kcx - trans_x - bx*bias_x,
																	.y = eri->grid.coord_range[1].istart + kcy - trans_y - by*bias_y,
																	.z = eri->grid.coord_range[2].istart + kcz - trans_z - bz*bias_z,
																};
																const union cartesian_grid_point_3d pt_l_p = {
																	.x = eri->grid.coord_range[0].istart + lcx - trans_x - bx*bias_x,
																	.y = eri->grid.coord_range[1].istart + lcy - trans_y - by*bias_y,
																	.z = eri->grid.coord_range[2].istart + lcz - trans_z - bz*bias_z,
																};
																const glong idx_i_p = cartesian_grid_point_3d_to_linear_index(&eri->grid, &pt_i_p);
																const glong idx_j_p = cartesian_grid_point_3d_to_linear_index(&eri->grid, &pt_j_p);
																const glong idx_k_p = cartesian_grid_point_3d_to_linear_index(&eri->grid, &pt_k_p);
																const glong idx_l_p = cartesian_grid_point_3d_to_linear_index(&eri->grid, &pt_l_p);

																idx_tensor_sparse = lmin(idx_tensor_sparse, minimum_octahedral_orbit_eri_tensor_index(num_points, (const glong**)octahedral_perm, idx_i_p, idx_j_p, idx_k_p, idx_l_p));
															}
														}
													}

													const double val = sparse_eri_gausslet_integrals_get_value(eri, idx_tensor_sparse);

													for (glong s = 0; s < num_states; ++s)
													{
														ret[idx_ik * num_states + s] += val * states[idx_jl * num_states + s];
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
void delete_sparse_eri_gausslet_integrals(struct sparse_eri_gausslet_integrals* eri)
{
	aligned_free(eri->four_indices);
	aligned_free(eri->integral_values);
	eri->num_entries = 0;
}
