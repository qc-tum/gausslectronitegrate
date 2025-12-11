#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>
#include "eri_integrals.h"
#include "gausslet_factors.h"
#include "symmetry.h"
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
	double** products;      //!< products[i * shifts.num + j] stores the products for shift index pair '(i, j)'
	struct range* indices;  //!< logical indices, for each shift pair
	struct range shifts;    //!< shifts
};


//________________________________________________________________________________________________________________________
///
/// \brief Pre-compute products of Gausslet factors for evaluating electron repulsion integrals.
///
static void compute_gausslet_factor_products(const struct gausslet_data* gdata, const struct range* shifts, const double tol, struct gausslet_factor_products* gfp)
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
static void delete_gausslet_factor_products(struct gausslet_factor_products* gfp)
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
	const double prefac = cubic_power(M_PI / 9.);

	struct gausslet_factor_products gfp;
	{
		const glong max_range_01 = lmax(lmax(
			labs(points[0].x - points[1].x),
			labs(points[0].y - points[1].y)),
			labs(points[0].z - points[1].z)) + 1;
		const glong max_range_23 = lmax(lmax(
			labs(points[2].x - points[3].x),
			labs(points[2].y - points[3].y)),
			labs(points[2].z - points[3].z)) + 1;
		const glong max_range = lmax(max_range_01, max_range_23);

		// unique shifts
		const struct range shifts = {
			.istart = -max_range + 1,
			.num    = 2 * max_range - 1,
		};

		compute_gausslet_factor_products(gdata, &shifts, tol, &gfp);
	}

	double center_diff[3];
	const double* products[3];
	union cartesian_grid_point_3d istart, num;
	for (int i = 0; i < 3; ++i)
	{
		center_diff[i] = 0.5 * ((points[0].c[i] + points[1].c[i]) - (points[2].c[i] + points[3].c[i]));
		const glong ishift01  = (points[0].c[i] - points[1].c[i]) - gfp.shifts.istart;
		const glong ishift23  = (points[2].c[i] - points[3].c[i]) - gfp.shifts.istart;
		assert(0 <= ishift01 && ishift01 < gfp.shifts.num);
		assert(0 <= ishift23 && ishift23 < gfp.shifts.num);
		const glong ishift0123 = ishift01 * gfp.shifts.num + ishift23;
		products[i] = gfp.products[ishift0123];
		istart.c[i] = gfp.indices [ishift0123].istart;
		num.c[i]    = gfp.indices [ishift0123].num;
	}

	double val = 0;
	double pt[3];
	for (glong sx = 0; sx < num.x; ++sx)
	{
		if (fabs(products[0][sx]) <= tol) {
			continue;
		}

		pt[0] = center_diff[0] + (istart.x + sx) / 6.0;

		for (glong sy = 0; sy < num.y; ++sy)
		{
			if (fabs(products[1][sy]) <= tol) {
				continue;
			}

			pt[1] = center_diff[1] + (istart.y + sy) / 6.0;

			for (glong sz = 0; sz < num.z; ++sz)
			{
				if (fabs(products[2][sz]) <= tol) {
					continue;
				}

				pt[2] = center_diff[2] + (istart.z + sz) / 6.0;

				val += products[0][sx] * products[1][sy] * products[2][sz] * gaussian_coulomb_integral_3d(1. / 3,  vec3_norm(pt));
			}
		}
	}
	val *= prefac;

	delete_gausslet_factor_products(&gfp);

	return val;
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the 3D grid point permutations effected by the octahedral point group elements.
///
static void evaluate_octahedral_grid_permutations(const struct cartesian_grid_3d* grid, glong* perm[48])
{
	const glong num_points = cartesian_grid_3d_num_points(grid);

	for (int i = 0; i < 48; ++i)
	{
		perm[i] = aligned_malloc(num_points * sizeof(perm[i][0]));
		compute_cartesian_grid_3d_permutation(grid, &octahedral_matrep[i], perm[i]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the minimal combined 4-index appearing within the octahedral group orbit
/// and i <-> j, k <-> l, (i, j) <-> (k, l) permutation symmetry.
///
static inline glong minimum_octahedral_orbit_eri_tensor_index(
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
static inline bool is_minimum_octahedral_orbit_eri_tensor_index(
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
/// \brief Evaluate the electron repulsion integrals (ERIs) for Gausslet orbitals,
/// exploiting translational invariance by only computing integrals with
/// the center of the enclosing box of the orbital grid points at the origin.
///
void compute_sparse_eri_gausslet_integrals(
	const struct gausslet_data* gdata, const struct cartesian_grid_3d* grid,
	const double tol, struct eri_gausslet_integrals* eri)
{
	// copy grid information
	eri->grid = *grid;

	const double prefac = cubic_power(M_PI / 9.);

	struct gausslet_factor_products gfp;
	{
		const glong max_range = lmax(lmax(
			grid->coord_range[0].num,
			grid->coord_range[1].num),
			grid->coord_range[2].num);
		assert(max_range > 0);

		// unique shifts
		const struct range shifts = {
			.istart = -max_range + 1,
			.num    = 2 * max_range - 1,
		};

		compute_gausslet_factor_products(gdata, &shifts, tol, &gfp);
	}

	glong* octahedral_perm[48];
	evaluate_octahedral_grid_permutations(grid, octahedral_perm);

	const glong num_points = cartesian_grid_3d_num_points(grid);
	eri->integral_values = aligned_calloc(num_points * num_points * num_points * num_points * sizeof(eri->integral_values[0]));

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
					{
						// x-coordinate of orbital box center times 2
						const glong center_x = 2 * grid->coord_range[0].istart + lmin(lmin(lmin(icx, jcx), kcx), lcx) +
						                                                         lmax(lmax(lmax(icx, jcx), kcx), lcx);
						if (center_x < -1 || 1 < center_x) {
							continue;
						}
					}

					const double center_diff_x = 0.5 * ((icx + jcx) - (kcx + lcx));
					const double* products_x;
					glong istart_x, num_x;
					{
						const glong ishift01 = (icx - jcx) - gfp.shifts.istart;
						const glong ishift23 = (kcx - lcx) - gfp.shifts.istart;
						assert(0 <= ishift01 && ishift01 < gfp.shifts.num);
						assert(0 <= ishift23 && ishift23 < gfp.shifts.num);
						const glong ishift0123 = ishift01 * gfp.shifts.num + ishift23;
						products_x = gfp.products[ishift0123];
						istart_x   = gfp.indices [ishift0123].istart;
						num_x      = gfp.indices [ishift0123].num;
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
									{
										// y-coordinate of orbital box center times 2
										const glong center_y = 2 * grid->coord_range[1].istart + lmin(lmin(lmin(icy, jcy), kcy), lcy) +
										                                                         lmax(lmax(lmax(icy, jcy), kcy), lcy);
										if (center_y < -1 || 1 < center_y) {
											continue;
										}
									}

									const double center_diff_y = 0.5 * ((icy + jcy) - (kcy + lcy));
									const double* products_y;
									glong istart_y, num_y;
									{
										const glong ishift01 = (icy - jcy) - gfp.shifts.istart;
										const glong ishift23 = (kcy - lcy) - gfp.shifts.istart;
										assert(0 <= ishift01 && ishift01 < gfp.shifts.num);
										assert(0 <= ishift23 && ishift23 < gfp.shifts.num);
										const glong ishift0123 = ishift01 * gfp.shifts.num + ishift23;
										products_y = gfp.products[ishift0123];
										istart_y   = gfp.indices [ishift0123].istart;
										num_y      = gfp.indices [ishift0123].num;
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
													{
														// z-coordinate of orbital box center times 2
														const glong center_z = 2 * grid->coord_range[2].istart + lmin(lmin(lmin(icz, jcz), kcz), lcz) +
														                                                         lmax(lmax(lmax(icz, jcz), kcz), lcz);
														if (center_z < -1 || 1 < center_z) {
															continue;
														}
													}

													const glong idx_i = cartesian_grid_3d_cartesian_to_linear_index(grid, icx, icy, icz);
													const glong idx_j = cartesian_grid_3d_cartesian_to_linear_index(grid, jcx, jcy, jcz);
													const glong idx_k = cartesian_grid_3d_cartesian_to_linear_index(grid, kcx, kcy, kcz);
													const glong idx_l = cartesian_grid_3d_cartesian_to_linear_index(grid, lcx, lcy, lcz);

													const glong idx_tensor = ((idx_i * num_points + idx_j) * num_points + idx_k) * num_points + idx_l;

													// use point group and permutation symmetries to avoid redundant calculations
													if (!is_minimum_octahedral_orbit_eri_tensor_index(num_points, (const glong**)octahedral_perm, idx_i, idx_j, idx_k, idx_l, idx_tensor)) {
														continue;
													}

													const double center_diff_z = 0.5 * ((icz + jcz) - (kcz + lcz));
													const double* products_z;
													glong istart_z, num_z;
													{
														const glong ishift01 = (icz - jcz) - gfp.shifts.istart;
														const glong ishift23 = (kcz - lcz) - gfp.shifts.istart;
														assert(0 <= ishift01 && ishift01 < gfp.shifts.num);
														assert(0 <= ishift23 && ishift23 < gfp.shifts.num);
														const glong ishift0123 = ishift01 * gfp.shifts.num + ishift23;
														products_z = gfp.products[ishift0123];
														istart_z   = gfp.indices [ishift0123].istart;
														num_z      = gfp.indices [ishift0123].num;
													}

													double val = 0;
													double pt[3];
													for (glong sx = 0; sx < num_x; ++sx)
													{
														if (fabs(products_x[sx]) <= tol) {
															continue;
														}

														pt[0] = center_diff_x + (istart_x + sx) / 6.0;

														for (glong sy = 0; sy < num_y; ++sy)
														{
															if (fabs(products_y[sy]) <= tol) {
																continue;
															}

															pt[1] = center_diff_y + (istart_y + sy) / 6.0;

															for (glong sz = 0; sz < num_z; ++sz)
															{
																if (fabs(products_z[sz]) <= tol) {
																	continue;
																}

																pt[2] = center_diff_z + (istart_z + sz) / 6.0;

																val += products_x[sx] * products_y[sy] * products_z[sz] * gaussian_coulomb_integral_3d(1. / 3,  vec3_norm(pt));
															}
														}
													}
													val *= prefac;

													eri->integral_values[idx_tensor] = val;
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
	delete_gausslet_factor_products(&gfp);
}


//________________________________________________________________________________________________________________________
///
/// \brief Fill all entries of the dense electron repulsion integral (ERI) tensor of degree four,
/// assuming that 'eri_dense' has been allocated already.
///
void fill_dense_eri_tensor(const struct eri_gausslet_integrals* restrict eri_sparse, struct eri_gausslet_integrals* restrict eri_dense)
{
	const glong num_points_sparse = cartesian_grid_3d_num_points(&eri_sparse->grid);
	const glong num_points_dense  = cartesian_grid_3d_num_points(&eri_dense->grid);
	assert(num_points_sparse > 0);
	assert(num_points_dense  > 0);

	glong* octahedral_perm[48];
	evaluate_octahedral_grid_permutations(&eri_sparse->grid, octahedral_perm);

	#pragma omp parallel for collapse(4)
	for (glong icx = 0; icx < eri_dense->grid.coord_range[0].num; ++icx)
	{
		for (glong jcx = 0; jcx < eri_dense->grid.coord_range[0].num; ++jcx)
		{
			for (glong kcx = 0; kcx < eri_dense->grid.coord_range[0].num; ++kcx)
			{
				for (glong lcx = 0; lcx < eri_dense->grid.coord_range[0].num; ++lcx)
				{
					// x-coordinate of orbital box center times 2
					const glong center_x = 2 * eri_dense->grid.coord_range[0].istart + lmin(lmin(lmin(icx, jcx), kcx), lcx) +
					                                                                   lmax(lmax(lmax(icx, jcx), kcx), lcx);
					const glong trans_x = center_x / 2;
					assert(-1 <= center_x - 2 * trans_x && center_x - 2 * trans_x <= 1);

					for (glong icy = 0; icy < eri_dense->grid.coord_range[1].num; ++icy)
					{
						for (glong jcy = 0; jcy < eri_dense->grid.coord_range[1].num; ++jcy)
						{
							for (glong kcy = 0; kcy < eri_dense->grid.coord_range[1].num; ++kcy)
							{
								for (glong lcy = 0; lcy < eri_dense->grid.coord_range[1].num; ++lcy)
								{
									// y-coordinate of orbital box center times 2
									const glong center_y = 2 * eri_dense->grid.coord_range[1].istart + lmin(lmin(lmin(icy, jcy), kcy), lcy) +
									                                                                   lmax(lmax(lmax(icy, jcy), kcy), lcy);
									const glong trans_y = center_y / 2;
									assert(-1 <= center_y - 2 * trans_y && center_y - 2 * trans_y <= 1);

									for (glong icz = 0; icz < eri_dense->grid.coord_range[2].num; ++icz)
									{
										for (glong jcz = 0; jcz < eri_dense->grid.coord_range[2].num; ++jcz)
										{
											for (glong kcz = 0; kcz < eri_dense->grid.coord_range[2].num; ++kcz)
											{
												for (glong lcz = 0; lcz < eri_dense->grid.coord_range[2].num; ++lcz)
												{
													// z-coordinate of orbital box center times 2
													const glong center_z = 2 * eri_dense->grid.coord_range[2].istart + lmin(lmin(lmin(icz, jcz), kcz), lcz) +
													                                                                   lmax(lmax(lmax(icz, jcz), kcz), lcz);
													const glong trans_z = center_z / 2;
													assert(-1 <= center_z - 2 * trans_z && center_z - 2 * trans_z <= 1);

													const glong idx_i = cartesian_grid_3d_cartesian_to_linear_index(&eri_dense->grid, icx, icy, icz);
													const glong idx_j = cartesian_grid_3d_cartesian_to_linear_index(&eri_dense->grid, jcx, jcy, jcz);
													const glong idx_k = cartesian_grid_3d_cartesian_to_linear_index(&eri_dense->grid, kcx, kcy, kcz);
													const glong idx_l = cartesian_grid_3d_cartesian_to_linear_index(&eri_dense->grid, lcx, lcy, lcz);
													const glong idx_tensor_dense  = ((idx_i * num_points_dense + idx_j) * num_points_dense + idx_k) * num_points_dense + idx_l;

													const union cartesian_grid_point_3d pt_i_p = {
														.x = eri_dense->grid.coord_range[0].istart + icx - trans_x,
														.y = eri_dense->grid.coord_range[1].istart + icy - trans_y,
														.z = eri_dense->grid.coord_range[2].istart + icz - trans_z,
													};
													const union cartesian_grid_point_3d pt_j_p = {
														.x = eri_dense->grid.coord_range[0].istart + jcx - trans_x,
														.y = eri_dense->grid.coord_range[1].istart + jcy - trans_y,
														.z = eri_dense->grid.coord_range[2].istart + jcz - trans_z,
													};
													const union cartesian_grid_point_3d pt_k_p = {
														.x = eri_dense->grid.coord_range[0].istart + kcx - trans_x,
														.y = eri_dense->grid.coord_range[1].istart + kcy - trans_y,
														.z = eri_dense->grid.coord_range[2].istart + kcz - trans_z,
													};
													const union cartesian_grid_point_3d pt_l_p = {
														.x = eri_dense->grid.coord_range[0].istart + lcx - trans_x,
														.y = eri_dense->grid.coord_range[1].istart + lcy - trans_y,
														.z = eri_dense->grid.coord_range[2].istart + lcz - trans_z,
													};
													const glong idx_i_p = cartesian_grid_point_3d_to_linear_index(&eri_sparse->grid, &pt_i_p);
													const glong idx_j_p = cartesian_grid_point_3d_to_linear_index(&eri_sparse->grid, &pt_j_p);
													const glong idx_k_p = cartesian_grid_point_3d_to_linear_index(&eri_sparse->grid, &pt_k_p);
													const glong idx_l_p = cartesian_grid_point_3d_to_linear_index(&eri_sparse->grid, &pt_l_p);
													const glong idx_tensor_sparse = minimum_octahedral_orbit_eri_tensor_index(num_points_sparse, (const glong**)octahedral_perm, idx_i_p, idx_j_p, idx_k_p, idx_l_p);

													eri_dense->integral_values[idx_tensor_dense] = eri_sparse->integral_values[idx_tensor_sparse];
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
/// \brief Delete electron repulsion integral storage structure (free memory).
///
void delete_eri_gausslet_integrals(struct eri_gausslet_integrals* eri)
{
	aligned_free(eri->integral_values);
}
