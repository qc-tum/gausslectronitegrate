#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>
#include "eri_integrals.h"
#include "gausslet_factors.h"
#include "aligned_memory.h"
#include "util.h"


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

	for (long k = 0; k < shifts->num; ++k)
	{
		for (long l = 0; l < shifts->num; ++l)
		{
			const struct range all_indices = range_minus(gf.indices[k], gf.indices[l]);

			// convolution
			double* all_products = aligned_calloc(all_indices.num * sizeof(all_products[0]));
			for (long i = 0; i < gf.indices[k].num; ++i)
			{
				for (long j = 0; j < gf.indices[l].num; ++j)
				{
					const long idx = (gf.indices[k].istart + i) - (gf.indices[l].istart + j) - all_indices.istart;
					assert(0 <= idx && idx < all_indices.num);
					all_products[idx] += gf.factors[k][i] * gf.factors[l][j];
				}
			}

			// filter out small numbers
			long min_index = -1;
			for (long m = 0; m < all_indices.num; ++m) {
				if (fabs(all_products[m]) > tol) {
					min_index = m;
					break;
				}
			}
			long max_index = -1;
			for (long m = all_indices.num - 1; m >= 0; --m) {
				if (fabs(all_products[m]) > tol) {
					max_index = m;
					break;
				}
			}
			if (min_index != -1)
			{
				assert(max_index != -1);
				assert(min_index <= max_index);

				const long kl = k * shifts->num + l;
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
	for (long i = 0; i < gfp->shifts.num * gfp->shifts.num; ++i) {
		if (gfp->products[i] != NULL) {
			aligned_free(gfp->products[i]);
		}
	}
	aligned_free(gfp->products);
	aligned_free(gfp->indices);
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the electron repulsion integrals (ERIs) for Gausslet orbitals,
/// exploiting translational invariance by subtracting the last orbital center:
/// (i, j | k, l) -> (i - l, j - l | k - l, 0).
///
void compute_eri_gausslet_integrals(
	const struct gausslet_data* gdata, const struct cartesian_grid_3d* grid,
	const double tol, struct eri_gausslet_integrals* eri)
{
	// copy grid information
	eri->grid = *grid;

	// grid of translated coordinates after subtracting the last orbital center: (i, j | k, l) -> (i - l, j - l | k - l, 0)
	eri->grid_trans.coord_range[0] = range_minus(grid->coord_range[0], grid->coord_range[0]);
	eri->grid_trans.coord_range[1] = range_minus(grid->coord_range[1], grid->coord_range[1]);
	eri->grid_trans.coord_range[2] = range_minus(grid->coord_range[2], grid->coord_range[2]);

	const double prefac = cubic_power(M_PI / 9.);

	struct gausslet_factor_products gfp;
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

		compute_gausslet_factor_products(gdata, &shifts, tol, &gfp);
	}

	const long num_points_trans = cartesian_grid_3d_num_points(&eri->grid_trans);
	eri->integral_values = aligned_calloc(num_points_trans * num_points_trans * num_points_trans * sizeof(eri->integral_values[0]));

	// index corresponding to logical coordinate zero
	const long lcx = -eri->grid_trans.coord_range[0].istart;
	const long lcy = -eri->grid_trans.coord_range[1].istart;
	const long lcz = -eri->grid_trans.coord_range[2].istart;

	for (long icx = 0; icx < eri->grid_trans.coord_range[0].num; ++icx)
	{
		for (long jcx = 0; jcx < eri->grid_trans.coord_range[0].num; ++jcx)
		{
			if (labs(icx - jcx) > grid->coord_range[0].num - 1) {
				continue;
			}
			// use reflection symmetry to avoid redundant calculations
			for (long kcx = -eri->grid_trans.coord_range[0].istart; kcx < eri->grid_trans.coord_range[0].num; ++kcx)
			{
				if (labs(icx - kcx) > grid->coord_range[0].num - 1 || labs(jcx - kcx) > grid->coord_range[0].num - 1) {
					continue;
				}

				const double center_diff_x = 0.5 * ((icx + jcx) - (kcx + lcx));

				const long ishift_ij_x   = (icx - jcx) - gfp.shifts.istart;
				const long ishift_kl_x   = (kcx - lcx) - gfp.shifts.istart;
				assert(0 <= ishift_ij_x && ishift_ij_x < gfp.shifts.num);
				assert(0 <= ishift_kl_x && ishift_kl_x < gfp.shifts.num);
				const long ishift_ijkl_x = ishift_ij_x * gfp.shifts.num + ishift_kl_x;
				const double* products_x = gfp.products[ishift_ijkl_x];
				const long istart_x      = gfp.indices [ishift_ijkl_x].istart;
				const long num_x         = gfp.indices [ishift_ijkl_x].num;

				for (long icy = 0; icy < eri->grid_trans.coord_range[1].num; ++icy)
				{
					for (long jcy = 0; jcy < eri->grid_trans.coord_range[1].num; ++jcy)
					{
						if (labs(icy - jcy) > grid->coord_range[1].num - 1) {
							continue;
						}
						// use reflection symmetry to avoid redundant calculations
						for (long kcy = -eri->grid_trans.coord_range[1].istart; kcy < eri->grid_trans.coord_range[1].num; ++kcy)
						{
							if (labs(icy - kcy) > grid->coord_range[1].num - 1 || labs(jcy - kcy) > grid->coord_range[1].num - 1) {
								continue;
							}

							const double center_diff_y = 0.5 * ((icy + jcy) - (kcy + lcy));

							const long ishift_ij_y   = (icy - jcy) - gfp.shifts.istart;
							const long ishift_kl_y   = (kcy - lcy) - gfp.shifts.istart;
							assert(0 <= ishift_ij_y && ishift_ij_y < gfp.shifts.num);
							assert(0 <= ishift_kl_y && ishift_kl_y < gfp.shifts.num);
							const long ishift_ijkl_y = ishift_ij_y * gfp.shifts.num + ishift_kl_y;
							const double* products_y = gfp.products[ishift_ijkl_y];
							const long istart_y      = gfp.indices [ishift_ijkl_y].istart;
							const long num_y         = gfp.indices [ishift_ijkl_y].num;

							for (long icz = 0; icz < eri->grid_trans.coord_range[2].num; ++icz)
							{
								for (long jcz = 0; jcz < eri->grid_trans.coord_range[2].num; ++jcz)
								{
									if (labs(icz - jcz) > grid->coord_range[2].num - 1) {
										continue;
									}
									// use reflection symmetry to avoid redundant calculations
									for (long kcz = -eri->grid_trans.coord_range[2].istart; kcz < eri->grid_trans.coord_range[2].num; ++kcz)
									{
										if (labs(icz - kcz) > grid->coord_range[2].num - 1 || labs(jcz - kcz) > grid->coord_range[2].num - 1) {
											continue;
										}

										const double center_diff_z = 0.5 * ((icz + jcz) - (kcz + lcz));

										const long ishift_ij_z   = (icz - jcz) - gfp.shifts.istart;
										const long ishift_kl_z   = (kcz - lcz) - gfp.shifts.istart;
										assert(0 <= ishift_ij_z && ishift_ij_z < gfp.shifts.num);
										assert(0 <= ishift_kl_z && ishift_kl_z < gfp.shifts.num);
										const long ishift_ijkl_z = ishift_ij_z * gfp.shifts.num + ishift_kl_z;
										const double* products_z = gfp.products[ishift_ijkl_z];
										const long istart_z      = gfp.indices [ishift_ijkl_z].istart;
										const long num_z         = gfp.indices [ishift_ijkl_z].num;

										const long idx_i = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid_trans, icx, icy, icz);
										const long idx_j = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid_trans, jcx, jcy, jcz);
										// use permutation symmetry to avoid redundant calculations
										if (idx_i < idx_j) {
											continue;
										}
										const long idx_k = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid_trans, kcx, kcy, kcz);

										double val = 0;
										double pt[3];
										for (long sx = 0; sx < num_x; ++sx)
										{
											if (fabs(products_x[sx]) <= tol) {
												continue;
											}

											pt[0] = center_diff_x + (istart_x + sx) / 6.0;

											for (long sy = 0; sy < num_y; ++sy)
											{
												if (fabs(products_y[sy]) <= tol) {
													continue;
												}

												pt[1] = center_diff_y + (istart_y + sy) / 6.0;

												for (long sz = 0; sz < num_z; ++sz)
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

										eri->integral_values[(idx_i * num_points_trans + idx_j) * num_points_trans + idx_k] = val;
									}
								}
							}
						}
					}
				}
			}
		}
	}

	// fill omitted entries according to symmetries
	for (long icx = 0; icx < eri->grid_trans.coord_range[0].num; ++icx)
	{
		for (long jcx = 0; jcx < eri->grid_trans.coord_range[0].num; ++jcx)
		{
			if (labs(icx - jcx) > grid->coord_range[0].num - 1) {
				continue;
			}
			for (long kcx = 0; kcx < eri->grid_trans.coord_range[0].num; ++kcx)
			{
				if (labs(icx - kcx) > grid->coord_range[0].num - 1 || labs(jcx - kcx) > grid->coord_range[0].num - 1) {
					continue;
				}

				long icx_p, jcx_p, kcx_p;
				if (eri->grid_trans.coord_range[0].istart + kcx >= 0)
				{
					icx_p = icx;
					jcx_p = jcx;
					kcx_p = kcx;
				}
				else
				{
					// logical reflection
					icx_p = -icx - 2 * eri->grid_trans.coord_range[0].istart;
					jcx_p = -jcx - 2 * eri->grid_trans.coord_range[0].istart;
					kcx_p = -kcx - 2 * eri->grid_trans.coord_range[0].istart;
				}

				for (long icy = 0; icy < eri->grid_trans.coord_range[1].num; ++icy)
				{
					for (long jcy = 0; jcy < eri->grid_trans.coord_range[1].num; ++jcy)
					{
						if (labs(icy - jcy) > grid->coord_range[1].num - 1) {
							continue;
						}
						for (long kcy = 0; kcy < eri->grid_trans.coord_range[1].num; ++kcy)
						{
							if (labs(icy - kcy) > grid->coord_range[1].num - 1 || labs(jcy - kcy) > grid->coord_range[1].num - 1) {
								continue;
							}

							long icy_p, jcy_p, kcy_p;
							if (eri->grid_trans.coord_range[1].istart + kcy >= 0)
							{
								icy_p = icy;
								jcy_p = jcy;
								kcy_p = kcy;
							}
							else
							{
								// logical reflection
								icy_p = -icy - 2 * eri->grid_trans.coord_range[1].istart;
								jcy_p = -jcy - 2 * eri->grid_trans.coord_range[1].istart;
								kcy_p = -kcy - 2 * eri->grid_trans.coord_range[1].istart;
							}

							for (long icz = 0; icz < eri->grid_trans.coord_range[2].num; ++icz)
							{
								for (long jcz = 0; jcz < eri->grid_trans.coord_range[2].num; ++jcz)
								{
									if (labs(icz - jcz) > grid->coord_range[2].num - 1) {
										continue;
									}
									for (long kcz = 0; kcz < eri->grid_trans.coord_range[2].num; ++kcz)
									{
										if (labs(icz - kcz) > grid->coord_range[2].num - 1 || labs(jcz - kcz) > grid->coord_range[2].num - 1) {
											continue;
										}

										long icz_p, jcz_p, kcz_p;
										if (eri->grid_trans.coord_range[2].istart + kcz >= 0)
										{
											icz_p = icz;
											jcz_p = jcz;
											kcz_p = kcz;
										}
										else
										{
											// logical reflection
											icz_p = -icz - 2 * eri->grid_trans.coord_range[2].istart;
											jcz_p = -jcz - 2 * eri->grid_trans.coord_range[2].istart;
											kcz_p = -kcz - 2 * eri->grid_trans.coord_range[2].istart;
										}

										const long idx_i = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid_trans, icx, icy, icz);
										const long idx_j = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid_trans, jcx, jcy, jcz);
										const long idx_k = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid_trans, kcx, kcy, kcz);

										long idx_i_p = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid_trans, icx_p, icy_p, icz_p);
										long idx_j_p = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid_trans, jcx_p, jcy_p, jcz_p);
										long idx_k_p = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid_trans, kcx_p, kcy_p, kcz_p);

										if (idx_i_p < idx_j_p)
										{
											// swap according to permutation symmetry
											long tmp = idx_i_p;
											idx_i_p = idx_j_p;
											idx_j_p = tmp;
										}

										eri->integral_values[(idx_i * num_points_trans + idx_j) * num_points_trans + idx_k]
											= eri->integral_values[(idx_i_p * num_points_trans + idx_j_p) * num_points_trans + idx_k_p];
									}
								}
							}
						}
					}
				}
			}
		}
	}

	delete_gausslet_factor_products(&gfp);
}


//________________________________________________________________________________________________________________________
///
/// \brief Reconstruct the full electron repulsion integral (ERI) tensor of degree four (without using translational invariance).
///
void reconstruct_full_eri_tensor(const struct eri_gausslet_integrals* eri, double* full_tensor)
{
	const long num_points       = cartesian_grid_3d_num_points(&eri->grid);
	const long num_points_trans = cartesian_grid_3d_num_points(&eri->grid_trans);

	for (long icx = 0; icx < eri->grid.coord_range[0].num; ++icx)
	{
		for (long jcx = 0; jcx < eri->grid.coord_range[0].num; ++jcx)
		{
			for (long kcx = 0; kcx < eri->grid.coord_range[0].num; ++kcx)
			{
				for (long lcx = 0; lcx < eri->grid.coord_range[0].num; ++lcx)
				{

					for (long icy = 0; icy < eri->grid.coord_range[1].num; ++icy)
					{
						for (long jcy = 0; jcy < eri->grid.coord_range[1].num; ++jcy)
						{
							for (long kcy = 0; kcy < eri->grid.coord_range[1].num; ++kcy)
							{
								for (long lcy = 0; lcy < eri->grid.coord_range[1].num; ++lcy)
								{

									for (long icz = 0; icz < eri->grid.coord_range[2].num; ++icz)
									{
										for (long jcz = 0; jcz < eri->grid.coord_range[2].num; ++jcz)
										{
											for (long kcz = 0; kcz < eri->grid.coord_range[2].num; ++kcz)
											{
												for (long lcz = 0; lcz < eri->grid.coord_range[2].num; ++lcz)
												{

													const long idx_i = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid, icx, icy, icz);
													const long idx_j = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid, jcx, jcy, jcz);
													const long idx_k = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid, kcx, kcy, kcz);
													const long idx_l = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid, lcx, lcy, lcz);

													const long idx_il = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid_trans, (icx - lcx) - eri->grid_trans.coord_range[0].istart, (icy - lcy) - eri->grid_trans.coord_range[1].istart, (icz - lcz) - eri->grid_trans.coord_range[2].istart);
													const long idx_jl = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid_trans, (jcx - lcx) - eri->grid_trans.coord_range[0].istart, (jcy - lcy) - eri->grid_trans.coord_range[1].istart, (jcz - lcz) - eri->grid_trans.coord_range[2].istart);
													const long idx_kl = cartesian_grid_3d_cartesian_to_linear_index(&eri->grid_trans, (kcx - lcx) - eri->grid_trans.coord_range[0].istart, (kcy - lcy) - eri->grid_trans.coord_range[1].istart, (kcz - lcz) - eri->grid_trans.coord_range[2].istart);

													full_tensor[((idx_i * num_points + idx_j) * num_points + idx_k) * num_points + idx_l]
														= eri->integral_values[(idx_il * num_points_trans + idx_jl) * num_points_trans + idx_kl];
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


//________________________________________________________________________________________________________________________
///
/// \brief Delete electron repulsion integral storage structure (free memory).
///
void delete_eri_gausslet_integrals(struct eri_gausslet_integrals* eri)
{
	aligned_free(eri->integral_values);
}
