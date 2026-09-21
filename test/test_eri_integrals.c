#define _USE_MATH_DEFINES
#include <math.h>
#include "eri_integrals.h"
#include "aligned_memory.h"
#include "hdf5_util.h"
#include "util.h"


static void tensor_multiply_axis(const double* restrict s, const glong* dim_s, const int ndim_s, const int i_ax, const double* restrict t, const glong td_t, double* restrict r);


char* test_eri_integrals()
{
	hid_t file = H5Fopen("../test/data/test_eri_integrals.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_eri_integrals failed";
	}

	struct gausslet_data gdata;
	{
		hsize_t dim[1];
		if (get_hdf5_attribute_dims(file, "gausslet_coeffs", dim) < 0) {
			return "obtaining dimensions of Gausslet coefficients failed";
		}
		gdata.indices.num = dim[0];
		if (gdata.indices.num % 2 != 1) {
			return "expecting an odd number of Gausslet coefficients";
		}
		gdata.indices.istart = -(gdata.indices.num - 1) / 2;

		gdata.coefficients = aligned_malloc(gdata.indices.num * sizeof(gdata.coefficients[0]));
		if (read_hdf5_attribute(file, "gausslet_coeffs", H5T_NATIVE_DOUBLE, gdata.coefficients) < 0) {
			return "reading Gausslet coefficients from disk failed";
		}
	}

	// ensure that grid for sparse tensor is symmetrically centered around the origin
	const struct cartesian_grid_3d grid_sparse = {
		.coord_range = {
			{ .istart = -1, .num = 3 },
			{ .istart = -1, .num = 3 },
			{ .istart = -1, .num = 3 },
		}
	};

	double tol;
	if (read_hdf5_attribute(file, "tol", H5T_NATIVE_DOUBLE, &tol) < 0) {
		return "reading tolerance from disk failed";
	}

	struct sparse_eri_indices eri_indices;
	enumerate_symmetry_reduced_eri_indices(&grid_sparse, &eri_indices);

	struct sparse_eri_gausslet_integrals eri_sparse;
	compute_sparse_eri_gausslet_integrals(&gdata, &grid_sparse, eri_indices.four_indices, eri_indices.num, tol, &eri_sparse);
	delete_sparse_eri_indices(&eri_indices);

	// reconstruct full tensor on another grid
	{
		const struct cartesian_grid_3d grid_dense = {
			.coord_range = {
				{ .istart = -1, .num = 2 },
				{ .istart =  0, .num = 2 },
				{ .istart = -1, .num = 3 },
			}
		};

		// reconstruct full tensor
		const glong num_points_dense = cartesian_grid_3d_num_points(&grid_dense);
		const glong num_entries_dense = num_points_dense * num_points_dense * num_points_dense * num_points_dense;
		double* eri_tensor = aligned_malloc(num_entries_dense * sizeof(eri_tensor[0]));
		fill_dense_eri_tensor(&eri_sparse, &grid_dense, eri_tensor);

		// reference data
		double* eri_tensor_ref = aligned_malloc(num_entries_dense * sizeof(eri_tensor_ref[0]));
		if (read_hdf5_dataset(file, "eri", H5T_NATIVE_DOUBLE, eri_tensor_ref) < 0) {
			return "reading electron repulsion integral values from disk failed";
		}

		// compare
		if (uniform_distance(num_entries_dense, eri_tensor, eri_tensor_ref) > 1e-13) {
			return "electron repulsion integral values do not match reference";
		}

		aligned_free(eri_tensor_ref);
		aligned_free(eri_tensor);
	}

	const glong num_points_sparse = cartesian_grid_3d_num_points(&grid_sparse);

	// reconstruct full tensor
	const glong num_entries_full = num_points_sparse * num_points_sparse * num_points_sparse * num_points_sparse;
	double* eri_tensor_full = aligned_malloc(num_entries_full * sizeof(eri_tensor_full[0]));
	fill_dense_eri_tensor(&eri_sparse, &grid_sparse, eri_tensor_full);

	// project ERI integrals
	{
		// define basis
		const glong num_states = 7;
		double* basis = aligned_malloc(num_points_sparse * num_states * sizeof(basis[0]));
		// fill basis states with pseudo-random entries
		for (glong i = 0; i < num_points_sparse * num_states; ++i) {
			basis[i] = 0.1 * sin(sqrt(3) * M_PI * ((557 * i) % 311) - 8) / (1.5 + cos(sqrt(5) * M_PI * ((541 * i) % 229)));
		}

		const glong num_entries_proj = num_states * num_states * num_states * num_states;
		double* eri_proj = aligned_malloc(num_entries_proj * sizeof(eri_proj[0]));
		project_sparse_eri_gausslet_integrals(&eri_sparse, basis, num_states, eri_proj);

		// reference calculation
		double* eri_proj_ref;
		{
			double* tmp = eri_tensor_full;
			glong dim_tmp[4] = { num_points_sparse, num_points_sparse, num_points_sparse, num_points_sparse };
			for (int i_ax = 0; i_ax < 4; ++i_ax)
			{
				glong n = 1;
				for (int i = 0; i < i_ax + 1; ++i) {
					n *= num_states;
				}
				for (int i = i_ax + 1; i < 4; ++i) {
					n *= num_points_sparse;
				}
				double* tmp_next = aligned_malloc(n * sizeof(tmp_next[0]));
				tensor_multiply_axis(tmp, dim_tmp, 4, i_ax, basis, num_states, tmp_next);
				if (i_ax > 0) {
					aligned_free(tmp);
				}
				tmp = tmp_next;
				dim_tmp[i_ax] = num_states;
			}

			eri_proj_ref = tmp;
		}

		// compare
		if (uniform_distance(num_entries_proj, eri_proj, eri_proj_ref) > 1e-12) {
			return "projected electron repulsion integral values do not match reference";
		}

		aligned_free(eri_proj_ref);
		aligned_free(eri_proj);
		aligned_free(basis);
	}

	// apply ERI tensor to two-particle states
	{
		const glong dim_state = num_points_sparse * num_points_sparse;

		// define states
		const glong num_states = 3;
		double* states = aligned_malloc(dim_state * num_states * sizeof(states[0]));
		// fill states with pseudo-random entries
		for (glong i = 0; i < dim_state * num_states; ++i) {
			states[i] = 0.12 * sin(sqrt(7) * M_PI * ((523 * i) % 379) - 4) / (1.8 + cos(sqrt(11) * M_PI * ((563 * i) % 281)));
		}

		double* eri_states = aligned_malloc(dim_state * num_states * sizeof(eri_states[0]));
		apply_sparse_eri_tensor(&eri_sparse, states, num_states, eri_states);

		// reference calculation
		double* eri_states_ref = aligned_calloc(dim_state * num_states * sizeof(eri_states_ref[0]));
		for (glong i = 0; i < num_points_sparse; ++i)
		{
			for (glong j = 0; j < num_points_sparse; ++j)
			{
				for (glong k = 0; k < num_points_sparse; ++k)
				{
					for (glong l = 0; l < num_points_sparse; ++l)
					{
						for (glong s = 0; s < num_states; ++s)
						{
							eri_states_ref[(i*num_points_sparse + k)*num_states + s] +=
								eri_tensor_full[((i*num_points_sparse + j)*num_points_sparse + k)*num_points_sparse + l] * states[(j*num_points_sparse + l)*num_states + s];
						}
					}
				}
			}
		}

		// compare
		if (uniform_distance(dim_state * num_states, eri_states, eri_states_ref) > 1e-12) {
			return "quantum states resulting from applying an electron repulsion integral tensor do not match reference";
		}

		aligned_free(eri_states_ref);
		aligned_free(eri_states);
		aligned_free(states);
	}

	aligned_free(eri_tensor_full);
	delete_sparse_eri_gausslet_integrals(&eri_sparse);
	aligned_free(gdata.coefficients);

	H5Fclose(file);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Multiply the 'i_ax' axis of 's' with the leading axis of the matrix 't', preserving the overall dimension ordering of 's'.
///
static void tensor_multiply_axis(const double* restrict s, const glong* dim_s, const int ndim_s, const int i_ax, const double* restrict t, const glong td_t, double* restrict r)
{
	assert(0 <= i_ax && i_ax < ndim_s);

	// product of leading dimensions of 's'
	glong ld_s = 1;
	for (int i = 0; i < i_ax; ++i) {
		ld_s *= dim_s[i];
	}
	// product of trailing dimensions of 's'
	glong td_s = 1;
	for (int i = i_ax + 1; i < ndim_s; ++i) {
		td_s *= dim_s[i];
	}

	for (glong i = 0; i < ld_s; ++i)
	{
		for (glong j = 0; j < td_t; ++j)
		{
			for (glong k = 0; k < td_s; ++k)
			{
				double v = 0;
				for (glong l = 0; l < dim_s[i_ax]; ++l) {
					v += s[(i * dim_s[i_ax] + l) * td_s + k] * t[l * td_t + j];
				}
				r[(i * td_t + j) * td_s + k] = v;
			}
		}
	}
}
