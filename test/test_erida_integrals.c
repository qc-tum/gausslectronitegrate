#define _USE_MATH_DEFINES
#include <math.h>
#include "erida_integrals.h"
#include "aligned_memory.h"
#include "hdf5_util.h"
#include "util.h"


char* test_erida_integrals()
{
	hid_t file = H5Fopen("../test/data/test_erida_integrals.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_erida_integrals failed";
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

	struct sparse_erida_indices erida_indices;
	enumerate_symmetry_reduced_erida_indices(&grid_sparse, &erida_indices);

	struct sparse_erida_gausslet_integrals erida_sparse;
	compute_sparse_erida_gausslet_integrals(&gdata, &grid_sparse, erida_indices.two_indices, erida_indices.num, tol, &erida_sparse);
	delete_sparse_erida_indices(&erida_indices);

	const struct cartesian_grid_3d grid_dense = {
		.coord_range = {
			{ .istart =  0, .num = 2 },
			{ .istart = -1, .num = 3 },
			{ .istart = -1, .num = 2 },
		}
	};

	// reconstruct full tensor
	const glong num_points_dense = cartesian_grid_3d_num_points(&grid_dense);
	const glong num_entries_dense = num_points_dense * num_points_dense;
	double* erida_matrix = aligned_malloc(num_entries_dense * sizeof(erida_matrix[0]));
	fill_dense_erida_matrix(&erida_sparse, &grid_dense, erida_matrix);

	// reference data
	double* erida_matrix_ref = aligned_malloc(num_entries_dense * sizeof(erida_matrix_ref[0]));
	if (read_hdf5_dataset(file, "erida", H5T_NATIVE_DOUBLE, erida_matrix_ref) < 0) {
		return "reading electron repulsion integral values from disk failed";
	}

	// compare
	if (uniform_distance(num_entries_dense, erida_matrix, erida_matrix_ref) > 1e-12) {
		return "electron repulsion integrals using the integral diagonal approximation do not match reference";
	}

	// compare with individual integral evaluations
	union cartesian_grid_point_3d points[2];
	for (glong i = 0; i < num_points_dense; ++i)
	{
		linear_index_to_cartesian_grid_point_3d(&grid_dense, i, &points[0]);

		for (glong j = 0; j < num_points_dense; ++j)
		{
			linear_index_to_cartesian_grid_point_3d(&grid_dense, j, &points[1]);

			const double val = compute_erida_gausslet_integral(&gdata, points, tol);
			if (fabs(val - erida_matrix[i * num_points_dense + j]) > 1e-13) {
				return "individual electron repulsion integral using the integral diagonal approximation does not match reference";
			}
		}
	}

	// project ERIDA integrals

	const glong num_points_sparse = cartesian_grid_3d_num_points(&grid_sparse);

	// define basis
	const glong num_states = 5;
	double* basis = aligned_malloc(num_points_sparse * num_states * sizeof(basis[0]));
	// fill basis states with pseudo-random entries
	for (glong i = 0; i < num_points_sparse * num_states; ++i) {
		basis[i] = -0.12 * cos(sqrt(7) * M_PI * ((409 * i) % 349) - 6) / (1.7 + sin(sqrt(3) * M_PI * ((557 * i) % 281)));
	}

	const glong num_entries_proj = num_states * num_states * num_states * num_states;
	double* eri_proj = aligned_malloc(num_entries_proj * sizeof(eri_proj[0]));
	project_sparse_erida_gausslet_integrals(&erida_sparse, basis, num_states, eri_proj);

	// reference calculation
	double* eri_proj_ref = aligned_calloc(num_entries_proj * sizeof(eri_proj_ref[0]));
	{
		// reconstruct full matrix
		const glong num_entries_full = num_points_sparse * num_points_sparse;
		double* erida_matrix_full = aligned_malloc(num_entries_full * sizeof(erida_matrix_full[0]));
		fill_dense_erida_matrix(&erida_sparse, &grid_sparse, erida_matrix_full);

		for (glong i = 0; i < num_points_sparse; ++i)
		{
			for (glong j = 0; j < num_points_sparse; ++j)
			{
				const double val = erida_matrix_full[i * num_points_sparse + j];

				for (glong p = 0; p < num_states; ++p)
				{
					for (glong q = 0; q < num_states; ++q)
					{
						for (glong r = 0; r < num_states; ++r)
						{
							for (glong s = 0; s < num_states; ++s)
							{
								eri_proj_ref[((p * num_states + q) * num_states + r) * num_states + s] +=
									  basis[i * num_states + p]
									* basis[i * num_states + q]
									* basis[j * num_states + r]
									* basis[j * num_states + s]
									* val;
							}
						}
					}
				}
			}
		}

		aligned_free(erida_matrix_full);
	}

	// compare
	if (uniform_distance(num_entries_proj, eri_proj, eri_proj_ref) > 1e-12) {
		return "projected electron repulsion integral values using the integral diagonal approximation do not match reference";
	}

	aligned_free(eri_proj_ref);
	aligned_free(eri_proj);
	aligned_free(basis);
	aligned_free(erida_matrix_ref);
	aligned_free(erida_matrix);
	delete_sparse_erida_gausslet_integrals(&erida_sparse);
	aligned_free(gdata.coefficients);

	H5Fclose(file);

	return 0;
}
