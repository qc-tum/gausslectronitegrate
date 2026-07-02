#pragma once

#include "grid.h"
#include "gausslet.h"


double compute_erida_gausslet_integral(const struct gausslet_data* gdata, const union cartesian_grid_point_3d points[2], const double tol);


//________________________________________________________________________________________________________________________
//


glong minimum_octahedral_orbit_erida_tensor_index(const glong num_points, const glong* octahedral_perm[48], const glong i, const glong j);


//________________________________________________________________________________________________________________________
///
/// \brief Array of linearized "diagonal" electron repulsion integral indices (i, j), and corresponding meta-information.
///
struct sparse_erida_indices
{
	glong* two_indices;             //!< linearized index tuples (i, j); array must be sorted
	glong num;                      //!< number of entries
	struct cartesian_grid_3d grid;  //!< underlying grid
};


void enumerate_symmetry_reduced_erida_indices(const struct cartesian_grid_3d* grid, struct sparse_erida_indices* ret);


void delete_sparse_erida_indices(struct sparse_erida_indices* indices);


//________________________________________________________________________________________________________________________
///
/// \brief Selected "diagonal" electron repulsion integral (ERIDA) values for Gausslet orbitals, and corresponding meta-information.
///
struct sparse_erida_gausslet_integrals
{
	double* integral_values;        //!< overlap integral values
	glong* two_indices;             //!< linearized index tuples (i, j) of each value; array must be sorted
	glong num_entries;              //!< number of entries
	struct cartesian_grid_3d grid;  //!< underlying grid
};


void compute_sparse_erida_gausslet_integrals(const struct gausslet_data* gdata,
	const struct cartesian_grid_3d* grid, const glong* two_indices, const glong num_indices,
	const double tol, struct sparse_erida_gausslet_integrals* erida);


double sparse_erida_gausslet_integrals_get_value(const struct sparse_erida_gausslet_integrals* erida, const glong index);


void fill_dense_erida_matrix(const struct sparse_erida_gausslet_integrals* erida_sparse, const struct cartesian_grid_3d* grid_dense, double* erida_matrix);


void delete_sparse_erida_gausslet_integrals(struct sparse_erida_gausslet_integrals* erida);
