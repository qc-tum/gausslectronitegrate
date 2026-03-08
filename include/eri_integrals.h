#pragma once

#include "grid.h"
#include "gausslet.h"


double compute_eri_gausslet_integral(const struct gausslet_data* gdata, const union cartesian_grid_point_3d points[4], const double tol);


void evaluate_octahedral_grid_permutations(const struct cartesian_grid_3d* grid, glong* perm[48]);


glong minimum_octahedral_orbit_eri_tensor_index(
	const glong num_points, const glong* octahedral_perm[48],
	const glong i, const glong j, const glong k, const glong l);


//________________________________________________________________________________________________________________________
///
/// \brief Array of linearized electron repulsion integral four-indices (ij|kl), and corresponding meta-information.
///
struct sparse_eri_indices
{
	glong* four_indices;            //!< linearized four-indices (ij|kl); array must be sorted
	glong num;                      //!< number of entries
	struct cartesian_grid_3d grid;  //!< underlying grid
};


void enumerate_symmetry_reduced_eri_indices(const struct cartesian_grid_3d* grid, struct sparse_eri_indices* ret);


void delete_sparse_eri_indices(struct sparse_eri_indices* indices);


//________________________________________________________________________________________________________________________
///
/// \brief Selected electron repulsion integral (ERI) values for Gausslet orbitals, and corresponding meta-information.
///
struct sparse_eri_gausslet_integrals
{
	double* integral_values;        //!< overlap integral values
	glong* four_indices;            //!< linearized four-index (ij|kl) of each value; array must be sorted
	glong num_entries;              //!< number of entries
	struct cartesian_grid_3d grid;  //!< underlying grid
};


void compute_sparse_eri_gausslet_integrals(const struct gausslet_data* gdata,
	const struct cartesian_grid_3d* grid, const glong* four_indices, const glong num_indices,
	const double tol, struct sparse_eri_gausslet_integrals* eri);


double sparse_eri_gausslet_integrals_get_value(const struct sparse_eri_gausslet_integrals* eri, const glong index);

void project_sparse_eri_gausslet_integrals(const struct sparse_eri_gausslet_integrals* eri, const double* basis, const glong num_states, double* eri_proj);

void fill_dense_eri_tensor(const struct sparse_eri_gausslet_integrals* eri_sparse, const struct cartesian_grid_3d* grid_dense, double* eri_tensor);


void delete_sparse_eri_gausslet_integrals(struct sparse_eri_gausslet_integrals* eri);
