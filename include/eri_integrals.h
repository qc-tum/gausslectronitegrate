#pragma once

#include "grid.h"
#include "gausslet.h"


double compute_eri_gausslet_integral(const struct gausslet_data* gdata, const union cartesian_grid_point_3d points[4], const double tol);


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


void compute_sparse_eri_gausslet_integrals(
	const struct gausslet_data* gdata, const struct cartesian_grid_3d* grid,
	const double tol, struct sparse_eri_gausslet_integrals* eri);


double sparse_eri_gausslet_integrals_get_value(const struct sparse_eri_gausslet_integrals* eri, const glong index);


void delete_sparse_eri_gausslet_integrals(struct sparse_eri_gausslet_integrals* eri);


//________________________________________________________________________________________________________________________
///
/// \brief Storing electron repulsion integrals (ERIs) for Gausslet orbitals, and corresponding meta-information.
///
struct eri_gausslet_integrals
{
	double* integral_values;        //!< overlap integral values, degree-four tensor of size "number of grid points"^4
	struct cartesian_grid_3d grid;  //!< underlying grid
};


void fill_dense_eri_tensor(const struct sparse_eri_gausslet_integrals* restrict eri_sparse, struct eri_gausslet_integrals* restrict eri_dense);


void delete_eri_gausslet_integrals(struct eri_gausslet_integrals* eri);
