#pragma once

#include "grid.h"
#include "gausslet.h"


double compute_eri_gausslet_integral(const struct gausslet_data* gdata, const union cartesian_grid_point_3d points[4], const double tol);


//________________________________________________________________________________________________________________________
///
/// \brief Storing electron repulsion integrals (ERIs) for Gausslet orbitals, and corresponding meta-information.
///
struct eri_gausslet_integrals
{
	double* integral_values;        //!< overlap integral values, degree-four tensor of size "number of grid points"^4
	struct cartesian_grid_3d grid;  //!< underlying grid
};


void compute_sparse_eri_gausslet_integrals(
	const struct gausslet_data* gdata, const struct cartesian_grid_3d* grid,
	const double tol, struct eri_gausslet_integrals* eri);

void fill_dense_eri_tensor(const struct eri_gausslet_integrals* eri_sparse, struct eri_gausslet_integrals* eri_dense);


void delete_eri_gausslet_integrals(struct eri_gausslet_integrals* eri);
