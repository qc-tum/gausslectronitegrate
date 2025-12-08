#pragma once

#include "grid.h"
#include "gausslet.h"


double compute_eri_gausslet_integral(const struct gausslet_data* gdata, const union cartesian_grid_point_3d points[4], const double tol);


//________________________________________________________________________________________________________________________
///
/// \brief Storing electron repulsion integrals (ERIs) for Gausslet orbitals, and corresponding meta-information.
///
/// Exploiting translational invariance by subtracting the last orbital center: (i, j | k, l) -> (i - l, j - l | k - l, 0).
///
struct eri_gausslet_integrals
{
	double* integral_values;              //!< overlap integral values, degree-three tensor of size "number of translation grid points"^3
	struct cartesian_grid_3d grid;        //!< underlying grid
	struct cartesian_grid_3d grid_trans;  //!< grid of translated coordinates after subtracting the last orbital center

};


void compute_eri_gausslet_integrals(
	const struct gausslet_data* gdata, const struct cartesian_grid_3d* grid,
	const double tol, struct eri_gausslet_integrals* eri);

void reconstruct_full_eri_tensor(const struct eri_gausslet_integrals* eri, double* full_tensor);


void delete_eri_gausslet_integrals(struct eri_gausslet_integrals* eri);
