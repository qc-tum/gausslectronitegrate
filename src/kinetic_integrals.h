#pragma once

#include "grid.h"
#include "gausslet.h"


//________________________________________________________________________________________________________________________
///
/// \brief Storing kinetic overlap integral values for Gausslet orbitals, and corresponding meta-information.
///
struct kinetic_gausslet_integrals
{
	double* integral_values;        //!< overlap integral values, square matrix of size "number of grid points"
	struct cartesian_grid_3d grid;  //!< underlying grid
};


void compute_kinetic_gausslet_integrals(const struct gausslet_data* gdata, const struct cartesian_grid_3d* grid, struct kinetic_gausslet_integrals* kgi);


void delete_kinetic_gausslet_integrals(struct kinetic_gausslet_integrals* kgi);
