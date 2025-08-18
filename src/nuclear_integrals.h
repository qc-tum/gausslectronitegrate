#pragma once

#include "grid.h"
#include "gausslet.h"


//________________________________________________________________________________________________________________________
///
/// \brief Description of an atomic nucleus.
///
struct atomic_nucleus
{
	double pos[3];  //!< position
	int charge;     //!< nuclear charge
};


//________________________________________________________________________________________________________________________
///
/// \brief Storing nuclear overlap integral values for Gausslet orbitals, and corresponding meta-information.
///
struct nuclear_gausslet_integrals
{
	double* integral_values;        //!< overlap integral values, square matrix of size "number of grid points"
	struct cartesian_grid_3d grid;  //!< underlying grid
	struct atomic_nucleus* nuclei;  //!< list of atomic nuclei
	int num_nuclei;                 //!< number of nuclei
};


void compute_nuclear_gausslet_integrals(
	const struct gausslet_data* gdata, const struct cartesian_grid_3d* grid,
	const struct atomic_nucleus* nuclei, const int num_nuclei,
	const double tol, struct nuclear_gausslet_integrals* ngi);


void delete_nuclear_gausslet_integrals(struct nuclear_gausslet_integrals* ngi);
