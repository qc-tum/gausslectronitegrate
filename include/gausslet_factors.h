#pragma once
#include "gausslet.h"


//________________________________________________________________________________________________________________________
///
/// \brief Temporary structure storing pre-computed products of Gausslet coefficients and Gaussian factors.
///
struct gausslet_factors
{
	double** factors;       //!< Gaussian factors: factors[i] stores the factors for shift index 'i'
	struct range* indices;  //!< logical indices, for each shift
	struct range shifts;    //!< shifts
};

void compute_gausslet_factors(const struct gausslet_data* gdata, const struct range* shifts, const double tol, struct gausslet_factors* gf);

void delete_gausslet_factors(struct gausslet_factors* gf);
