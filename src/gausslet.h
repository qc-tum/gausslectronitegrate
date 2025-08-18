#pragma once

#include "range.h"


//________________________________________________________________________________________________________________________
///
/// \brief Data defining a Gausslet function as linear combination of elementary Gaussians.
///
struct gausslet_data
{
	double* coefficients;   //!< coefficients of elementary Gaussians
	struct range indices;   //!< corresponding indices
};
