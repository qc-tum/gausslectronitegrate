#pragma once

#include "grid.h"
#include "matrix.h"
#include "util.h"


//________________________________________________________________________________________________________________________
//

// octahedral point group

extern const struct imat3x3 octahedral_matrep[48];

extern const int octahedral_multiplication_table[48][48];


//________________________________________________________________________________________________________________________
//


void compute_cartesian_grid_3d_permutation(const struct cartesian_grid_3d* grid, const struct imat3x3* trans, glong* perm);
