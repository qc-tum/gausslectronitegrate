#pragma once

#include "grid.h"
#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Integer 3x3 matrix.
///
struct imat3x3
{
	glong data[3][3];  //!< integer entries of the matrix
};


extern const struct imat3x3 rep_octahedral[48];


//________________________________________________________________________________________________________________________
///
/// \brief Apply an integer transformation to a cubic lattice point.
///
static inline void transform_cartesian_grid_point_3d(const struct imat3x3* trans, const union cartesian_grid_point_3d* pt, union cartesian_grid_point_3d* ret)
{
	ret->x = trans->data[0][0] * pt->x + trans->data[0][1] * pt->y + trans->data[0][2] * pt->z;
	ret->y = trans->data[1][0] * pt->x + trans->data[1][1] * pt->y + trans->data[1][2] * pt->z;
	ret->z = trans->data[2][0] * pt->x + trans->data[2][1] * pt->y + trans->data[2][2] * pt->z;
}
