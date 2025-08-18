#pragma once

#include "range.h"


//________________________________________________________________________________________________________________________
///
/// \brief Specification of a Cartesian grid in three dimensions.
///
struct cartesian_grid_3d
{
	struct range coord_range[3];  //!< coordinate ranges of the x, y and z axes
};


//________________________________________________________________________________________________________________________
///
/// \brief Number of grid points of a Cartesian grid in three dimensions.
///
static inline long cartesian_grid_3d_num_points(const struct cartesian_grid_3d* grid)
{
	return grid->coord_range[0].num
	     * grid->coord_range[1].num
	     * grid->coord_range[2].num;
}


//________________________________________________________________________________________________________________________
///
/// \brief Convert a Cartesian grid index to a linear index.
///
static inline long cartesian_grid_3d_cartesian_to_linear_index(const struct cartesian_grid_3d* grid, const long ix, const long iy, const long iz)
{
	return (ix * grid->coord_range[1].num + iy) * grid->coord_range[2].num + iz;
}
