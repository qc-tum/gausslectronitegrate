#pragma once

#include "range.h"


//________________________________________________________________________________________________________________________
///
/// \brief Specification of a single Cartesian grid point in three dimensions.
///
union cartesian_grid_point_3d
{
	struct {
		glong x, y, z;  //!< x, y, z coordinates
	};
	glong c[3];         //!< array of coordinates
};


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
static inline glong cartesian_grid_3d_num_points(const struct cartesian_grid_3d* grid)
{
	return grid->coord_range[0].num
	     * grid->coord_range[1].num
	     * grid->coord_range[2].num;
}


//________________________________________________________________________________________________________________________
///
/// \brief Convert a Cartesian grid index to a linear index.
///
static inline glong cartesian_grid_3d_cartesian_to_linear_index(const struct cartesian_grid_3d* grid, const glong ix, const glong iy, const glong iz)
{
	assert(0 <= ix && ix < grid->coord_range[0].num);
	assert(0 <= iy && iy < grid->coord_range[1].num);
	assert(0 <= iz && iz < grid->coord_range[2].num);

	return (ix * grid->coord_range[1].num + iy) * grid->coord_range[2].num + iz;
}


//________________________________________________________________________________________________________________________
///
/// \brief Convert a Cartesian grid point to a linear index.
///
static inline glong cartesian_grid_point_3d_to_linear_index(const struct cartesian_grid_3d* grid, const union cartesian_grid_point_3d* pt)
{
	return cartesian_grid_3d_cartesian_to_linear_index(grid,
		pt->x - grid->coord_range[0].istart,
		pt->y - grid->coord_range[1].istart,
		pt->z - grid->coord_range[2].istart);
}


//________________________________________________________________________________________________________________________
///
/// \brief Convert a linear index to a Cartesian grid point.
///
static inline void linear_index_to_cartesian_grid_point_3d(const struct cartesian_grid_3d* grid, const glong idx, union cartesian_grid_point_3d* pt)
{
	const glong iz = idx % grid->coord_range[2].num;
	const glong rm = idx / grid->coord_range[2].num;
	const glong iy =  rm % grid->coord_range[1].num;
	const glong ix =  rm / grid->coord_range[1].num;

	pt->x = grid->coord_range[0].istart + ix;
	pt->y = grid->coord_range[1].istart + iy;
	pt->z = grid->coord_range[2].istart + iz;
}
