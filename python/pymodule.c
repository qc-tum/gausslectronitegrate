#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "kinetic_integrals.h"
#include "nuclear_integrals.h"
#include "eri_integrals.h"
#include "aligned_memory.h"


static int parse_gausslet_coefficients(PyObject* py_gausslet_coeffs, const char* syntax, struct gausslet_data* gdata)
{
	PyArrayObject* py_array_gausslet_coeffs = (PyArrayObject*)PyArray_ContiguousFromObject(py_gausslet_coeffs, NPY_DOUBLE, 1, 1);
	if (py_array_gausslet_coeffs == NULL) {
		char msg[1024];
		sprintf(msg, "converting input argument 'gausslet_coeffs' to a NumPy array with degree 1 failed; syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return -1;
	}

	// argument checks
	if (PyArray_NDIM(py_array_gausslet_coeffs) != 1) {
		char msg[1024];
		sprintf(msg, "'gausslet_coeffs' must have degree 1, received %i; syntax: %s", PyArray_NDIM(py_array_gausslet_coeffs), syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return -1;
	}
	if (PyArray_DIM(py_array_gausslet_coeffs, 0) % 2 != 1) {
		char msg[1024];
		sprintf(msg, "'gausslet_coeffs' must have an odd number of entries; syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return -1;
	}
	if (PyArray_TYPE(py_array_gausslet_coeffs) != NPY_DOUBLE) {
		char msg[1024];
		sprintf(msg, "'gausslet_coeffs' must have 'double' format entries; syntax: %s", syntax);
		PyErr_SetString(PyExc_TypeError, msg);
		return -1;
	}
	if (!(PyArray_FLAGS(py_array_gausslet_coeffs) & NPY_ARRAY_C_CONTIGUOUS)) {
		char msg[1024];
		sprintf(msg, "'gausslet_coeffs' does not have contiguous C storage format; syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return -1;
	}

	gdata->indices.num    = PyArray_DIM(py_array_gausslet_coeffs, 0);
	gdata->indices.istart = -(gdata->indices.num - 1) / 2;
	gdata->coefficients   = aligned_malloc(gdata->indices.num * sizeof(gdata->coefficients[0]));
	memcpy(gdata->coefficients, PyArray_DATA(py_array_gausslet_coeffs), gdata->indices.num * sizeof(gdata->coefficients[0]));

	Py_DECREF(py_array_gausslet_coeffs);

	return 0;
}


static int parse_cartesian_grid(PyObject* py_grid, const char* syntax, struct cartesian_grid_3d* grid)
{
	if (!PySequence_Check(py_grid)) {
		char msg[1024];
		sprintf(msg, "cannot interpret 'grid' as a sequence; syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return -1;
	}
	if (PySequence_Length(py_grid) != 3) {
		char msg[1024];
		sprintf(msg, "'grid' must be a sequence of length three (corresponding to the Cartesian axis directions); syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return -1;
	}

	for (int i = 0; i < 3; ++i)
	{
		PyObject* py_coord_range = PySequence_GetItem(py_grid, i);

		if (!PySequence_Check(py_coord_range)) {
			char msg[1024];
			sprintf(msg, "cannot interpret 'grid[%i]' as a sequence; syntax: %s", i, syntax);
			PyErr_SetString(PyExc_ValueError, msg);
			return -1;
		}
		if (PySequence_Length(py_coord_range) != 2) {
			char msg[1024];
			sprintf(msg, "'grid[%i]' must be a sequence of length two (start coordinate and number of grid points along direction %i); syntax: %s", i, i, syntax);
			PyErr_SetString(PyExc_ValueError, msg);
			return -1;
		}

		PyObject* py_istart = PySequence_GetItem(py_coord_range, 0);
		grid->coord_range[i].istart = PyLong_AsLong(py_istart);
		if (PyErr_Occurred()) {
			char msg[1024];
			sprintf(msg, "cannot interpret the first entry in 'grid[%i]' as an integer; syntax: %s", i, syntax);
			PyErr_SetString(PyExc_ValueError, msg);
			return -1;
		}
		Py_DECREF(py_istart);

		PyObject* py_num = PySequence_GetItem(py_coord_range, 1);
		grid->coord_range[i].num = PyLong_AsLong(py_num);
		if (PyErr_Occurred()) {
			char msg[1024];
			sprintf(msg, "cannot interpret the second entry in 'grid[%i]' as an integer; syntax: %s", i, syntax);
			PyErr_SetString(PyExc_ValueError, msg);
			return -1;
		}
		Py_DECREF(py_num);

		if (grid->coord_range[i].num <= 0) {
			char msg[1024];
			sprintf(msg, "number of grid points along direction %i (second entry in 'grid[%i]') must be a positive integer; syntax: %s", i, i, syntax);
			PyErr_SetString(PyExc_ValueError, msg);
			return -1;
		}

		Py_DECREF(py_coord_range);
	}

	return 0;
}


static int parse_cartesian_grid_points(PyObject* py_grid_points, const char* syntax, const int num, union cartesian_grid_point_3d* grid_points)
{
	if (!PySequence_Check(py_grid_points)) {
		char msg[1024];
		sprintf(msg, "cannot interpret 'grid_points' as a sequence; syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return -1;
	}
	if (PySequence_Length(py_grid_points) != num) {
		char msg[1024];
		sprintf(msg, "'grid_points' must be a sequence of length %i; syntax: %s", num, syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return -1;
	}

	for (int k = 0; k < num; ++k)
	{
		PyObject* py_point = PySequence_GetItem(py_grid_points, k);

		if (!PySequence_Check(py_point)) {
			char msg[1024];
			sprintf(msg, "cannot interpret 'grid_points[%i]' as a sequence; syntax: %s", k, syntax);
			PyErr_SetString(PyExc_ValueError, msg);
			return -1;
		}
		if (PySequence_Length(py_point) != 3) {
			char msg[1024];
			sprintf(msg, "'grid_points[%i]' must be a sequence of length three (corresponding to the Cartesian coordinates); syntax: %s", k, syntax);
			PyErr_SetString(PyExc_ValueError, msg);
			return -1;
		}

		for (int i = 0; i < 3; ++i)
		{
			PyObject* py_ci = PySequence_GetItem(py_point, i);
			grid_points[k].c[i] = PyLong_AsLong(py_ci);
			if (PyErr_Occurred()) {
				char msg[1024];
				sprintf(msg, "cannot interpret the %i-th entry in 'grid_points[%i]' as an integer; syntax: %s", i, k, syntax);
				PyErr_SetString(PyExc_ValueError, msg);
				return -1;
			}
			Py_DECREF(py_ci);
		}

		Py_DECREF(py_point);
	}

	return 0;
}


struct nuclear_configuration
{
	struct atomic_nucleus* nuclei;
	int num_nuclei;
};


static int parse_nuclei(PyObject* py_nuclear_positions, PyObject* py_nuclear_charges, const char* syntax, struct nuclear_configuration* nuclear_conf)
{
	if (!PySequence_Check(py_nuclear_positions)) {
		char msg[1024];
		sprintf(msg, "cannot interpret 'nuclear_positions' as a sequence; syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return -1;
	}
	if (!PySequence_Check(py_nuclear_charges)) {
		char msg[1024];
		sprintf(msg, "cannot interpret 'nuclear_charges' as a sequence; syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return -1;
	}
	if (PySequence_Length(py_nuclear_positions) != PySequence_Length(py_nuclear_charges)) {
		char msg[1024];
		sprintf(msg, "number of nuclear positions must be equal to the number of nuclear charges; syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return -1;
	}

	nuclear_conf->num_nuclei = (int)PySequence_Length(py_nuclear_positions);
	if (nuclear_conf->num_nuclei == 0) {
		char msg[1024];
		sprintf(msg, "'nuclear_positions' cannot be an empty sequence; syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return -1;
	}

	nuclear_conf->nuclei = aligned_malloc(nuclear_conf->num_nuclei * sizeof(nuclear_conf->nuclei[0]));

	for (int i = 0; i < nuclear_conf->num_nuclei; ++i)
	{
		PyObject* py_nuclear_pos = PySequence_GetItem(py_nuclear_positions, i);

		PyArrayObject* py_array_nuclear_pos = (PyArrayObject*)PyArray_ContiguousFromObject(py_nuclear_pos, NPY_DOUBLE, 1, 1);
		if (py_array_nuclear_pos == NULL) {
			char msg[1024];
			sprintf(msg, "converting 'nuclear_positions[%i]' to a NumPy array with degree 1 failed; syntax: %s", i, syntax);
			PyErr_SetString(PyExc_ValueError, msg);
			return -1;
		}

		// argument checks
		if (PyArray_NDIM(py_array_nuclear_pos) != 1) {
			char msg[1024];
			sprintf(msg, "'nuclear_positions[%i]' must have degree 1; syntax: %s", i, syntax);
			PyErr_SetString(PyExc_ValueError, msg);
			return -1;
		}
		if (PyArray_DIM(py_array_nuclear_pos, 0) != 3) {
			char msg[1024];
			sprintf(msg, "'nuclear_positions[%i]' must be a vector with three entries; syntax: %s", i, syntax);
			PyErr_SetString(PyExc_ValueError, msg);
			return -1;
		}

		memcpy(nuclear_conf->nuclei[i].pos, PyArray_DATA(py_array_nuclear_pos), sizeof(nuclear_conf->nuclei[i].pos));

		Py_DECREF(py_array_nuclear_pos);
		Py_DECREF(py_nuclear_pos);

		PyObject* py_nuclear_charge = PySequence_GetItem(py_nuclear_charges, i);

		nuclear_conf->nuclei[i].charge = (int)PyLong_AsLong(py_nuclear_charge);
		if (PyErr_Occurred()) {
			char msg[1024];
			sprintf(msg, "cannot interpret 'nuclear_charges[%i]' as an integer; syntax: %s", i, syntax);
			PyErr_SetString(PyExc_ValueError, msg);
			return -1;
		}

		Py_DECREF(py_nuclear_charge);
	}

	return 0;
}


static PyObject* Py_grid_num_points(PyObject* Py_UNUSED(self), PyObject* args)
{
	const char* syntax = "grid_num_points(grid)";

	PyObject* py_grid;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "O", &py_grid)) {
		char msg[1024];
		sprintf(msg, "error parsing input; syntax: %s", syntax);
		PyErr_SetString(PyExc_SyntaxError, msg);
		return NULL;
	}

	// grid specification
	struct cartesian_grid_3d grid;
	if (parse_cartesian_grid(py_grid, syntax, &grid) < 0) {
		return NULL;
	}

	return PyLong_FromLong(cartesian_grid_3d_num_points(&grid));
}


static PyObject* Py_grid_point_to_linear_index(PyObject* Py_UNUSED(self), PyObject* args)
{
	const char* syntax = "grid_point_to_linear_index(grid, px, py, pz)";

	PyObject* py_grid;
	union cartesian_grid_point_3d pt;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "Olll", &py_grid, &pt.x, &pt.y, &pt.z)) {
		char msg[1024];
		sprintf(msg, "error parsing input; syntax: %s", syntax);
		PyErr_SetString(PyExc_SyntaxError, msg);
		return NULL;
	}

	// grid specification
	struct cartesian_grid_3d grid;
	if (parse_cartesian_grid(py_grid, syntax, &grid) < 0) {
		return NULL;
	}

	// range checks
	if (pt.x < grid.coord_range[0].istart || grid.coord_range[0].istart + grid.coord_range[0].num - 1 < pt.x) {
		char msg[1024];
		sprintf(msg, "'px' out of grid coordinate range; syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}
	if (pt.y < grid.coord_range[1].istart || grid.coord_range[1].istart + grid.coord_range[1].num - 1 < pt.y) {
		char msg[1024];
		sprintf(msg, "'py' out of grid coordinate range; syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}
	if (pt.z < grid.coord_range[2].istart || grid.coord_range[2].istart + grid.coord_range[2].num - 1 < pt.z) {
		char msg[1024];
		sprintf(msg, "'pz' out of grid coordinate range; syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}

	return PyLong_FromLong(cartesian_grid_point_3d_to_linear_index(&grid, &pt));
}


static PyObject* Py_linear_index_to_grid_point(PyObject* Py_UNUSED(self), PyObject* args)
{
	const char* syntax = "linear_index_to_grid_point(grid, idx)";

	PyObject* py_grid;
	glong idx;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "Ol", &py_grid, &idx)) {
		char msg[1024];
		sprintf(msg, "error parsing input; syntax: %s", syntax);
		PyErr_SetString(PyExc_SyntaxError, msg);
		return NULL;
	}

	// grid specification
	struct cartesian_grid_3d grid;
	if (parse_cartesian_grid(py_grid, syntax, &grid) < 0) {
		return NULL;
	}

	// range check
	if (idx < 0 || cartesian_grid_3d_num_points(&grid) <= idx) {
		char msg[1024];
		sprintf(msg, "'idx' out of range; syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}

	union cartesian_grid_point_3d pt;
	linear_index_to_cartesian_grid_point_3d(&grid, idx, &pt);

	return PyTuple_Pack(3,
		PyLong_FromLong(pt.x),
		PyLong_FromLong(pt.y),
		PyLong_FromLong(pt.z));
}


static PyObject* Py_compute_kinetic_gausslet_integral(PyObject* Py_UNUSED(self), PyObject* args)
{
	const char* syntax = "compute_kinetic_gausslet_integral(gausslet_coeffs, grid_points)";

	PyObject* py_gausslet_coeffs;
	PyObject* py_grid_points;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "OO", &py_gausslet_coeffs, &py_grid_points)) {
		char msg[1024];
		sprintf(msg, "error parsing input; syntax: %s", syntax);
		PyErr_SetString(PyExc_SyntaxError, msg);
		return NULL;
	}

	// Gausslet coefficients
	struct gausslet_data gdata;
	if (parse_gausslet_coefficients(py_gausslet_coeffs, syntax, &gdata) < 0) {
		return NULL;
	}

	// grid points
	union cartesian_grid_point_3d grid_points[2];
	if (parse_cartesian_grid_points(py_grid_points, syntax, 2, grid_points) < 0) {
		return NULL;
	}

	// compute kinetic overlap integral
	const double kgi = compute_kinetic_gausslet_integral(&gdata, grid_points);

	aligned_free(gdata.coefficients);

	return PyFloat_FromDouble(kgi);
}


static PyObject* Py_compute_kinetic_gausslet_integrals(PyObject* Py_UNUSED(self), PyObject* args)
{
	const char* syntax = "compute_kinetic_gausslet_integrals(gausslet_coeffs, grid)";

	PyObject* py_gausslet_coeffs;
	PyObject* py_grid;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "OO", &py_gausslet_coeffs, &py_grid)) {
		char msg[1024];
		sprintf(msg, "error parsing input; syntax: %s", syntax);
		PyErr_SetString(PyExc_SyntaxError, msg);
		return NULL;
	}

	// Gausslet coefficients
	struct gausslet_data gdata;
	if (parse_gausslet_coefficients(py_gausslet_coeffs, syntax, &gdata) < 0) {
		return NULL;
	}

	// grid specification
	struct cartesian_grid_3d grid;
	if (parse_cartesian_grid(py_grid, syntax, &grid) < 0) {
		return NULL;
	}

	// compute kinetic overlap integrals
	struct kinetic_gausslet_integrals kgi;
	compute_kinetic_gausslet_integrals(&gdata, &grid, &kgi);

	// create NumPy array of degree 2 containing integral values (return value)
	const glong num_points = cartesian_grid_3d_num_points(&grid);
	npy_intp dims[2] = { num_points, num_points };
	PyArrayObject* py_integral_values = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
	if (py_integral_values == NULL) {
		char msg[1024];
		sprintf(msg, "error creating NumPy array for return value - consider decreasing the number of grid points; syntax: %s", syntax);
		PyErr_SetString(PyExc_RuntimeError, msg);
		return NULL;
	}
	memcpy(PyArray_DATA(py_integral_values), kgi.integral_values, num_points * num_points * sizeof(kgi.integral_values[0]));

	delete_kinetic_gausslet_integrals(&kgi);
	aligned_free(gdata.coefficients);

	return (PyObject*)py_integral_values;
}


static PyObject* Py_compute_nuclear_gausslet_integral(PyObject* Py_UNUSED(self), PyObject* args)
{
	const char* syntax = "compute_nuclear_gausslet_integral(gausslet_coeffs, grid_points, nuclear_positions, nuclear_charges, tol)";

	PyObject* py_gausslet_coeffs;
	PyObject* py_grid_points;
	PyObject* py_nuclear_positions;
	PyObject* py_nuclear_charges;
	double tol;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "OOOOd", &py_gausslet_coeffs, &py_grid_points, &py_nuclear_positions, &py_nuclear_charges, &tol)) {
		char msg[1024];
		sprintf(msg, "error parsing input; syntax: %s", syntax);
		PyErr_SetString(PyExc_SyntaxError, msg);
		return NULL;
	}

	// Gausslet coefficients
	struct gausslet_data gdata;
	if (parse_gausslet_coefficients(py_gausslet_coeffs, syntax, &gdata) < 0) {
		return NULL;
	}

	// grid points
	union cartesian_grid_point_3d grid_points[2];
	if (parse_cartesian_grid_points(py_grid_points, syntax, 2, grid_points) < 0) {
		return NULL;
	}

	// nuclei
	struct nuclear_configuration nuclear_conf;
	if (parse_nuclei(py_nuclear_positions, py_nuclear_charges, syntax, &nuclear_conf) < 0) {
		return NULL;
	}

	// compute nuclear overlap integral
	const double ngi = compute_nuclear_gausslet_integral(&gdata, grid_points, nuclear_conf.nuclei, nuclear_conf.num_nuclei, tol);

	aligned_free(nuclear_conf.nuclei);
	aligned_free(gdata.coefficients);

	return PyFloat_FromDouble(ngi);
}


static PyObject* Py_compute_nuclear_gausslet_integrals(PyObject* Py_UNUSED(self), PyObject* args)
{
	const char* syntax = "compute_nuclear_gausslet_integrals(gausslet_coeffs, grid, nuclear_positions, nuclear_charges, tol)";

	PyObject* py_gausslet_coeffs;
	PyObject* py_grid;
	PyObject* py_nuclear_positions;
	PyObject* py_nuclear_charges;
	double tol;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "OOOOd", &py_gausslet_coeffs, &py_grid, &py_nuclear_positions, &py_nuclear_charges, &tol)) {
		char msg[1024];
		sprintf(msg, "error parsing input; syntax: %s", syntax);
		PyErr_SetString(PyExc_SyntaxError, msg);
		return NULL;
	}

	// Gausslet coefficients
	struct gausslet_data gdata;
	if (parse_gausslet_coefficients(py_gausslet_coeffs, syntax, &gdata) < 0) {
		return NULL;
	}

	// grid specification
	struct cartesian_grid_3d grid;
	if (parse_cartesian_grid(py_grid, syntax, &grid) < 0) {
		return NULL;
	}

	// nuclei
	struct nuclear_configuration nuclear_conf;
	if (parse_nuclei(py_nuclear_positions, py_nuclear_charges, syntax, &nuclear_conf) < 0) {
		return NULL;
	}

	// compute nuclear overlap integrals
	struct nuclear_gausslet_integrals ngi;
	compute_nuclear_gausslet_integrals(&gdata, &grid, nuclear_conf.nuclei, nuclear_conf.num_nuclei, tol, &ngi);

	// create NumPy array of degree 2 containing integral values (return value)
	const glong num_points = cartesian_grid_3d_num_points(&grid);
	npy_intp dims[2] = { num_points, num_points };
	PyArrayObject* py_integral_values = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
	if (py_integral_values == NULL) {
		char msg[1024];
		sprintf(msg, "error creating NumPy array for return value - consider decreasing the number of grid points; syntax: %s", syntax);
		PyErr_SetString(PyExc_RuntimeError, msg);
		return NULL;
	}
	memcpy(PyArray_DATA(py_integral_values), ngi.integral_values, num_points * num_points * sizeof(ngi.integral_values[0]));

	delete_nuclear_gausslet_integrals(&ngi);
	aligned_free(nuclear_conf.nuclei);
	aligned_free(gdata.coefficients);

	return (PyObject*)py_integral_values;
}


static PyObject* Py_compute_eri_gausslet_integral(PyObject* Py_UNUSED(self), PyObject* args)
{
	const char* syntax = "compute_eri_gausslet_integral(gausslet_coeffs, grid_points, tol)";

	PyObject* py_gausslet_coeffs;
	PyObject* py_grid_points;
	double tol;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "OOd", &py_gausslet_coeffs, &py_grid_points, &tol)) {
		char msg[1024];
		sprintf(msg, "error parsing input; syntax: %s", syntax);
		PyErr_SetString(PyExc_SyntaxError, msg);
		return NULL;
	}

	// Gausslet coefficients
	struct gausslet_data gdata;
	if (parse_gausslet_coefficients(py_gausslet_coeffs, syntax, &gdata) < 0) {
		return NULL;
	}

	// grid points
	union cartesian_grid_point_3d grid_points[4];
	if (parse_cartesian_grid_points(py_grid_points, syntax, 4, grid_points) < 0) {
		return NULL;
	}

	// compute electron repulsion integral (ERI)
	const double eri = compute_eri_gausslet_integral(&gdata, grid_points, tol);

	aligned_free(gdata.coefficients);

	return PyFloat_FromDouble(eri);
}


static PyObject* Py_compute_eri_gausslet_integrals(PyObject* Py_UNUSED(self), PyObject* args)
{
	const char* syntax = "compute_eri_gausslet_integrals(gausslet_coeffs, grid, tol)";

	PyObject* py_gausslet_coeffs;
	PyObject* py_grid;
	double tol;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "OOd", &py_gausslet_coeffs, &py_grid, &tol)) {
		char msg[1024];
		sprintf(msg, "error parsing input; syntax: %s", syntax);
		PyErr_SetString(PyExc_SyntaxError, msg);
		return NULL;
	}

	// Gausslet coefficients
	struct gausslet_data gdata;
	if (parse_gausslet_coefficients(py_gausslet_coeffs, syntax, &gdata) < 0) {
		return NULL;
	}

	// grid specification
	struct cartesian_grid_3d grid;
	if (parse_cartesian_grid(py_grid, syntax, &grid) < 0) {
		return NULL;
	}

	// extend to a rotationally symmetric grid for the "sparse" calculation
	const glong extent = lmax(lmax(
		lmax(labs(grid.coord_range[0].istart), labs(grid.coord_range[0].istart + grid.coord_range[0].num - 1)),
		lmax(labs(grid.coord_range[1].istart), labs(grid.coord_range[1].istart + grid.coord_range[1].num - 1))),
		lmax(labs(grid.coord_range[2].istart), labs(grid.coord_range[2].istart + grid.coord_range[2].num - 1)));
	struct cartesian_grid_3d grid_sparse = {
		.coord_range = {
			{ .istart = -extent, .num = 2 * extent + 1 },
			{ .istart = -extent, .num = 2 * extent + 1 },
			{ .istart = -extent, .num = 2 * extent + 1 },
		}
	};

	// compute electron repulsion integrals (ERIs), exploiting translational invariance
	struct eri_gausslet_integrals eri_sparse;
	compute_sparse_eri_gausslet_integrals(&gdata, &grid_sparse, tol, &eri_sparse);
	aligned_free(gdata.coefficients);

	// create NumPy array of degree 4 to store the full ERI tensor
	const glong num_points = cartesian_grid_3d_num_points(&grid);
	npy_intp dims[4] = { num_points, num_points, num_points, num_points };
	PyArrayObject* py_eri_tensor = (PyArrayObject*)PyArray_SimpleNew(4, dims, NPY_DOUBLE);
	if (py_eri_tensor == NULL) {
		char msg[1024];
		sprintf(msg, "error creating NumPy array for return value - consider decreasing the number of grid points; syntax: %s", syntax);
		PyErr_SetString(PyExc_RuntimeError, msg);
		return NULL;
	}

	// fill dense tensor entries
	struct eri_gausslet_integrals eri_dense = {
		.integral_values = PyArray_DATA(py_eri_tensor),
		.grid = grid,
	};
	fill_dense_eri_tensor(&eri_sparse, &eri_dense);

	delete_eri_gausslet_integrals(&eri_sparse);

	return (PyObject*)py_eri_tensor;
}


//________________________________________________________________________________________________________________________
///
/// \brief Get the maximum number of OpenMP threads, or 0 if OpenMP is not available.
///
static PyObject* Py_get_max_openmp_threads(PyObject* Py_UNUSED(self), PyObject* Py_UNUSED(args))
{
	#ifdef _OPENMP
	return PyLong_FromLong(omp_get_max_threads());
	#else
	return PyLong_FromLong(0);
	#endif
}


static PyMethodDef methods[] = {
	{
		.ml_name  = "grid_num_points",
		.ml_meth  = Py_grid_num_points,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Compute the overall number of points of a Cartesian grid.",
	},
	{
		.ml_name  = "grid_point_to_linear_index",
		.ml_meth  = Py_grid_point_to_linear_index,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Convert a Cartesian grid point to a linear index.",
	},
	{
		.ml_name  = "linear_index_to_grid_point",
		.ml_meth  = Py_linear_index_to_grid_point,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Convert a linear index to a Cartesian grid point.",
	},
	{
		.ml_name  = "compute_kinetic_gausslet_integral",
		.ml_meth  = Py_compute_kinetic_gausslet_integral,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Evaluate a single kinetic overlap integrals for Gausslet orbitals.",
	},
	{
		.ml_name  = "compute_kinetic_gausslet_integrals",
		.ml_meth  = Py_compute_kinetic_gausslet_integrals,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Evaluate the kinetic overlap integrals for Gausslet orbitals.",
	},
	{
		.ml_name  = "compute_nuclear_gausslet_integral",
		.ml_meth  = Py_compute_nuclear_gausslet_integral,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Evaluate a single nuclear overlap integral for Gausslet orbitals and the specified nuclear positions.",
	},
	{
		.ml_name  = "compute_nuclear_gausslet_integrals",
		.ml_meth  = Py_compute_nuclear_gausslet_integrals,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Evaluate the nuclear overlap integrals for Gausslet orbitals and the specified nuclear positions.",
	},
	{
		.ml_name  = "compute_eri_gausslet_integral",
		.ml_meth  = Py_compute_eri_gausslet_integral,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Evaluate a single electron repulsion integral (ERI) for Gausslet orbitals",
	},
	{
		.ml_name  = "compute_eri_gausslet_integrals",
		.ml_meth  = Py_compute_eri_gausslet_integrals,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Evaluate the electron repulsion integral (ERI) tensor of degree 4 for Gausslet orbitals",
	},
	{
		.ml_name  = "get_max_openmp_threads",
		.ml_meth  = Py_get_max_openmp_threads,
		.ml_flags = METH_NOARGS,
		.ml_doc   = "Get the maximum number of OpenMP threads, or 0 if OpenMP is not available.",
	},
	{
		0  // sentinel
	},
};


static struct PyModuleDef module = {
	.m_base     = PyModuleDef_HEAD_INIT,
	.m_name     = "gausslectronitegrate",  // name of module
	.m_doc      = "computing kinetic, nuclear and electron repulsion integrals for Gausslet orbitals",  // module documentation, may be NULL
	.m_size     = -1,                      // size of per-interpreter state of the module, or -1 if the module keeps state in global variables
	.m_methods  = methods,                 // module methods
	.m_slots    = NULL,                    // slot definitions for multi-phase initialization
	.m_traverse = NULL,                    // traversal function to call during GC traversal of the module object, or NULL if not needed
	.m_clear    = NULL,                    // a clear function to call during GC clearing of the module object, or NULL if not needed
	.m_free     = NULL,                    // a function to call during deallocation of the module object, or NULL if not needed
};


PyMODINIT_FUNC PyInit_gausslectronitegrate_pymodule(void)
{
	// import NumPy array module (required)
	import_array();

	PyObject* m = PyModule_Create(&module);
	if (m == NULL) {
		return NULL;
	}

	return m;
}
