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
#include "erida_integrals.h"
#include "symmetry.h"
#include "hdf5_util.h"
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


static int parse_cartesian_grid_point(PyObject* py_point, const char* syntax, union cartesian_grid_point_3d* grid_point)
{
	if (!PySequence_Check(py_point)) {
		char msg[1024];
		sprintf(msg, "cannot interpret grid point as a sequence; syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return -1;
	}
	if (PySequence_Length(py_point) != 3) {
		char msg[1024];
		sprintf(msg, "grid point must be a sequence of length three (corresponding to the Cartesian coordinates); syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return -1;
	}

	for (int i = 0; i < 3; ++i)
	{
		PyObject* py_ci = PySequence_GetItem(py_point, i);
		grid_point->c[i] = PyLong_AsLong(py_ci);
		if (PyErr_Occurred()) {
			char msg[1024];
			sprintf(msg, "cannot interpret the %i-th entry in grid point as an integer; syntax: %s", i, syntax);
			PyErr_SetString(PyExc_ValueError, msg);
			return -1;
		}
		Py_DECREF(py_ci);
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
	const char* syntax = "grid_point_to_linear_index(grid, point)";

	PyObject* py_grid;
	PyObject* py_point;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "OO", &py_grid, &py_point)) {
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

	// grid point
	union cartesian_grid_point_3d point;
	if (parse_cartesian_grid_point(py_point, syntax, &point) < 0) {
		return NULL;
	}

	// range checks
	if (point.x < grid.coord_range[0].istart || grid.coord_range[0].istart + grid.coord_range[0].num - 1 < point.x) {
		char msg[1024];
		sprintf(msg, "'point.x' out of grid coordinate range; syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}
	if (point.y < grid.coord_range[1].istart || grid.coord_range[1].istart + grid.coord_range[1].num - 1 < point.y) {
		char msg[1024];
		sprintf(msg, "'point.y' out of grid coordinate range; syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}
	if (point.z < grid.coord_range[2].istart || grid.coord_range[2].istart + grid.coord_range[2].num - 1 < point.z) {
		char msg[1024];
		sprintf(msg, "'point.z' out of grid coordinate range; syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}

	return PyLong_FromLong(cartesian_grid_point_3d_to_linear_index(&grid, &point));
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


//________________________________________________________________________________________________________________________
///
/// \brief Python sparse electron repulsion integral (ERI) object.
///
typedef struct
{
	PyObject_HEAD
	struct gausslet_data gdata;
	struct sparse_eri_gausslet_integrals eri;
	glong* octahedral_perm[48];
	double tol;
}
PySparseERIObject;


static PyObject* PySparseERI_new(PyTypeObject* type, PyObject* Py_UNUSED(args), PyObject* Py_UNUSED(kwds))
{
	PySparseERIObject* self = (PySparseERIObject*)type->tp_alloc(type, 0);
	if (self != NULL) {
		memset(&self->eri, 0, sizeof(self->eri));
	}
	return (PyObject*)self;
}


static void PySparseERI_dealloc(PySparseERIObject* self)
{
	if (self->eri.integral_values != NULL)
	{
		// assuming that the ERI structure has been initialized
		for (int i = 0; i < 48; ++i) {
			aligned_free(self->octahedral_perm[i]);
		}
		aligned_free(self->gdata.coefficients);
		delete_sparse_eri_gausslet_integrals(&self->eri);
	}
	Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject* PySparseERI_integral_value(PySparseERIObject* self, PyObject* args)
{
	if (self->eri.integral_values == NULL) {
		PyErr_SetString(PyExc_ValueError, "SparseERI object has not been initialized yet");
		return NULL;
	}

	const char* syntax = "integral_value(grid_points)";

	// parse input arguments
	PyObject* py_grid_points;
	if (!PyArg_ParseTuple(args, "O", &py_grid_points)) {
		char msg[1024];
		sprintf(msg, "error parsing input; syntax: %s", syntax);
		PyErr_SetString(PyExc_SyntaxError, msg);
		return NULL;
	}
	union cartesian_grid_point_3d grid_points[4];
	if (parse_cartesian_grid_points(py_grid_points, syntax, 4, grid_points) < 0) {
		return NULL;
	}

	const struct cartesian_grid_3d* grid = &self->eri.grid;

	const glong num_points = cartesian_grid_3d_num_points(grid);
	assert(num_points > 0);

	// shift orbital box center to origin
	for (int i = 0; i < 3; ++i)
	{
		// i-th coordinate of orbital box center times 2
		const glong center = lmin(lmin(lmin(grid_points[0].c[i], grid_points[1].c[i]), grid_points[2].c[i]), grid_points[3].c[i]) +
		                     lmax(lmax(lmax(grid_points[0].c[i], grid_points[1].c[i]), grid_points[2].c[i]), grid_points[3].c[i]);
		const glong trans = center / 2;
		assert(-1 <= center - 2 * trans && center - 2 * trans <= 1);
		grid_points[0].c[i] -= trans;
		grid_points[1].c[i] -= trans;
		grid_points[2].c[i] -= trans;
		grid_points[3].c[i] -= trans;
	}

	// check input coordinate ranges
	for (int k = 0; k < 4; ++k)
	{
		for (int i = 0; i < 3; ++i)
		{
			const glong idx = grid_points[k].c[i] - grid->coord_range[i].istart;

			if (!(0 <= idx && idx < grid->coord_range[i].num))
			{
				const char* coord_str[3] = { "x", "y", "z" };
				char msg[1024];
				sprintf(msg, "%s-coordinate of grid point %i (after box center translation) out of range; syntax: %s", coord_str[i], k, syntax);
				PyErr_SetString(PyExc_SyntaxError, msg);
				return NULL;
			}
		}
	}

	const glong idx0 = cartesian_grid_point_3d_to_linear_index(&self->eri.grid, &grid_points[0]);
	const glong idx1 = cartesian_grid_point_3d_to_linear_index(&self->eri.grid, &grid_points[1]);
	const glong idx2 = cartesian_grid_point_3d_to_linear_index(&self->eri.grid, &grid_points[2]);
	const glong idx3 = cartesian_grid_point_3d_to_linear_index(&self->eri.grid, &grid_points[3]);
	const glong idx_tensor = minimum_octahedral_orbit_eri_tensor_index(num_points, (const glong**)self->octahedral_perm, idx0, idx1, idx2, idx3);

	const double value = sparse_eri_gausslet_integrals_get_value(&self->eri, idx_tensor);

	return PyFloat_FromDouble(value);
}


static PyObject* PySparseERI_project(PySparseERIObject* self, PyObject* args)
{
	if (self->eri.integral_values == NULL) {
		PyErr_SetString(PyExc_ValueError, "SparseERI object has not been initialized yet");
		return NULL;
	}

	const struct cartesian_grid_3d* grid = &self->eri.grid;

	const glong num_points = cartesian_grid_3d_num_points(grid);
	assert(num_points > 0);

	const char* syntax = "project(basis)";

	// parse input arguments
	PyObject* py_obj_basis;
	if (!PyArg_ParseTuple(args, "O", &py_obj_basis)) {
		char msg[1024];
		sprintf(msg, "error parsing input; syntax: %s", syntax);
		PyErr_SetString(PyExc_SyntaxError, msg);
		return NULL;
	}

	// convert input argument to NumPy array
	PyArrayObject* py_basis = (PyArrayObject*)PyArray_ContiguousFromObject(py_obj_basis, NPY_DOUBLE, 2, 2);
	if (py_basis == NULL) {
		char msg[1024];
		sprintf(msg, "converting input argument to a NumPy array failed; syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}
	if (PyArray_DIM(py_basis, 0) != num_points) {
		char msg[1024];
		sprintf(msg, "leading dimension of basis matrix must be equal to the number of grid points (%li); syntax: %s", num_points, syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		Py_DECREF(py_basis);
		return NULL;
	}
	const npy_intp num_states = PyArray_DIM(py_basis, 1);
	if (num_states == 0) {
		char msg[1024];
		sprintf(msg, "number of basis states cannot be zero; syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		Py_DECREF(py_basis);
		return NULL;
	}

	npy_intp dims[4] = { num_states, num_states, num_states, num_states };
	PyArrayObject* py_eri_proj = (PyArrayObject*)PyArray_SimpleNew(4, dims, NPY_DOUBLE);
	if (py_eri_proj == NULL) {
		char msg[1024];
		sprintf(msg, "error creating NumPy array for return value; syntax: %s", syntax);
		PyErr_SetString(PyExc_RuntimeError, msg);
		Py_DECREF(py_basis);
		return NULL;
	}

	project_sparse_eri_gausslet_integrals(&self->eri, PyArray_DATA(py_basis), num_states, PyArray_DATA(py_eri_proj));

	return (PyObject*)py_eri_proj;
}


static PyObject* PySparseERI_fill_tensor(PySparseERIObject* self, PyObject* args)
{
	if (self->eri.integral_values == NULL) {
		PyErr_SetString(PyExc_ValueError, "SparseERI object has not been initialized yet");
		return NULL;
	}

	const char* syntax = "fill_tensor(dense_grid)";

	// parse input arguments
	PyObject* py_dense_grid;
	if (!PyArg_ParseTuple(args, "O", &py_dense_grid)) {
		char msg[1024];
		sprintf(msg, "error parsing input; syntax: %s", syntax);
		PyErr_SetString(PyExc_SyntaxError, msg);
		return NULL;
	}

	// dense grid specification
	struct cartesian_grid_3d dense_grid;
	if (parse_cartesian_grid(py_dense_grid, syntax, &dense_grid) < 0) {
		return NULL;
	}

	// check extent
	const glong extent = lmax(lmax(
		lmax(labs(dense_grid.coord_range[0].istart), labs(dense_grid.coord_range[0].istart + dense_grid.coord_range[0].num - 1)),
		lmax(labs(dense_grid.coord_range[1].istart), labs(dense_grid.coord_range[1].istart + dense_grid.coord_range[1].num - 1))),
		lmax(labs(dense_grid.coord_range[2].istart), labs(dense_grid.coord_range[2].istart + dense_grid.coord_range[2].num - 1)));
	if (self->eri.grid.coord_range[0].num < 2 * extent + 1 ||
	    self->eri.grid.coord_range[1].num < 2 * extent + 1 ||
	    self->eri.grid.coord_range[2].num < 2 * extent + 1)
	{
		char msg[1024];
		sprintf(msg, "sparse ERI grid too small to fill all values for the provided grid (extent %li); syntax: %s", extent, syntax);
		PyErr_SetString(PyExc_SyntaxError, msg);
		return NULL;
	}

	// create NumPy array of degree 4 to store the full ERI tensor
	const glong num_points_dense = cartesian_grid_3d_num_points(&dense_grid);
	npy_intp dims[4] = { num_points_dense, num_points_dense, num_points_dense, num_points_dense };
	PyArrayObject* py_dense_eri_tensor = (PyArrayObject*)PyArray_SimpleNew(4, dims, NPY_DOUBLE);
	if (py_dense_eri_tensor == NULL) {
		char msg[1024];
		sprintf(msg, "error creating NumPy array for return value - consider decreasing the number of grid points; syntax: %s", syntax);
		PyErr_SetString(PyExc_RuntimeError, msg);
		return NULL;
	}

	// fill dense tensor entries
	fill_dense_eri_tensor(&self->eri, &dense_grid, PyArray_DATA(py_dense_eri_tensor));

	return (PyObject*)py_dense_eri_tensor;
}


static PyObject* PySparseERI_apply(PySparseERIObject* self, PyObject* args)
{
	if (self->eri.integral_values == NULL) {
		PyErr_SetString(PyExc_ValueError, "SparseERI object has not been initialized yet");
		return NULL;
	}

	const glong num_points = cartesian_grid_3d_num_points(&self->eri.grid);
	assert(num_points > 0);

	const glong dim_state = num_points * num_points;

	const char* syntax = "apply(states)";

	// parse input arguments
	PyObject* py_obj_states;
	if (!PyArg_ParseTuple(args, "O", &py_obj_states)) {
		char msg[1024];
		sprintf(msg, "error parsing input; syntax: %s", syntax);
		PyErr_SetString(PyExc_SyntaxError, msg);
		return NULL;
	}

	// convert input argument to NumPy array
	PyArrayObject* py_states = (PyArrayObject*)PyArray_ContiguousFromObject(py_obj_states, NPY_DOUBLE, 1, 2);
	if (py_states == NULL) {
		char msg[1024];
		sprintf(msg, "converting input argument to a NumPy array failed; syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}
	if (PyArray_DIM(py_states, 0) != dim_state) {
		char msg[1024];
		sprintf(msg, "leading dimension of the input state matrix must be equal to the number of grid points squared (%li); syntax: %s", dim_state, syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		Py_DECREF(py_states);
		return NULL;
	}
	const npy_intp num_states = (PyArray_NDIM(py_states) == 1 ? 1 : PyArray_DIM(py_states, 1));
	if (num_states == 0) {
		char msg[1024];
		sprintf(msg, "number of input states cannot be zero; syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		Py_DECREF(py_states);
		return NULL;
	}

	npy_intp dims[2] = { dim_state, num_states };
	PyArrayObject* py_eri_states = (PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(py_states), dims, NPY_DOUBLE);
	if (py_eri_states == NULL) {
		char msg[1024];
		sprintf(msg, "error creating NumPy array for return value; syntax: %s", syntax);
		PyErr_SetString(PyExc_RuntimeError, msg);
		Py_DECREF(py_states);
		return NULL;
	}

	apply_sparse_eri_tensor(&self->eri, PyArray_DATA(py_states), num_states, PyArray_DATA(py_eri_states));

	return (PyObject*)py_eri_states;
}


static PyMethodDef PySparseERI_methods[] = {
	{
		.ml_name  = "integral_value",
		.ml_meth  = (PyCFunction)PySparseERI_integral_value,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Return the ERI integral value at the specified coordinates.",
	},
	{
		.ml_name  = "project",
		.ml_meth  = (PyCFunction)PySparseERI_project,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Project the ERI tensor onto the specified basis along each axis.",
	},
	{
		.ml_name  = "fill_tensor",
		.ml_meth  = (PyCFunction)PySparseERI_fill_tensor,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Fill the dense electron repulsion integral (ERI) tensor of degree 4 for Gausslet orbitals.",
	},
	{
		.ml_name  = "apply",
		.ml_meth  = (PyCFunction)PySparseERI_apply,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Apply the electron repulsion integral (ERI) tensor interpreted as a matrix in physicist's convention to a list of states.",
	},
	{
		0  // sentinel
	},
};


static PyObject* PySparseERI_grid(PySparseERIObject* self, void* Py_UNUSED(closure))
{
	if (self->eri.integral_values == NULL) {
		PyErr_SetString(PyExc_ValueError, "SparseERI object has not been initialized yet");
		return NULL;
	}

	PyObject* py_coord_range[3];
	for (int i = 0; i < 3; ++i) {
		py_coord_range[i] = PyTuple_Pack(2,
			PyLong_FromLong(self->eri.grid.coord_range[i].istart),
			PyLong_FromLong(self->eri.grid.coord_range[i].num));
	}

	return PyTuple_Pack(3, py_coord_range[0], py_coord_range[1], py_coord_range[2]);
}


static PyObject* PySparseERI_tol(PySparseERIObject* self, void* Py_UNUSED(closure))
{
	if (self->eri.integral_values == NULL) {
		PyErr_SetString(PyExc_ValueError, "SparseERI object has not been initialized yet");
		return NULL;
	}

	return PyFloat_FromDouble(self->tol);
}


static PyObject* PySparseERI_num_entries(PySparseERIObject* self, void* Py_UNUSED(closure))
{
	if (self->eri.integral_values == NULL) {
		PyErr_SetString(PyExc_ValueError, "SparseERI object has not been initialized yet");
		return NULL;
	}

	return PyLong_FromLong(self->eri.num_entries);
}


static PyObject* PySparseERI_gausslet_coeffs(PySparseERIObject* self, void* Py_UNUSED(closure))
{
	if (self->eri.integral_values == NULL) {
		PyErr_SetString(PyExc_ValueError, "SparseERI object has not been initialized yet");
		return NULL;
	}

	npy_intp dims[1] = { self->gdata.indices.num };
	PyArrayObject* py_coeffs = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	if (py_coeffs == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "error creating NumPy vector");
		return NULL;
	}
	memcpy(PyArray_DATA(py_coeffs), self->gdata.coefficients, self->gdata.indices.num * sizeof(double));

	return (PyObject*)py_coeffs;
}


static struct PyGetSetDef PySparseERI_getset[] = {
	{
		.name    = "grid",
		.get     = (getter)PySparseERI_grid,
		.set     = NULL,
		.doc     = "underlying Cartesian grid",
		.closure = NULL,
	},
	{
		.name    = "tol",
		.get     = (getter)PySparseERI_tol,
		.set     = NULL,
		.doc     = "truncation tolerance used for calculating the ERIs",
		.closure = NULL,
	},
	{
		.name    = "num_entries",
		.get     = (getter)PySparseERI_num_entries,
		.set     = NULL,
		.doc     = "number of stored ERI values after translational, octahedral and permutational symmetry reductions",
		.closure = NULL,
	},
	{
		.name    = "gausslet_coeffs",
		.get     = (getter)PySparseERI_gausslet_coeffs,
		.set     = NULL,
		.doc     = "Gausslet coefficients used for calculating the ERIs",
		.closure = NULL,
	},
	{
		0  // sentinel
	},
};


static PyTypeObject PySparseERIType = {
	.ob_base      = PyVarObject_HEAD_INIT(NULL, 0)
	.tp_name      = "gausslectronitegrate.SparseERI",
	.tp_doc       = PyDoc_STR("SparseERI object"),
	.tp_basicsize = sizeof(PySparseERIObject),
	.tp_itemsize  = 0,
	.tp_flags     = Py_TPFLAGS_DEFAULT,
	.tp_new       = PySparseERI_new,
	.tp_init      = NULL,
	.tp_dealloc   = (destructor)PySparseERI_dealloc,
	.tp_methods   = PySparseERI_methods,
	.tp_getset    = PySparseERI_getset,
};


static PyObject* Py_compute_sparse_eri_gausslet_integrals(PyObject* Py_UNUSED(self), PyObject* args)
{
	const char* syntax = "compute_sparse_eri_gausslet_integrals(gausslet_coeffs, grid, tol)";

	PySparseERIObject* py_eri = (PySparseERIObject*)PySparseERI_new(&PySparseERIType, NULL, NULL);
	if (py_eri == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "error creating SparseERI object");
		return NULL;
	}

	PyObject* py_gausslet_coeffs;
	PyObject* py_grid;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "OOd", &py_gausslet_coeffs, &py_grid, &py_eri->tol)) {
		char msg[1024];
		sprintf(msg, "error parsing input; syntax: %s", syntax);
		PyErr_SetString(PyExc_SyntaxError, msg);
		return NULL;
	}

	if (py_eri->tol < 0) {
		char msg[1024];
		sprintf(msg, "'tol' cannot be negative, received %g; syntax: %s", py_eri->tol, syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}

	// Gausslet coefficients
	if (parse_gausslet_coefficients(py_gausslet_coeffs, syntax, &py_eri->gdata) < 0) {
		return NULL;
	}

	// grid specification
	struct cartesian_grid_3d grid;
	if (parse_cartesian_grid(py_grid, syntax, &grid) < 0) {
		return NULL;
	}
	// ensure that grid is rotationally symmetric
	{
		const glong extent = lmax(lmax(
			lmax(labs(grid.coord_range[0].istart), labs(grid.coord_range[0].istart + grid.coord_range[0].num - 1)),
			lmax(labs(grid.coord_range[1].istart), labs(grid.coord_range[1].istart + grid.coord_range[1].num - 1))),
			lmax(labs(grid.coord_range[2].istart), labs(grid.coord_range[2].istart + grid.coord_range[2].num - 1)));

		for (int i = 0; i < 3; ++i)
		{
			if (grid.coord_range[i].istart != -extent || grid.coord_range[i].num != 2 * extent + 1)
			{
				char msg[1024];
				sprintf(msg, "specified grid is not rotationally symmetric; syntax: %s", syntax);
				PyErr_SetString(PyExc_ValueError, msg);
				return NULL;
			}
		}
	}

	// precompute the 3D grid point permutations effected by the octahedral point group elements
	evaluate_octahedral_grid_permutations(&grid, py_eri->octahedral_perm);

	// compute electron repulsion integrals (ERIs), exploiting translational, octahedral and permutational symmetries
	struct sparse_eri_indices eri_indices;
	enumerate_symmetry_reduced_eri_indices(&grid, &eri_indices);
	compute_sparse_eri_gausslet_integrals(&py_eri->gdata, &grid, eri_indices.four_indices, eri_indices.num, py_eri->tol, &py_eri->eri);
	delete_sparse_eri_indices(&eri_indices);

	return (PyObject*)py_eri;
}


static PyObject* Py_load_sparse_eri_gausslet_integrals(PyObject* Py_UNUSED(self), PyObject* args)
{
	const char* syntax = "load_sparse_eri_gausslet_integrals(filename)";

	PySparseERIObject* py_eri = (PySparseERIObject*)PySparseERI_new(&PySparseERIType, NULL, NULL);
	if (py_eri == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "error creating SparseERI object");
		return NULL;
	}

	// parse input arguments
	const char* filename;
	Py_ssize_t fnlen;
	if (!PyArg_ParseTuple(args, "s#", &filename, &fnlen)) {
		char msg[1024];
		sprintf(msg, "error parsing input; syntax: %s", syntax);
		PyErr_SetString(PyExc_SyntaxError, msg);
		return NULL;
	}

	hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0)
	{
		char msg[1024];
		sprintf(msg, "'H5Fopen' failed for file '%s'; syntax: %s", filename, syntax);
		PyErr_SetString(PyExc_RuntimeError, msg);
		return NULL;
	}

	if (sizeof(long long) != 8)
	{
		char msg[1024];
		sprintf(msg, "'long long' type is %li bytes on native platform, expecting 8 bytes", sizeof(long long));
		PyErr_WarnEx(PyExc_RuntimeWarning, msg, 1);
	}

	// grid specification
	{
		glong grid_length;
		if (read_hdf5_attribute(file, "grid_length", H5T_NATIVE_LLONG, &grid_length) < 0)  // assuming that "long long" is 64 bit
		{
			H5Fclose(file);
			char msg[1024];
			sprintf(msg, "reading 'grid_length' attribute from HDF5 file '%s' failed", filename);
			PyErr_SetString(PyExc_RuntimeError, msg);
			return NULL;
		}
		if (grid_length <= 0)
		{
			H5Fclose(file);
			char msg[1024];
			sprintf(msg, "read invalid negative 'grid_length' attribute from HDF5 file '%s'", filename);
			PyErr_SetString(PyExc_ValueError, msg);
			return NULL;
		}

		// require a rotationally symmetric grid
		const struct cartesian_grid_3d grid = {
			.coord_range = {
				{ .istart = -(grid_length - 1) / 2, .num = grid_length },
				{ .istart = -(grid_length - 1) / 2, .num = grid_length },
				{ .istart = -(grid_length - 1) / 2, .num = grid_length },
			}
		};
		py_eri->eri.grid = grid;
	}

	// precompute the 3D grid point permutations effected by the octahedral point group elements
	evaluate_octahedral_grid_permutations(&py_eri->eri.grid, py_eri->octahedral_perm);

	// truncation tolerance
	if (read_hdf5_attribute(file, "tol", H5T_NATIVE_DOUBLE, &py_eri->tol) < 0)
	{
		H5Fclose(file);
		char msg[1024];
		sprintf(msg, "reading 'tol' attribute from HDF5 file '%s' failed", filename);
		PyErr_SetString(PyExc_RuntimeError, msg);
		return NULL;
	}

	// Gausslet coefficients
	{
		hsize_t dim[1];
		if (get_hdf5_dataset_dims(file, "gausslet_coeffs", dim) < 0)
		{
			H5Fclose(file);
			char msg[1024];
			sprintf(msg, "obtaining dimensions of Gausslet coefficients from HDF5 file '%s' failed", filename);
			PyErr_SetString(PyExc_RuntimeError, msg);
			return NULL;
		}
		py_eri->gdata.indices.num = dim[0];
		if (py_eri->gdata.indices.num % 2 != 1)
		{
			H5Fclose(file);
			char msg[1024];
			sprintf(msg, "expecting an odd number of Gausslet coefficients in HDF5 file '%s'", filename);
			PyErr_SetString(PyExc_RuntimeError, msg);
			return NULL;
		}
		py_eri->gdata.indices.istart = -(py_eri->gdata.indices.num - 1) / 2;

		py_eri->gdata.coefficients = aligned_malloc(py_eri->gdata.indices.num * sizeof(py_eri->gdata.coefficients[0]));
		if (read_hdf5_dataset(file, "gausslet_coeffs", H5T_NATIVE_DOUBLE, py_eri->gdata.coefficients) < 0)
		{
			H5Fclose(file);
			char msg[1024];
			sprintf(msg, "reading Gausslet coefficients from HDF5 file '%s' failed", filename);
			PyErr_SetString(PyExc_RuntimeError, msg);
			return NULL;
		}
	}

	// integral values and indices
	{
		hsize_t dim[1];
		if (get_hdf5_dataset_dims(file, "integral_values", dim) < 0)
		{
			H5Fclose(file);
			char msg[1024];
			sprintf(msg, "obtaining dimensions of integral values from HDF5 file '%s' failed", filename);
			PyErr_SetString(PyExc_RuntimeError, msg);
			return NULL;
		}
		py_eri->eri.num_entries = dim[0];

		py_eri->eri.integral_values = aligned_malloc(py_eri->eri.num_entries * sizeof(py_eri->eri.integral_values[0]));
		if (read_hdf5_dataset(file, "integral_values", H5T_NATIVE_DOUBLE, py_eri->eri.integral_values) < 0)
		{
			H5Fclose(file);
			char msg[1024];
			sprintf(msg, "reading integral values from HDF5 file '%s' failed", filename);
			PyErr_SetString(PyExc_RuntimeError, msg);
			return NULL;
		}

		py_eri->eri.four_indices = aligned_malloc(py_eri->eri.num_entries * sizeof(py_eri->eri.four_indices[0]));
		if (read_hdf5_dataset(file, "four_indices", H5T_NATIVE_LLONG, py_eri->eri.four_indices) < 0)  // assuming that "long long" is 64 bit
		{
			H5Fclose(file);
			char msg[1024];
			sprintf(msg, "reading four-indices from HDF5 file '%s' failed", filename);
			PyErr_SetString(PyExc_RuntimeError, msg);
			return NULL;
		}
	}

	H5Fclose(file);

	return (PyObject*)py_eri;
}


static PyObject* Py_compute_erida_gausslet_integral(PyObject* Py_UNUSED(self), PyObject* args)
{
	const char* syntax = "compute_erida_gausslet_integral(gausslet_coeffs, grid_points, tol)";

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
	union cartesian_grid_point_3d grid_points[2];
	if (parse_cartesian_grid_points(py_grid_points, syntax, 2, grid_points) < 0) {
		return NULL;
	}

	// compute electron repulsion integral using the integral diagonal approximation (ERIDA)
	const double erida = compute_erida_gausslet_integral(&gdata, grid_points, tol);

	aligned_free(gdata.coefficients);

	return PyFloat_FromDouble(erida);
}


//________________________________________________________________________________________________________________________
///
/// \brief Python object storing sparse electron repulsion integrals using the integral diagonal approximation (ERIDA).
///
typedef struct
{
	PyObject_HEAD
	struct gausslet_data gdata;
	struct sparse_erida_gausslet_integrals erida;
	glong* octahedral_perm[48];
	double tol;
}
PySparseERIDAObject;


static PyObject* PySparseERIDA_new(PyTypeObject* type, PyObject* Py_UNUSED(args), PyObject* Py_UNUSED(kwds))
{
	PySparseERIDAObject* self = (PySparseERIDAObject*)type->tp_alloc(type, 0);
	if (self != NULL) {
		memset(&self->erida, 0, sizeof(self->erida));
	}
	return (PyObject*)self;
}


static void PySparseERIDA_dealloc(PySparseERIDAObject* self)
{
	if (self->erida.integral_values != NULL)
	{
		// assuming that the ERIDA structure has been initialized
		for (int i = 0; i < 48; ++i) {
			aligned_free(self->octahedral_perm[i]);
		}
		aligned_free(self->gdata.coefficients);
		delete_sparse_erida_gausslet_integrals(&self->erida);
	}
	Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject* PySparseERIDA_integral_value(PySparseERIDAObject* self, PyObject* args)
{
	if (self->erida.integral_values == NULL) {
		PyErr_SetString(PyExc_ValueError, "SparseERIDA object has not been initialized yet");
		return NULL;
	}

	const char* syntax = "integral_value(grid_points)";

	// parse input arguments
	PyObject* py_grid_points;
	if (!PyArg_ParseTuple(args, "O", &py_grid_points)) {
		char msg[1024];
		sprintf(msg, "error parsing input; syntax: %s", syntax);
		PyErr_SetString(PyExc_SyntaxError, msg);
		return NULL;
	}
	union cartesian_grid_point_3d grid_points[2];
	if (parse_cartesian_grid_points(py_grid_points, syntax, 2, grid_points) < 0) {
		return NULL;
	}

	const struct cartesian_grid_3d* grid = &self->erida.grid;

	const glong num_points = cartesian_grid_3d_num_points(grid);
	assert(num_points > 0);

	// shift orbital box center to origin
	union cartesian_grid_point_3d bias;
	for (int i = 0; i < 3; ++i)
	{
		// i-th coordinate of orbital box center times 2
		const glong center = lmin(grid_points[0].c[i], grid_points[1].c[i]) +
		                     lmax(grid_points[0].c[i], grid_points[1].c[i]);
		const glong trans = center / 2;
		bias.c[i] = center - 2 * trans;
		assert(-1 <= bias.c[i] && bias.c[i] <= 1);
		grid_points[0].c[i] -= trans;
		grid_points[1].c[i] -= trans;
	}

	// check input coordinate ranges
	for (int k = 0; k < 2; ++k)
	{
		for (int i = 0; i < 3; ++i)
		{
			const glong idx = grid_points[k].c[i] - grid->coord_range[i].istart;

			if (!(0 <= idx && idx < grid->coord_range[i].num))
			{
				const char* coord_str[3] = { "x", "y", "z" };
				char msg[1024];
				sprintf(msg, "%s-coordinate of grid point %i (after box center translation) out of range; syntax: %s", coord_str[i], k, syntax);
				PyErr_SetString(PyExc_SyntaxError, msg);
				return NULL;
			}
		}
	}

	glong idx_tensor = GLONG_MAX;
	for (glong bx = 0; bx <= labs(bias.x); ++bx)
	{
		for (glong by = 0; by <= labs(bias.y); ++by)
		{
			for (glong bz = 0; bz <= labs(bias.z); ++bz)
			{
				const union cartesian_grid_point_3d pt_i_p = {
					.x = grid_points[0].x - bx*bias.x,
					.y = grid_points[0].y - by*bias.y,
					.z = grid_points[0].z - bz*bias.z,
				};
				const union cartesian_grid_point_3d pt_j_p = {
					.x = grid_points[1].x - bx*bias.x,
					.y = grid_points[1].y - by*bias.y,
					.z = grid_points[1].z - bz*bias.z,
				};
				const glong idx_i_p = cartesian_grid_point_3d_to_linear_index(&self->erida.grid, &pt_i_p);
				const glong idx_j_p = cartesian_grid_point_3d_to_linear_index(&self->erida.grid, &pt_j_p);

				idx_tensor = lmin(idx_tensor, minimum_octahedral_orbit_erida_tensor_index(num_points, (const glong**)self->octahedral_perm, idx_i_p, idx_j_p));
			}
		}
	}

	const double value = sparse_erida_gausslet_integrals_get_value(&self->erida, idx_tensor);

	return PyFloat_FromDouble(value);
}


static PyObject* PySparseERIDA_project(PySparseERIDAObject* self, PyObject* args)
{
	if (self->erida.integral_values == NULL) {
		PyErr_SetString(PyExc_ValueError, "SparseERIDA object has not been initialized yet");
		return NULL;
	}

	const struct cartesian_grid_3d* grid = &self->erida.grid;

	const glong num_points = cartesian_grid_3d_num_points(grid);
	assert(num_points > 0);

	const char* syntax = "project(basis)";

	// parse input arguments
	PyObject* py_obj_basis;
	if (!PyArg_ParseTuple(args, "O", &py_obj_basis)) {
		char msg[1024];
		sprintf(msg, "error parsing input; syntax: %s", syntax);
		PyErr_SetString(PyExc_SyntaxError, msg);
		return NULL;
	}

	// convert input argument to NumPy array
	PyArrayObject* py_basis = (PyArrayObject*)PyArray_ContiguousFromObject(py_obj_basis, NPY_DOUBLE, 2, 2);
	if (py_basis == NULL) {
		char msg[1024];
		sprintf(msg, "converting input argument to a NumPy array failed; syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}
	if (PyArray_DIM(py_basis, 0) != num_points) {
		char msg[1024];
		sprintf(msg, "leading dimension of basis matrix must be equal to the number of grid points (%li); syntax: %s", num_points, syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		Py_DECREF(py_basis);
		return NULL;
	}
	const npy_intp num_states = PyArray_DIM(py_basis, 1);
	if (num_states == 0) {
		char msg[1024];
		sprintf(msg, "number of basis states cannot be zero; syntax: %s", syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		Py_DECREF(py_basis);
		return NULL;
	}

	npy_intp dims[4] = { num_states, num_states, num_states, num_states };
	PyArrayObject* py_eri_proj = (PyArrayObject*)PyArray_SimpleNew(4, dims, NPY_DOUBLE);
	if (py_eri_proj == NULL) {
		char msg[1024];
		sprintf(msg, "error creating NumPy array for return value; syntax: %s", syntax);
		PyErr_SetString(PyExc_RuntimeError, msg);
		Py_DECREF(py_basis);
		return NULL;
	}

	project_sparse_erida_gausslet_integrals(&self->erida, PyArray_DATA(py_basis), num_states, PyArray_DATA(py_eri_proj));

	return (PyObject*)py_eri_proj;
}


static PyObject* PySparseERIDA_fill_matrix(PySparseERIDAObject* self, PyObject* args)
{
	if (self->erida.integral_values == NULL) {
		PyErr_SetString(PyExc_ValueError, "SparseERIDA object has not been initialized yet");
		return NULL;
	}

	const char* syntax = "fill_matrix(dense_grid)";

	// parse input arguments
	PyObject* py_dense_grid;
	if (!PyArg_ParseTuple(args, "O", &py_dense_grid)) {
		char msg[1024];
		sprintf(msg, "error parsing input; syntax: %s", syntax);
		PyErr_SetString(PyExc_SyntaxError, msg);
		return NULL;
	}

	// dense grid specification
	struct cartesian_grid_3d dense_grid;
	if (parse_cartesian_grid(py_dense_grid, syntax, &dense_grid) < 0) {
		return NULL;
	}

	// check extent
	const glong extent = lmax(lmax(
		lmax(labs(dense_grid.coord_range[0].istart), labs(dense_grid.coord_range[0].istart + dense_grid.coord_range[0].num - 1)),
		lmax(labs(dense_grid.coord_range[1].istart), labs(dense_grid.coord_range[1].istart + dense_grid.coord_range[1].num - 1))),
		lmax(labs(dense_grid.coord_range[2].istart), labs(dense_grid.coord_range[2].istart + dense_grid.coord_range[2].num - 1)));
	if (self->erida.grid.coord_range[0].num < 2 * extent + 1 ||
	    self->erida.grid.coord_range[1].num < 2 * extent + 1 ||
	    self->erida.grid.coord_range[2].num < 2 * extent + 1)
	{
		char msg[1024];
		sprintf(msg, "sparse ERIDA grid too small to fill all values for the provided grid (extent %li); syntax: %s", extent, syntax);
		PyErr_SetString(PyExc_SyntaxError, msg);
		return NULL;
	}

	// create NumPy array of degree 2 to store the full ERIDA matrix
	const glong num_points_dense = cartesian_grid_3d_num_points(&dense_grid);
	npy_intp dims[2] = { num_points_dense, num_points_dense };
	PyArrayObject* py_dense_erida_matrix = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
	if (py_dense_erida_matrix == NULL) {
		char msg[1024];
		sprintf(msg, "error creating NumPy array for return value - consider decreasing the number of grid points; syntax: %s", syntax);
		PyErr_SetString(PyExc_RuntimeError, msg);
		return NULL;
	}

	// fill dense entries
	fill_dense_erida_matrix(&self->erida, &dense_grid, PyArray_DATA(py_dense_erida_matrix));

	return (PyObject*)py_dense_erida_matrix;
}


static PyMethodDef PySparseERIDA_methods[] = {
	{
		.ml_name  = "integral_value",
		.ml_meth  = (PyCFunction)PySparseERIDA_integral_value,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Return the ERIDA integral value at the specified coordinates.",
	},
	{
		.ml_name  = "project",
		.ml_meth  = (PyCFunction)PySparseERIDA_project,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Compute the ERI tensor (of degree four) of the specified basis functions based on the ERIDA matrix.",
	},
	{
		.ml_name  = "fill_matrix",
		.ml_meth  = (PyCFunction)PySparseERIDA_fill_matrix,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Fill the dense matrix entries with the electron repulsion integrals using the integral diagonal approximation (ERIDA) for Gausslet orbitals."
	},
	{
		0  // sentinel
	},
};


static PyObject* PySparseERIDA_grid(PySparseERIDAObject* self, void* Py_UNUSED(closure))
{
	if (self->erida.integral_values == NULL) {
		PyErr_SetString(PyExc_ValueError, "SparseERIDA object has not been initialized yet");
		return NULL;
	}

	PyObject* py_coord_range[3];
	for (int i = 0; i < 3; ++i) {
		py_coord_range[i] = PyTuple_Pack(2,
			PyLong_FromLong(self->erida.grid.coord_range[i].istart),
			PyLong_FromLong(self->erida.grid.coord_range[i].num));
	}

	return PyTuple_Pack(3, py_coord_range[0], py_coord_range[1], py_coord_range[2]);
}


static PyObject* PySparseERIDA_tol(PySparseERIDAObject* self, void* Py_UNUSED(closure))
{
	if (self->erida.integral_values == NULL) {
		PyErr_SetString(PyExc_ValueError, "SparseERIDA object has not been initialized yet");
		return NULL;
	}

	return PyFloat_FromDouble(self->tol);
}


static PyObject* PySparseERIDA_num_entries(PySparseERIDAObject* self, void* Py_UNUSED(closure))
{
	if (self->erida.integral_values == NULL) {
		PyErr_SetString(PyExc_ValueError, "SparseERIDA object has not been initialized yet");
		return NULL;
	}

	return PyLong_FromLong(self->erida.num_entries);
}


static PyObject* PySparseERIDA_gausslet_coeffs(PySparseERIDAObject* self, void* Py_UNUSED(closure))
{
	if (self->erida.integral_values == NULL) {
		PyErr_SetString(PyExc_ValueError, "SparseERIDA object has not been initialized yet");
		return NULL;
	}

	npy_intp dims[1] = { self->gdata.indices.num };
	PyArrayObject* py_coeffs = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	if (py_coeffs == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "error creating NumPy vector");
		return NULL;
	}
	memcpy(PyArray_DATA(py_coeffs), self->gdata.coefficients, self->gdata.indices.num * sizeof(double));

	return (PyObject*)py_coeffs;
}


static struct PyGetSetDef PySparseERIDA_getset[] = {
	{
		.name    = "grid",
		.get     = (getter)PySparseERIDA_grid,
		.set     = NULL,
		.doc     = "underlying Cartesian grid",
		.closure = NULL,
	},
	{
		.name    = "tol",
		.get     = (getter)PySparseERIDA_tol,
		.set     = NULL,
		.doc     = "truncation tolerance used for calculating the ERIDAs",
		.closure = NULL,
	},
	{
		.name    = "num_entries",
		.get     = (getter)PySparseERIDA_num_entries,
		.set     = NULL,
		.doc     = "number of stored ERIDA values after translational, octahedral and permutational symmetry reductions",
		.closure = NULL,
	},
	{
		.name    = "gausslet_coeffs",
		.get     = (getter)PySparseERIDA_gausslet_coeffs,
		.set     = NULL,
		.doc     = "Gausslet coefficients used for calculating the ERIDAs",
		.closure = NULL,
	},
	{
		0  // sentinel
	},
};


static PyTypeObject PySparseERIDAType = {
	.ob_base      = PyVarObject_HEAD_INIT(NULL, 0)
	.tp_name      = "gausslectronitegrate.SparseERIDA",
	.tp_doc       = PyDoc_STR("SparseERIDA object"),
	.tp_basicsize = sizeof(PySparseERIDAObject),
	.tp_itemsize  = 0,
	.tp_flags     = Py_TPFLAGS_DEFAULT,
	.tp_new       = PySparseERIDA_new,
	.tp_init      = NULL,
	.tp_dealloc   = (destructor)PySparseERIDA_dealloc,
	.tp_methods   = PySparseERIDA_methods,
	.tp_getset    = PySparseERIDA_getset,
};


static PyObject* Py_compute_sparse_erida_gausslet_integrals(PyObject* Py_UNUSED(self), PyObject* args)
{
	const char* syntax = "compute_sparse_erida_gausslet_integrals(gausslet_coeffs, grid, tol)";

	PySparseERIDAObject* py_erida = (PySparseERIDAObject*)PySparseERIDA_new(&PySparseERIDAType, NULL, NULL);
	if (py_erida == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "error creating SparseERIDA object");
		return NULL;
	}

	PyObject* py_gausslet_coeffs;
	PyObject* py_grid;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "OOd", &py_gausslet_coeffs, &py_grid, &py_erida->tol)) {
		char msg[1024];
		sprintf(msg, "error parsing input; syntax: %s", syntax);
		PyErr_SetString(PyExc_SyntaxError, msg);
		return NULL;
	}

	if (py_erida->tol < 0) {
		char msg[1024];
		sprintf(msg, "'tol' cannot be negative, received %g; syntax: %s", py_erida->tol, syntax);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}

	// Gausslet coefficients
	if (parse_gausslet_coefficients(py_gausslet_coeffs, syntax, &py_erida->gdata) < 0) {
		return NULL;
	}

	// grid specification
	struct cartesian_grid_3d grid;
	if (parse_cartesian_grid(py_grid, syntax, &grid) < 0) {
		return NULL;
	}
	// ensure that grid is rotationally symmetric
	{
		const glong extent = lmax(lmax(
			lmax(labs(grid.coord_range[0].istart), labs(grid.coord_range[0].istart + grid.coord_range[0].num - 1)),
			lmax(labs(grid.coord_range[1].istart), labs(grid.coord_range[1].istart + grid.coord_range[1].num - 1))),
			lmax(labs(grid.coord_range[2].istart), labs(grid.coord_range[2].istart + grid.coord_range[2].num - 1)));

		for (int i = 0; i < 3; ++i)
		{
			if (grid.coord_range[i].istart != -extent || grid.coord_range[i].num != 2 * extent + 1)
			{
				char msg[1024];
				sprintf(msg, "specified grid is not rotationally symmetric; syntax: %s", syntax);
				PyErr_SetString(PyExc_ValueError, msg);
				return NULL;
			}
		}
	}

	// precompute the 3D grid point permutations effected by the octahedral point group elements
	evaluate_octahedral_grid_permutations(&grid, py_erida->octahedral_perm);

	// compute electron repulsion integrals (ERIDAs), exploiting translational, octahedral and permutational symmetries
	struct sparse_erida_indices erida_indices;
	enumerate_symmetry_reduced_erida_indices(&grid, &erida_indices);
	compute_sparse_erida_gausslet_integrals(&py_erida->gdata, &grid, erida_indices.two_indices, erida_indices.num, py_erida->tol, &py_erida->erida);
	delete_sparse_erida_indices(&erida_indices);

	return (PyObject*)py_erida;
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
		.ml_doc   = "Evaluate a single electron repulsion integral (ERI) for Gausslet orbitals.",
	},
	{
		.ml_name  = "compute_sparse_eri_gausslet_integrals",
		.ml_meth  = Py_compute_sparse_eri_gausslet_integrals,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Evaluate the sparse representation of the electron repulsion integral (ERI) tensor of degree 4 for Gausslet orbitals.",
	},
	{
		.ml_name  = "load_sparse_eri_gausslet_integrals",
		.ml_meth  = Py_load_sparse_eri_gausslet_integrals,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Load the sparse representation of the electron repulsion integral (ERI) tensor of degree 4 for Gausslet orbitals from an HDF5 file.",
	},
	{
		.ml_name  = "compute_erida_gausslet_integral",
		.ml_meth  = Py_compute_erida_gausslet_integral,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Evaluate a single electron repulsion integral using the integral diagonal approximation (ERIDA) for Gausslet orbitals.",
	},
	{
		.ml_name  = "compute_sparse_erida_gausslet_integrals",
		.ml_meth  = Py_compute_sparse_erida_gausslet_integrals,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Evaluate the sparse representation of the electron repulsion integrals using the integral diagonal approximation (ERIDA) for Gausslet orbitals.",
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

	if (PyType_Ready(&PySparseERIType) < 0) {
		return NULL;
	}

	if (PyType_Ready(&PySparseERIDAType) < 0) {
		return NULL;
	}

	PyObject* m = PyModule_Create(&module);
	if (m == NULL) {
		return NULL;
	}

	// register sparse ERI type
	Py_INCREF(&PySparseERIType);
	if (PyModule_AddObject(m, "SparseERI", (PyObject*)&PySparseERIType) < 0) {
		Py_DECREF(&PySparseERIType);
		Py_DECREF(m);
		return NULL;
	}

	// register sparse ERIDA type
	Py_INCREF(&PySparseERIDAType);
	if (PyModule_AddObject(m, "SparseERIDA", (PyObject*)&PySparseERIDAType) < 0) {
		Py_DECREF(&PySparseERIDAType);
		Py_DECREF(m);
		return NULL;
	}

	return m;
}
