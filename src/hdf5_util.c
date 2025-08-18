#include <stdio.h>
#include "hdf5_util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Get the number of dimensions (degree) of an HDF5 dataset.
///
herr_t get_hdf5_dataset_ndims(hid_t file, const char* name, int* ndims)
{
	hid_t dset = H5Dopen(file, name, H5P_DEFAULT);
	if (dset < 0)
	{
		fprintf(stderr, "'H5Dopen' for '%s' failed\n", name);
		return -1;
	}

	hid_t space = H5Dget_space(dset);
	if (space < 0)
	{
		fprintf(stderr, "'H5Dget_space'for '%s' failed\n", name);
		return -1;
	}

	*ndims = H5Sget_simple_extent_ndims(space);

	H5Sclose(space);
	H5Dclose(dset);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Get the dimensions of an HDF5 dataset.
///
herr_t get_hdf5_dataset_dims(hid_t file, const char* name, hsize_t* dims)
{
	hid_t dset = H5Dopen(file, name, H5P_DEFAULT);
	if (dset < 0)
	{
		fprintf(stderr, "'H5Dopen' for '%s' failed\n", name);
		return -1;
	}

	hid_t space = H5Dget_space(dset);
	if (space < 0)
	{
		fprintf(stderr, "'H5Dget_space' for '%s' failed\n", name);
		return -1;
	}

	// get dimensions
	H5Sget_simple_extent_dims(space, dims, NULL);

	H5Sclose(space);
	H5Dclose(dset);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Read an HDF5 dataset from a file.
///
herr_t read_hdf5_dataset(hid_t file, const char* name, hid_t mem_type, void* data)
{
	hid_t dset = H5Dopen(file, name, H5P_DEFAULT);
	if (dset < 0)
	{
		fprintf(stderr, "'H5Dopen' for '%s' failed\n", name);
		return -1;
	}

	herr_t status = H5Dread(dset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	if (status < 0)
	{
		fprintf(stderr, "'H5Dread' for '%s' failed, return value: %d\n", name, status);
		return status;
	}

	H5Dclose(dset);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Write an HDF5 dataset to a file.
///
herr_t write_hdf5_dataset(hid_t file, const char* name, int degree, const hsize_t dims[], hid_t mem_type_store, hid_t mem_type_input, const void* data)
{
	// create dataspace
	hid_t space = H5Screate_simple(degree, dims, NULL);
	if (space < 0)
	{
		fprintf(stderr, "'H5Screate_simple' for '%s' failed\n", name);
		return -1;
	}

	// property list to disable time tracking
	hid_t cplist = H5Pcreate(H5P_DATASET_CREATE);
	herr_t status = H5Pset_obj_track_times(cplist, 0);
	if (status < 0)
	{
		fprintf(stderr, "creating property list failed, return value: %d\n", status);
		return status;
	}

	// create dataset
	hid_t dset = H5Dcreate(file, name, mem_type_store, space, H5P_DEFAULT, cplist, H5P_DEFAULT);
	if (dset < 0)
	{
		fprintf(stderr, "'H5Dcreate' for '%s' failed\n", name);
		return -1;
	}

	// write the data to the dataset
	status = H5Dwrite(dset, mem_type_input, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	if (status < 0)
	{
		fprintf(stderr, "'H5Dwrite' for '%s' failed, return value: %d\n", name, status);
		return status;
	}

	H5Dclose(dset);
	H5Pclose(cplist);
	H5Sclose(space);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Get the dimensions of an HDF5 attribute.
///
herr_t get_hdf5_attribute_dims(hid_t file, const char* name, hsize_t* dims)
{
	hid_t attr = H5Aopen(file, name, H5P_DEFAULT);
	if (attr < 0)
	{
		fprintf(stderr, "'H5Aopen' for '%s' failed\n", name);
		return -1;
	}

	hid_t space = H5Aget_space(attr);
	if (space < 0)
	{
		fprintf(stderr, "'H5Aget_space' for '%s' failed\n", name);
		return -1;
	}

	// get dimensions
	H5Sget_simple_extent_dims(space, dims, NULL);

	H5Sclose(space);
	H5Aclose(attr);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Read an HDF5 attribute from a file.
///
herr_t read_hdf5_attribute(hid_t file, const char* name, hid_t mem_type, void* data)
{
	hid_t attr = H5Aopen(file, name, H5P_DEFAULT);
	if (attr < 0)
	{
		fprintf(stderr, "'H5Aopen' for '%s' failed\n", name);
		return -1;
	}

	herr_t status = H5Aread(attr, mem_type, data);
	if (status < 0)
	{
		fprintf(stderr, "'H5Aread' for '%s' failed, return value: %d\n", name, status);
		return status;
	}

	status = H5Aclose(attr);
	if (status < 0)
	{
		fprintf(stderr, "'H5Aclose' for '%s' failed, return value: %d\n", name, status);
		return -1;
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Write an HDF5 scalar attribute to a file.
///
herr_t write_hdf5_scalar_attribute(hid_t file, const char* name, hid_t mem_type_store, hid_t mem_type_input, const void* data)
{
	hid_t space = H5Screate(H5S_SCALAR);
	if (space < 0)
	{
		fprintf(stderr, "'H5Screate' for '%s' failed\n", name);
		return -1;
	}

	hid_t attr = H5Acreate(file, name, mem_type_store, space, H5P_DEFAULT, H5P_DEFAULT);
	if (attr < 0)
	{
		fprintf(stderr, "'H5Acreate' for '%s' failed\n", name);
		return -1;
	}

	herr_t status = H5Awrite(attr, mem_type_input, data);
	if (status < 0)
	{
		fprintf(stderr, "'H5Awrite' for '%s' failed, return value: %d\n", name, status);
		return status;
	}

	status = H5Aclose(attr);
	if (status < 0)
	{
		fprintf(stderr, "'H5Aclose' for '%s' failed, return value: %d\n", name, status);
		return status;
	}
	status = H5Sclose(space);
	if (status < 0)
	{
		fprintf(stderr, "'H5Sclose' for '%s' failed, return value: %d\n", name, status);
		return status;
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Write an HDF5 vector attribute to a file.
///
herr_t write_hdf5_vector_attribute(hid_t file, const char* name, hid_t mem_type_store, hid_t mem_type_input, const long length, const void* data)
{
	hsize_t dims[1] = { length };
	hid_t space = H5Screate_simple(1, dims, NULL);
	if (space < 0)
	{
		fprintf(stderr, "'H5Screate_simple' for '%s' failed\n", name);
		return -1;
	}

	hid_t attr = H5Acreate(file, name, mem_type_store, space, H5P_DEFAULT, H5P_DEFAULT);
	if (attr < 0)
	{
		fprintf(stderr, "'H5Acreate' for '%s' failed\n", name);
		return -1;
	}

	herr_t status = H5Awrite(attr, mem_type_input, data);
	if (status < 0)
	{
		fprintf(stderr, "'H5Awrite' for '%s' failed, return value: %d\n", name, status);
		return status;
	}

	status = H5Aclose(attr);
	if (status < 0)
	{
		fprintf(stderr, "'H5Aclose' for '%s' failed, return value: %d\n", name, status);
		return status;
	}
	status = H5Sclose(space);
	if (status < 0)
	{
		fprintf(stderr, "'H5Sclose' for '%s' failed, return value: %d\n", name, status);
		return status;
	}

	return 0;
}
