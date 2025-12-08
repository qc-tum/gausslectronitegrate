#pragma once

#include <hdf5.h>
#include "util.h"


herr_t get_hdf5_dataset_ndims(hid_t file, const char* name, int* ndims);

herr_t get_hdf5_dataset_dims(hid_t file, const char* name, hsize_t* dims);

herr_t read_hdf5_dataset(hid_t file, const char* name, hid_t mem_type, void* data);

herr_t write_hdf5_dataset(hid_t file, const char* name, int degree, const hsize_t dims[], hid_t mem_type_store, hid_t mem_type_input, const void* data);

herr_t get_hdf5_attribute_dims(hid_t file, const char* name, hsize_t* dims);

herr_t read_hdf5_attribute(hid_t file, const char* name, hid_t mem_type, void* data);

herr_t write_hdf5_scalar_attribute(hid_t file, const char* name, hid_t mem_type_store, hid_t mem_type_input, const void* data);

herr_t write_hdf5_vector_attribute(hid_t file, const char* name, hid_t mem_type_store, hid_t mem_type_input, const glong length, const void* data);
