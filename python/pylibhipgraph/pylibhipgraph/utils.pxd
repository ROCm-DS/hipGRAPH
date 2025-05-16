# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibhipgraph._hipgraph_c.array cimport hipgraph_type_erased_device_array_view_t
from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.resource_handle cimport (
    data_type_id_t,
    hipgraph_resource_handle_t,
)


cdef assert_success(hipgraph_error_code_t code,
                    hipgraph_error_t* err,
                    api_name)

cdef assert_CAI_type(obj, var_name, allow_None=*)

cdef assert_AI_type(obj, var_name, allow_None=*)

cdef get_numpy_type_from_c_type(data_type_id_t c_type)

cdef get_c_type_from_numpy_type(numpy_type)

cdef get_c_weight_type_from_numpy_edge_ids_type(numpy_type)

cdef get_numpy_edge_ids_type_from_c_weight_type(data_type_id_t c_type)

cdef copy_to_cupy_array(
   hipgraph_resource_handle_t* c_resource_handle_ptr,
   hipgraph_type_erased_device_array_view_t* device_array_view_ptr)

cdef copy_to_cupy_array_ids(
   hipgraph_resource_handle_t* c_resource_handle_ptr,
   hipgraph_type_erased_device_array_view_t* device_array_view_ptr)

cdef hipgraph_type_erased_device_array_view_t* \
    create_hipgraph_type_erased_device_array_view_from_py_obj(python_obj)

cdef create_cupy_array_view_for_device_ptr(
    hipgraph_type_erased_device_array_view_t* device_array_view_ptr,
    owning_py_object)

cdef extern from "stdint.h":
    size_t SIZE_MAX
