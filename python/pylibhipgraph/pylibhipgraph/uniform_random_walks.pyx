# Copyright (c) 2022-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

from libc.stdint cimport uintptr_t
from pylibhipgraph._hipgraph_c.algorithms cimport (
    hipgraph_random_walk_result_free,
    hipgraph_random_walk_result_get_max_path_length,
    hipgraph_random_walk_result_get_paths,
    hipgraph_random_walk_result_get_weights,
    hipgraph_random_walk_result_t,
    hipgraph_uniform_random_walks,
)
from pylibhipgraph._hipgraph_c.array cimport (
    hipgraph_type_erased_device_array_view_create,
    hipgraph_type_erased_device_array_view_free,
    hipgraph_type_erased_device_array_view_t,
)
from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.graph cimport hipgraph_graph_t
from pylibhipgraph._hipgraph_c.resource_handle cimport hipgraph_resource_handle_t
from pylibhipgraph.graphs cimport _GPUGraph
from pylibhipgraph.resource_handle cimport ResourceHandle
from pylibhipgraph.utils cimport (
    assert_CAI_type,
    assert_success,
    copy_to_cupy_array,
    get_c_type_from_numpy_type,
)


def uniform_random_walks(ResourceHandle resource_handle,
                         _GPUGraph input_graph,
                         start_vertices,
                         size_t max_length):
    """
    Compute uniform random walks for each nodes in 'start_vertices'

    Parameters
    ----------
    resource_handle: ResourceHandle
        Handle to the underlying device and host resources needed for
        referencing data and running algorithms.

    input_graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    start_vertices: device array type
        Device array containing the list of starting vertices from which
        to run the uniform random walk

    max_length: size_t
        The maximum depth of the uniform random walks


    Returns
    -------
    A tuple containing two device arrays and an size_t which are respectively
    the vertices path, the edge path weights and the maximum path length

    """
    cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef hipgraph_graph_t* c_graph_ptr = input_graph.c_graph_ptr

    assert_CAI_type(start_vertices, "start_vertices")

    cdef hipgraph_random_walk_result_t* result_ptr
    cdef hipgraph_error_code_t error_code
    cdef hipgraph_error_t* error_ptr

    cdef uintptr_t cai_start_ptr = \
        start_vertices.__cuda_array_interface__["data"][0]

    cdef hipgraph_type_erased_device_array_view_t* weights_ptr

    cdef hipgraph_type_erased_device_array_view_t* start_ptr = \
        hipgraph_type_erased_device_array_view_create(
            <void*>cai_start_ptr,
            len(start_vertices),
            get_c_type_from_numpy_type(start_vertices.dtype))

    error_code = hipgraph_uniform_random_walks(
        c_resource_handle_ptr,
        c_graph_ptr,
        start_ptr,
        max_length,
        &result_ptr,
        &error_ptr)
    assert_success(error_code, error_ptr, "hipgraph_uniform_random_walks")

    cdef hipgraph_type_erased_device_array_view_t* path_ptr = \
        hipgraph_random_walk_result_get_paths(result_ptr)

    if input_graph.weights_view_ptr is NULL and input_graph.weights_view_ptr_ptr is NULL:
        cupy_weights = None
    else:
        weights_ptr = hipgraph_random_walk_result_get_weights(result_ptr)
        cupy_weights = copy_to_cupy_array(c_resource_handle_ptr, weights_ptr)

    max_path_length = \
        hipgraph_random_walk_result_get_max_path_length(result_ptr)

    cupy_paths = copy_to_cupy_array(c_resource_handle_ptr, path_ptr)

    hipgraph_random_walk_result_free(result_ptr)
    hipgraph_type_erased_device_array_view_free(start_ptr)

    return (cupy_paths, cupy_weights, max_path_length)
