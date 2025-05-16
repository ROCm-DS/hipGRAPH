# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

from libc.stdint cimport uintptr_t
from pylibhipgraph._hipgraph_c.array cimport hipgraph_type_erased_device_array_view_t
from pylibhipgraph._hipgraph_c.core_algorithms cimport (
    hipgraph_core_number,
    hipgraph_core_result_free,
    hipgraph_core_result_get_core_numbers,
    hipgraph_core_result_get_vertices,
    hipgraph_core_result_t,
    hipgraph_k_core_degree_type_t,
)
from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.graph cimport hipgraph_graph_t
from pylibhipgraph._hipgraph_c.resource_handle cimport (
    bool_t,
    data_type_id_t,
    hipgraph_resource_handle_t,
)
from pylibhipgraph.graphs cimport _GPUGraph
from pylibhipgraph.resource_handle cimport ResourceHandle
from pylibhipgraph.utils cimport (
    assert_success,
    copy_to_cupy_array,
    get_c_type_from_numpy_type,
)


def core_number(ResourceHandle resource_handle,
                _GPUGraph graph,
                degree_type,
                bool_t do_expensive_check):
    """
    Computes core number.

    Parameters
    ----------
    resource_handle: ResourceHandle
        Handle to the underlying device and host resource needed for
        referencing data and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    degree_type: str
        This option determines if the core number computation should be based
        on input, output, or both directed edges, with valid values being
        "incoming", "outgoing", and "bidirectional" respectively.
        This option is currently ignored in this release, and setting it will
        result in a warning.

    do_expensive_check: bool
        If True, performs more extensive tests on the inputs to ensure
        validity, at the expense of increased run time.

    Returns
    -------
    A tuple of device arrays, where the first item in the tuple is a device
    array containing the vertices and the second item in the tuple is a device
    array containing the core numbers for the corresponding vertices.

    Examples
    --------
    # FIXME: No example yet

    """
    cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef hipgraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef hipgraph_core_result_t* result_ptr
    cdef hipgraph_error_code_t error_code
    cdef hipgraph_error_t* error_ptr

    degree_type_map = {
        "incoming": hipgraph_k_core_degree_type_t.K_CORE_DEGREE_TYPE_IN,
        "outgoing": hipgraph_k_core_degree_type_t.K_CORE_DEGREE_TYPE_OUT,
        "bidirectional": hipgraph_k_core_degree_type_t.K_CORE_DEGREE_TYPE_INOUT}

    error_code = hipgraph_core_number(c_resource_handle_ptr,
                                     c_graph_ptr,
                                     degree_type_map[degree_type],
                                     do_expensive_check,
                                     &result_ptr,
                                     &error_ptr)
    assert_success(error_code, error_ptr, "hipgraph_core_number")

    cdef hipgraph_type_erased_device_array_view_t* vertices_ptr = \
        hipgraph_core_result_get_vertices(result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* values_ptr = \
        hipgraph_core_result_get_core_numbers(result_ptr)

    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    cupy_values = copy_to_cupy_array(c_resource_handle_ptr, values_ptr)

    hipgraph_core_result_free(result_ptr)

    return (cupy_vertices, cupy_values)
