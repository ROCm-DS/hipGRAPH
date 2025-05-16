# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

from libc.stdint cimport uintptr_t

import warnings

from pylibhipgraph._hipgraph_c.array cimport hipgraph_type_erased_device_array_view_t
from pylibhipgraph._hipgraph_c.core_algorithms cimport (
    hipgraph_core_result_create,
    hipgraph_core_result_free,
    hipgraph_core_result_t,
    hipgraph_k_core,
    hipgraph_k_core_degree_type_t,
    hipgraph_k_core_result_free,
    hipgraph_k_core_result_get_dst_vertices,
    hipgraph_k_core_result_get_src_vertices,
    hipgraph_k_core_result_get_weights,
    hipgraph_k_core_result_t,
)
from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.graph cimport hipgraph_graph_t
from pylibhipgraph._hipgraph_c.resource_handle cimport (
    bool_t,
    hipgraph_resource_handle_t,
)
from pylibhipgraph.graphs cimport _GPUGraph
from pylibhipgraph.resource_handle cimport ResourceHandle
from pylibhipgraph.utils cimport (
    assert_success,
    copy_to_cupy_array,
    create_hipgraph_type_erased_device_array_view_from_py_obj,
)


def k_core(ResourceHandle resource_handle,
           _GPUGraph graph,
           size_t k,
           degree_type,
           core_result,
           bool_t do_expensive_check):
    """
    Compute the k-core of the graph G
    A k-core of a graph is a maximal subgraph that
    contains nodes of degree k or more. This call does not support a graph
    with self-loops and parallel edges.

    Parameters
    ----------
    resource_handle: ResourceHandle
        Handle to the underlying device and host resource needed for
        referencing data and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    k : size_t (default=None)
        Order of the core. This value must not be negative. If set to None
        the main core is returned.

    degree_type: str
        This option determines if the core number computation should be based
        on input, output, or both directed edges, with valid values being
        "incoming", "outgoing", and "bidirectional" respectively.
        This option is currently ignored in this release, and setting it will
        result in a warning.

    core_result : device array type
        Precomputed core number of the nodes of the graph G
        If set to None, the core numbers of the nodes are calculated
        internally.

    do_expensive_check: bool
        If True, performs more extensive tests on the inputs to ensure
        validity, at the expense of increased run time.

    Returns
    -------
    A tuple of device arrays contaning the sources, destinations vertices
    and the weights.

    Examples
    --------
    # FIXME: No example yet

    """
    cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef hipgraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef hipgraph_core_result_t* core_result_ptr
    cdef hipgraph_k_core_result_t* k_core_result_ptr
    cdef hipgraph_error_code_t error_code
    cdef hipgraph_error_t* error_ptr


    degree_type_map = {
        "incoming": hipgraph_k_core_degree_type_t.K_CORE_DEGREE_TYPE_IN,
        "outgoing": hipgraph_k_core_degree_type_t.K_CORE_DEGREE_TYPE_OUT,
        "bidirectional": hipgraph_k_core_degree_type_t.K_CORE_DEGREE_TYPE_INOUT}

    cdef hipgraph_type_erased_device_array_view_t* \
        vertices_view_ptr = \
            create_hipgraph_type_erased_device_array_view_from_py_obj(
                core_result["vertex"])

    cdef hipgraph_type_erased_device_array_view_t* \
        core_numbers_view_ptr = \
            create_hipgraph_type_erased_device_array_view_from_py_obj(
                core_result["values"])

    # Create a core_number result
    error_code = hipgraph_core_result_create(c_resource_handle_ptr,
                                            vertices_view_ptr,
                                            core_numbers_view_ptr,
                                            &core_result_ptr,
                                            &error_ptr)
    assert_success(error_code, error_ptr, "hipgraph_core_result_create")


    # compute k_core
    error_code = hipgraph_k_core(c_resource_handle_ptr,
                                c_graph_ptr,
                                k,
                                degree_type_map[degree_type],
                                core_result_ptr,
                                do_expensive_check,
                                &k_core_result_ptr,
                                &error_ptr)
    assert_success(error_code, error_ptr, "hipgraph_k_core_number")


    cdef hipgraph_type_erased_device_array_view_t* src_vertices_ptr = \
        hipgraph_k_core_result_get_src_vertices(k_core_result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* dst_vertices_ptr = \
        hipgraph_k_core_result_get_dst_vertices(k_core_result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* weights_ptr = \
        hipgraph_k_core_result_get_weights(k_core_result_ptr)

    cupy_src_vertices = copy_to_cupy_array(c_resource_handle_ptr, src_vertices_ptr)
    cupy_dst_vertices = copy_to_cupy_array(c_resource_handle_ptr, dst_vertices_ptr)

    if weights_ptr is not NULL:
        cupy_weights = copy_to_cupy_array(c_resource_handle_ptr, weights_ptr)
    else:
        cupy_weights = None

    hipgraph_k_core_result_free(k_core_result_ptr)
    hipgraph_core_result_free(core_result_ptr)

    return (cupy_src_vertices, cupy_dst_vertices, cupy_weights)
