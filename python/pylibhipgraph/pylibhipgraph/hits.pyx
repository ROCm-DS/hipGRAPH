# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

from libc.stdint cimport uintptr_t
from pylibhipgraph._hipgraph_c.array cimport (
    hipgraph_type_erased_device_array_view_create,
    hipgraph_type_erased_device_array_view_free,
    hipgraph_type_erased_device_array_view_t,
)
from pylibhipgraph._hipgraph_c.centrality_algorithms cimport (
    hipgraph_hits,
    hipgraph_hits_result_free,
    hipgraph_hits_result_get_authorities,
    hipgraph_hits_result_get_hubs,
    hipgraph_hits_result_get_vertices,
    hipgraph_hits_result_t,
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
    assert_CAI_type,
    assert_success,
    copy_to_cupy_array,
    get_c_type_from_numpy_type,
)


def hits(ResourceHandle resource_handle,
        _GPUGraph graph,
        double tol,
        size_t max_iter,
        initial_hubs_guess_vertices,
        initial_hubs_guess_values,
        bool_t normalized,
        bool_t do_expensive_check):
    """
    Compute HITS hubs and authorities values for each vertex

    The HITS algorithm computes two numbers for a node.  Authorities
    estimates the node value based on the incoming links.  Hubs estimates
    the node value based on outgoing links.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    tol : float, optional (default=1.0e-5)
        Set the tolerance the approximation, this parameter should be a small
        magnitude value.  This parameter is not currently supported.

    max_iter : int, optional (default=100)
        The maximum number of iterations before an answer is returned.

    initial_hubs_guess_vertices : device array type, optional (default=None)
        Device array containing the pointer to the array of initial hub guess vertices

    initial_hubs_guess_values : device array type, optional (default=None)
        Device array containing the pointer to the array of initial hub guess values

    normalized : bool, optional (default=True)


    do_expensive_check : bool
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.

    Returns
    -------
    A tuple of device arrays, where the third item in the tuple is a device
    array containing the vertex identifiers, the first and second items are device
    arrays containing respectively the hubs and authorities values for the corresponding
    vertices

    Examples
    --------
    # FIXME: No example yet

    """

    cdef uintptr_t cai_initial_hubs_guess_vertices_ptr = <uintptr_t>NULL
    cdef uintptr_t cai_initial_hubs_guess_values_ptr = <uintptr_t>NULL

    cdef hipgraph_type_erased_device_array_view_t* initial_hubs_guess_vertices_view_ptr = NULL
    cdef hipgraph_type_erased_device_array_view_t* initial_hubs_guess_values_view_ptr = NULL

    # FIXME: Add check ensuring that both initial_hubs_guess_vertices
    # and initial_hubs_guess_values are passed when calling only pylibhipgraph HITS.
    # This is already True for hipgraph HITS

    if initial_hubs_guess_vertices is not None:
        assert_CAI_type(initial_hubs_guess_vertices, "initial_hubs_guess_vertices")

        cai_initial_hubs_guess_vertices_ptr = \
        initial_hubs_guess_vertices.__cuda_array_interface__["data"][0]

        initial_hubs_guess_vertices_view_ptr = \
            hipgraph_type_erased_device_array_view_create(
                <void*>cai_initial_hubs_guess_vertices_ptr,
                len(initial_hubs_guess_vertices),
                get_c_type_from_numpy_type(initial_hubs_guess_vertices.dtype))

    if initial_hubs_guess_values is not None:
        assert_CAI_type(initial_hubs_guess_values, "initial_hubs_guess_values")

        cai_initial_hubs_guess_values_ptr = \
        initial_hubs_guess_values.__cuda_array_interface__["data"][0]

        initial_hubs_guess_values_view_ptr = \
            hipgraph_type_erased_device_array_view_create(
                <void*>cai_initial_hubs_guess_values_ptr,
                len(initial_hubs_guess_values),
                get_c_type_from_numpy_type(initial_hubs_guess_values.dtype))

    cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef hipgraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef hipgraph_hits_result_t* result_ptr
    cdef hipgraph_error_code_t error_code
    cdef hipgraph_error_t* error_ptr

    error_code = hipgraph_hits(c_resource_handle_ptr,
                              c_graph_ptr,
                              tol,
                              max_iter,
                              initial_hubs_guess_vertices_view_ptr,
                              initial_hubs_guess_values_view_ptr,
                              normalized,
                              do_expensive_check,
                              &result_ptr,
                              &error_ptr)
    assert_success(error_code, error_ptr, "hipgraph_mg_hits")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef hipgraph_type_erased_device_array_view_t* vertices_ptr = \
        hipgraph_hits_result_get_vertices(result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* hubs_ptr = \
        hipgraph_hits_result_get_hubs(result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* authorities_ptr = \
        hipgraph_hits_result_get_authorities(result_ptr)

    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    cupy_hubs = copy_to_cupy_array(c_resource_handle_ptr, hubs_ptr)
    cupy_authorities = copy_to_cupy_array(c_resource_handle_ptr,
                                          authorities_ptr)

    hipgraph_hits_result_free(result_ptr)

    if initial_hubs_guess_vertices is not None:
        hipgraph_type_erased_device_array_view_free(
            initial_hubs_guess_vertices_view_ptr)

    if initial_hubs_guess_values is not None:
        hipgraph_type_erased_device_array_view_free(
            initial_hubs_guess_values_view_ptr)

    return (cupy_vertices, cupy_hubs, cupy_authorities)
