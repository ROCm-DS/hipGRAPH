# Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

from libc.stdint cimport uintptr_t
from pylibhipgraph._hipgraph_c.array cimport (
    hipgraph_type_erased_device_array_view_free,
    hipgraph_type_erased_device_array_view_t,
)
from pylibhipgraph._hipgraph_c.centrality_algorithms cimport (
    hipgraph_betweenness_centrality,
    hipgraph_centrality_result_free,
    hipgraph_centrality_result_get_values,
    hipgraph_centrality_result_get_vertices,
    hipgraph_centrality_result_t,
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
    assert_CAI_type,
    assert_success,
    copy_to_cupy_array,
    create_hipgraph_type_erased_device_array_view_from_py_obj,
)

from pylibhipgraph.select_random_vertices import select_random_vertices


def betweenness_centrality(ResourceHandle resource_handle,
                           _GPUGraph graph,
                           k,
                           random_state,
                           bool_t normalized,
                           bool_t include_endpoints,
                           bool_t do_expensive_check):
    """
    Compute the betweenness centrality for all vertices of the graph G.
    Betweenness centrality is a measure of the number of shortest paths that
    pass through a vertex.  A vertex with a high betweenness centrality score
    has more paths passing through it and is therefore believed to be more
    important.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    k : int or device array type or None, optional (default=None)
        If k is not None, use k node samples to estimate betweenness.  Higher
        values give better approximation.  If k is a device array type,
        use the content of the list for estimation: the list should contain
        vertex identifiers. If k is None (the default), all the vertices are
        used to estimate betweenness.  Vertices obtained through sampling or
        defined as a list will be used as sources for traversals inside the
        algorithm.

    random_state : int, optional (default=None)
        if k is specified and k is an integer, use random_state to initialize the
        random number generator.
        Using None defaults to a hash of process id, time, and hostname
        If k is either None or list or cudf objects: random_state parameter is
        ignored.

    normalized : bool_t
        Normalization will ensure that values are in [0, 1].

    include_endpoints : bool_t
        If true, include the endpoints in the shortest path counts.

    do_expensive_check : bool_t
        A flag to run expensive checks for input arguments if True.

    Returns
    -------

    Examples
    --------

    """

    if isinstance(k, int):
        # randomly select vertices

        #'select_random_vertices' internally creates a
        # 'pylibhipgraph.random.HipGraphRandomState'
        vertex_list = select_random_vertices(
            resource_handle, graph, random_state, k)
    else:
        vertex_list = k

    cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef hipgraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef hipgraph_centrality_result_t* result_ptr
    cdef hipgraph_error_code_t error_code
    cdef hipgraph_error_t* error_ptr

    cdef hipgraph_type_erased_device_array_view_t* \
        vertex_list_view_ptr = \
            create_hipgraph_type_erased_device_array_view_from_py_obj(
                vertex_list)

    error_code = hipgraph_betweenness_centrality(c_resource_handle_ptr,
                                                c_graph_ptr,
                                                vertex_list_view_ptr,
                                                normalized,
                                                include_endpoints,
                                                do_expensive_check,
                                                &result_ptr,
                                                &error_ptr)
    assert_success(error_code, error_ptr, "hipgraph_betweenness_centrality")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef hipgraph_type_erased_device_array_view_t* vertices_ptr = \
        hipgraph_centrality_result_get_vertices(result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* values_ptr = \
        hipgraph_centrality_result_get_values(result_ptr)

    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    cupy_values = copy_to_cupy_array(c_resource_handle_ptr, values_ptr)

    hipgraph_centrality_result_free(result_ptr)
    hipgraph_type_erased_device_array_view_free(vertex_list_view_ptr)

    return (cupy_vertices, cupy_values)
