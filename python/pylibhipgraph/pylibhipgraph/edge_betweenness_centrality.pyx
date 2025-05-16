# Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3


from pylibhipgraph._hipgraph_c.array cimport (
    hipgraph_type_erased_device_array_view_free,
    hipgraph_type_erased_device_array_view_t,
)
from pylibhipgraph._hipgraph_c.centrality_algorithms cimport (
    hipgraph_edge_betweenness_centrality,
    hipgraph_edge_centrality_result_free,
    hipgraph_edge_centrality_result_get_dst_vertices,
    hipgraph_edge_centrality_result_get_edge_ids,
    hipgraph_edge_centrality_result_get_src_vertices,
    hipgraph_edge_centrality_result_get_values,
    hipgraph_edge_centrality_result_t,
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

from pylibhipgraph.select_random_vertices import select_random_vertices


def edge_betweenness_centrality(ResourceHandle resource_handle,
                                _GPUGraph graph,
                                k,
                                random_state,
                                bool_t normalized,
                                bool_t do_expensive_check):
    """
    Compute the edge betweenness centrality for all edges of the graph G.
    Betweenness centrality is a measure of the number of shortest paths
    that pass over an edge.  An edge with a high betweenness centrality
    score has more paths passing over it and is therefore believed to be
    more important.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    k : int or device array type or None, optional (default=None)
        If k is not None, use k node samples to estimate the edge betweenness.
        Higher values give better approximation.  If k is a device array type,
        the contents are assumed to be vertex identifiers to be used for estimation.
        If k is None (the default), all the vertices are used to estimate the edge
        betweenness.  Vertices obtained through sampling or defined as a list will
        be used as sources for traversals inside the algorithm.

    random_state : int, optional (default=None)
        if k is specified and k is an integer, use random_state to initialize the
        random number generator.
        Using None defaults to a hash of process id, time, and hostname
        If k is either None or list or cudf objects: random_state parameter is
        ignored.

    normalized : bool_t
        Normalization will ensure that values are in [0, 1].

    do_expensive_check : bool_t
        A flag to run expensive checks for input arguments if True.

    Returns
    -------
    A tuple of device arrays corresponding to the sources, destinations, edge
    betweenness centrality scores and edge ids (if provided).

    array containing the vertices and the second item in the tuple is a device
    array containing the eigenvector centrality scores for the corresponding
    vertices.
    Examples
    --------
    >>> import pylibhipgraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5],
    ...     dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4],
    ...     dtype=numpy.int32)
    >>> edge_ids = cupy.asarray(
    ...     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    ...     dtype=numpy.int32)
    >>> resource_handle = pylibhipgraph.ResourceHandle()
    >>> graph_props = pylibhipgraph.GraphProperties(
    ...     is_symmetric=False, is_multigraph=False)
    >>> G = pylibhipgraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, store_transposed=False,
    ...     renumber=False, do_expensive_check=False, edge_id_array=edge_ids)
    >>> (srcs, dsts, values, edge_ids) = pylibhipgraph.edge_betweenness_centrality(
            resource_handle, G, None, None, True, False)
    >>> srcs
    [0 0 1 1 1 1 2 2 2 3 3 3 4 4 5 5]
    >>> dsts
    [1 2 0 2 3 4 0 1 3 1 2 5 1 5 3 4]
    >>> values
    [0.10555556 0.06111111 0.10555556 0.06666667 0.09444445 0.14444445
     0.06111111 0.06666667 0.09444445 0.09444445 0.09444445 0.12222222
     0.14444445 0.07777778 0.12222222 0.07777778]
    >>> edge_ids
    [ 0 11  8 12  1  2  3  4  5  9 13  6 10  7 14 15]

    """

    if isinstance(k, int):
        # randomly select vertices

        #'select_random_vertices' internally creates a
        # 'pylibhipgraph.random.HipGraphRandomState'
        vertex_list = select_random_vertices(
            resource_handle, graph, random_state, k)
    else:
        # FiXME: Add CAPI check ensuring that k is a cuda array interface
        vertex_list = k

    cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef hipgraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef hipgraph_edge_centrality_result_t* result_ptr
    cdef hipgraph_error_code_t error_code
    cdef hipgraph_error_t* error_ptr

    cdef hipgraph_type_erased_device_array_view_t* \
        vertex_list_view_ptr = \
            create_hipgraph_type_erased_device_array_view_from_py_obj(
                vertex_list)

    error_code = hipgraph_edge_betweenness_centrality(c_resource_handle_ptr,
                                                c_graph_ptr,
                                                vertex_list_view_ptr,
                                                normalized,
                                                do_expensive_check,
                                                &result_ptr,
                                                &error_ptr)
    assert_success(error_code, error_ptr, "hipgraph_edge_betweenness_centrality")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef hipgraph_type_erased_device_array_view_t* src_ptr = \
        hipgraph_edge_centrality_result_get_src_vertices(result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* dst_ptr = \
        hipgraph_edge_centrality_result_get_dst_vertices(result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* values_ptr = \
        hipgraph_edge_centrality_result_get_values(result_ptr)

    if graph.edge_id_view_ptr is NULL and graph.edge_id_view_ptr_ptr is NULL:
        cupy_edge_ids = None
    else:
        edge_ids_ptr = hipgraph_edge_centrality_result_get_edge_ids(result_ptr)
        cupy_edge_ids = copy_to_cupy_array(c_resource_handle_ptr, edge_ids_ptr)


    cupy_src_vertices = copy_to_cupy_array(c_resource_handle_ptr, src_ptr)
    cupy_dst_vertices = copy_to_cupy_array(c_resource_handle_ptr, dst_ptr)
    cupy_values = copy_to_cupy_array(c_resource_handle_ptr, values_ptr)

    hipgraph_edge_centrality_result_free(result_ptr)
    hipgraph_type_erased_device_array_view_free(vertex_list_view_ptr)

    return (cupy_src_vertices, cupy_dst_vertices, cupy_values, cupy_edge_ids)
