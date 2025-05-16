# Copyright (c) 2022-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibhipgraph._hipgraph_c.algorithms cimport (
    hipgraph_paths_result_free,
    hipgraph_paths_result_get_distances,
    hipgraph_paths_result_get_predecessors,
    hipgraph_paths_result_get_vertices,
    hipgraph_paths_result_t,
    hipgraph_sssp,
)
from pylibhipgraph._hipgraph_c.array cimport hipgraph_type_erased_device_array_view_t
from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.graph cimport hipgraph_graph_t
from pylibhipgraph._hipgraph_c.resource_handle cimport (
    bool_t,
    data_type_id_t,
    hipgraph_resource_handle_t,
)
from pylibhipgraph.graphs cimport _GPUGraph
from pylibhipgraph.resource_handle cimport ResourceHandle
from pylibhipgraph.utils cimport assert_success, copy_to_cupy_array


def sssp(ResourceHandle resource_handle,
        _GPUGraph graph,
        size_t source,
        double cutoff,
        bool_t compute_predecessors,
        bool_t do_expensive_check):
    """
    Compute the distance and predecessors for shortest paths from the specified
    source to all the vertices in the graph. The returned distances array will
    contain the distance from the source to each vertex in the returned vertex
    array at the same index. The returned predecessors array will contain the
    previous vertex in the shortest path for each vertex in the vertex array at
    the same index. Vertices that are unreachable will have a distance of
    infinity denoted by the maximum value of the data type and the predecessor
    set as -1. The source vertex predecessor will be set to -1. Graphs with
    negative weight cycles are not supported.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    source :
        The vertex identifier of the source vertex.

    cutoff :
        Maximum edge weight sum to consider.

    compute_predecessors : bool
       This parameter must be set to True for this release.

    do_expensive_check : bool
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.

    Returns
    -------
    A 3-tuple, where the first item in the tuple is a device array containing
    the vertex identifiers, the second item is a device array containing the
    distance for each vertex from the source vertex, and the third item is a
    device array containing the vertex identifier of the preceding vertex in the
    path for that vertex. For example, the vertex identifier at the ith element
    of the vertex array has a distance from the source vertex of the ith element
    in the distance array, and the preceding vertex in the path is the ith
    element in the predecessor array.

    Examples
    --------
    >>> import pylibhipgraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 2], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 2, 3], dtype=numpy.int32)
    >>> weights = cupy.asarray([1.0, 1.0, 1.0], dtype=numpy.float32)
    >>> resource_handle = pylibhipgraph.ResourceHandle()
    >>> graph_props = pylibhipgraph.GraphProperties(
    ...     is_symmetric=False, is_multigraph=False)
    >>> G = pylibhipgraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=False, renumber=False, do_expensive_check=False)
    >>> (vertices, distances, predecessors) = pylibhipgraph.sssp(
    ...     resource_handle, G, source=1, cutoff=999,
    ...     compute_predecessors=True, do_expensive_check=False)
    >>> vertices
    array([0, 1, 2, 3], dtype=int32)
    >>> distances
    array([3.4028235e+38, 0.0000000e+00, 1.0000000e+00, 2.0000000e+00],
          dtype=float32)
    >>> predecessors
    array([-1, -1,  1,  2], dtype=int32)
    """

    # FIXME: import these modules here for now until a better pattern can be
    # used for optional imports (perhaps 'import_optional()' from hipgraph), or
    # these are made hard dependencies.
    try:
        import cupy
    except ModuleNotFoundError:
        raise RuntimeError("sssp requires the cupy package, which could not "
                           "be imported")
    try:
        import numpy
    except ModuleNotFoundError:
        raise RuntimeError("sssp requires the numpy package, which could not "
                           "be imported")

    if compute_predecessors is False:
        raise ValueError("compute_predecessors must be True for the current "
                         "release.")

    cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef hipgraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef hipgraph_paths_result_t* result_ptr
    cdef hipgraph_error_code_t error_code
    cdef hipgraph_error_t* error_ptr

    error_code = hipgraph_sssp(c_resource_handle_ptr,
                              c_graph_ptr,
                              source,
                              cutoff,
                              compute_predecessors,
                              do_expensive_check,
                              &result_ptr,
                              &error_ptr)
    assert_success(error_code, error_ptr, "hipgraph_sssp")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef hipgraph_type_erased_device_array_view_t* vertices_ptr = \
        hipgraph_paths_result_get_vertices(result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* distances_ptr = \
        hipgraph_paths_result_get_distances(result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* predecessors_ptr = \
        hipgraph_paths_result_get_predecessors(result_ptr)

    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    cupy_distances = copy_to_cupy_array(c_resource_handle_ptr, distances_ptr)
    cupy_predecessors = copy_to_cupy_array(c_resource_handle_ptr,
                                           predecessors_ptr)

    hipgraph_paths_result_free(result_ptr)

    return (cupy_vertices, cupy_distances, cupy_predecessors)
