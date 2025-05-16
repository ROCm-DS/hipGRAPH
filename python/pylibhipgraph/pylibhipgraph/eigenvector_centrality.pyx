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
    hipgraph_centrality_result_free,
    hipgraph_centrality_result_get_values,
    hipgraph_centrality_result_get_vertices,
    hipgraph_centrality_result_t,
    hipgraph_eigenvector_centrality,
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


def eigenvector_centrality(ResourceHandle resource_handle,
                           _GPUGraph graph,
                           double epsilon,
                           size_t max_iterations,
                           bool_t do_expensive_check):
    """
    Compute the Eigenvector centrality for the nodes of the graph.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    epsilon : double
        Error tolerance to check convergence

    max_iterations: size_t
        Maximum number of Eignevector Centrality iterations

    do_expensive_check : bool_t
        A flag to run expensive checks for input arguments if True.

    Returns
    -------
    A tuple of device arrays, where the first item in the tuple is a device
    array containing the vertices and the second item in the tuple is a device
    array containing the eigenvector centrality scores for the corresponding
    vertices.

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
    ...     store_transposed=True, renumber=False, do_expensive_check=False)
    >>> (vertices, values) = pylibhipgraph.eigenvector_centrality(
                                resource_handle, G, 1e-6, 1000, False)

    """

    cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef hipgraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef hipgraph_centrality_result_t* result_ptr
    cdef hipgraph_error_code_t error_code
    cdef hipgraph_error_t* error_ptr

    error_code = hipgraph_eigenvector_centrality(c_resource_handle_ptr,
                                                c_graph_ptr,
                                                epsilon,
                                                max_iterations,
                                                do_expensive_check,
                                                &result_ptr,
                                                &error_ptr)
    assert_success(error_code, error_ptr, "hipgraph_eigenvector_centrality")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef hipgraph_type_erased_device_array_view_t* vertices_ptr = \
        hipgraph_centrality_result_get_vertices(result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* values_ptr = \
        hipgraph_centrality_result_get_values(result_ptr)

    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    cupy_values = copy_to_cupy_array(c_resource_handle_ptr, values_ptr)

    hipgraph_centrality_result_free(result_ptr)

    return (cupy_vertices, cupy_values)
