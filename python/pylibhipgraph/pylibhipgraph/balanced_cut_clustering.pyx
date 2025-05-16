# Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3


from pylibhipgraph._hipgraph_c.array cimport hipgraph_type_erased_device_array_view_t
from pylibhipgraph._hipgraph_c.community_algorithms cimport (
    hipgraph_balanced_cut_clustering,
    hipgraph_clustering_result_free,
    hipgraph_clustering_result_get_clusters,
    hipgraph_clustering_result_get_vertices,
    hipgraph_clustering_result_t,
)
from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.graph cimport hipgraph_graph_t
from pylibhipgraph._hipgraph_c.resource_handle cimport (
    bool_t,
    hipgraph_resource_handle_t,
)
from pylibhipgraph.graphs cimport _GPUGraph
from pylibhipgraph.resource_handle cimport ResourceHandle
from pylibhipgraph.utils cimport assert_success, copy_to_cupy_array


def balanced_cut_clustering(ResourceHandle resource_handle,
                            _GPUGraph graph,
                            num_clusters,
                            num_eigen_vects,
                            evs_tolerance,
                            evs_max_iter,
                            kmean_tolerance,
                            kmean_max_iter,
                            bool_t do_expensive_check
                            ):
    """
    Compute a clustering/partitioning of the given graph using the spectral
    balanced cut method.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph
        The input graph.

    num_clusters : size_t
        Specifies the number of clusters to find, must be greater than 1.

    num_eigen_vects : size_t
        Specifies the number of eigenvectors to use. Must be lower or equal to
        num_clusters.

    evs_tolerance: double
        Specifies the tolerance to use in the eigensolver.

    evs_max_iter: size_t
        Specifies the maximum number of iterations for the eigensolver.

    kmean_tolerance: double
        Specifies the tolerance to use in the k-means solver.

    kmean_max_iter: size_t
        Specifies the maximum number of iterations for the k-means solver.

    do_expensive_check : bool_t
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.

    Returns
    -------
    A tuple containing the clustering vertices, clusters

    Examples
    --------
    >>> import pylibhipgraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 2], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 2, 0], dtype=numpy.int32)
    >>> weights = cupy.asarray([1.0, 1.0, 1.0], dtype=numpy.float32)
    >>> resource_handle = pylibhipgraph.ResourceHandle()
    >>> graph_props = pylibhipgraph.GraphProperties(
    ...     is_symmetric=True, is_multigraph=False)
    >>> G = pylibhipgraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=True, renumber=False, do_expensive_check=False)
    >>> (vertices, clusters) = pylibhipgraph.balanced_cut_clustering(
    ...     resource_handle, G, num_clusters=5, num_eigen_vects=2, evs_tolerance=0.00001
    ...     evs_max_iter=100, kmean_tolerance=0.00001, kmean_max_iter=100)
    # FIXME: Fix docstring results.
    >>> vertices
    ############
    >>> clusters
    ############

    """

    cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef hipgraph_graph_t* c_graph_ptr = graph.c_graph_ptr
    cdef hipgraph_clustering_result_t* result_ptr
    cdef hipgraph_error_code_t error_code
    cdef hipgraph_error_t* error_ptr

    error_code = hipgraph_balanced_cut_clustering(c_resource_handle_ptr,
                                                 c_graph_ptr,
                                                 num_clusters,
                                                 num_eigen_vects,
                                                 evs_tolerance,
                                                 evs_max_iter,
                                                 kmean_tolerance,
                                                 kmean_max_iter,
                                                 do_expensive_check,
                                                 &result_ptr,
                                                 &error_ptr)
    assert_success(error_code, error_ptr, "hipgraph_balanced_cut_clustering")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef hipgraph_type_erased_device_array_view_t* vertices_ptr = \
        hipgraph_clustering_result_get_vertices(result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* clusters_ptr = \
        hipgraph_clustering_result_get_clusters(result_ptr)

    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    cupy_clusters = copy_to_cupy_array(c_resource_handle_ptr, clusters_ptr)

    hipgraph_clustering_result_free(result_ptr)

    return (cupy_vertices, cupy_clusters)
