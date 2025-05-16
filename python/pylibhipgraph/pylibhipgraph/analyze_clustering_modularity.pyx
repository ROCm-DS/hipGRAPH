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
from pylibhipgraph._hipgraph_c.community_algorithms cimport (
    hipgraph_analyze_clustering_modularity,
    hipgraph_clustering_result_t,
)
from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.graph cimport hipgraph_graph_t
from pylibhipgraph._hipgraph_c.resource_handle cimport hipgraph_resource_handle_t
from pylibhipgraph.graphs cimport _GPUGraph
from pylibhipgraph.resource_handle cimport ResourceHandle
from pylibhipgraph.utils cimport (
    assert_success,
    create_hipgraph_type_erased_device_array_view_from_py_obj,
)


def analyze_clustering_modularity(ResourceHandle resource_handle,
                                  _GPUGraph graph,
                                  size_t num_clusters,
                                  vertex,
                                  cluster,
                                  ):
    """
    Compute modularity score of the specified clustering.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph
        The input graph.

    num_clusters : size_t
        Specifies the number of clusters to find, must be greater than 1.

    vertex : device array type
        Vertex ids from the clustering to analyze.

    cluster : device array type
        Cluster ids from the clustering to analyze.

    Returns
    -------
    The modularity score of the specified clustering.

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
    >>> (vertex, cluster) = pylibhipgraph.spectral_modularity_maximization(
    ...     resource_handle, G, num_clusters=5, num_eigen_vects=2, evs_tolerance=0.00001
    ...     evs_max_iter=100, kmean_tolerance=0.00001, kmean_max_iter=100)
        # FIXME: Fix docstring result.
    >>> vertices
    ############
    >>> clusters
    ############
    >>> score = pylibhipgraph.analyze_clustering_modularity(
    ...     resource_handle, G, num_clusters=5, vertex=vertex, cluster=cluster)
    >>> score
    ############


    """

    cdef double score = 0

    cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef hipgraph_graph_t* c_graph_ptr = graph.c_graph_ptr
    cdef hipgraph_clustering_result_t* result_ptr
    cdef hipgraph_error_code_t error_code
    cdef hipgraph_error_t* error_ptr

    cdef hipgraph_type_erased_device_array_view_t* \
        vertex_view_ptr = \
            create_hipgraph_type_erased_device_array_view_from_py_obj(
                vertex)

    cdef hipgraph_type_erased_device_array_view_t* \
        cluster_view_ptr = \
            create_hipgraph_type_erased_device_array_view_from_py_obj(
                cluster)


    error_code = hipgraph_analyze_clustering_modularity(c_resource_handle_ptr,
                                                       c_graph_ptr,
                                                       num_clusters,
                                                       vertex_view_ptr,
                                                       cluster_view_ptr,
                                                       &score,
                                                       &error_ptr)
    assert_success(error_code, error_ptr, "hipgraph_analyze_clustering_modularity")

    if vertex is not None:
        hipgraph_type_erased_device_array_view_free(vertex_view_ptr)
    if cluster is not None:
        hipgraph_type_erased_device_array_view_free(cluster_view_ptr)

    return score
