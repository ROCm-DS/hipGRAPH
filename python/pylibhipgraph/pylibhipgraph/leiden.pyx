# Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3


from pylibhipgraph._hipgraph_c.array cimport hipgraph_type_erased_device_array_view_t
from pylibhipgraph._hipgraph_c.community_algorithms cimport (
    hipgraph_hierarchical_clustering_result_free,
    hipgraph_hierarchical_clustering_result_get_clusters,
    hipgraph_hierarchical_clustering_result_get_modularity,
    hipgraph_hierarchical_clustering_result_get_vertices,
    hipgraph_hierarchical_clustering_result_t,
    hipgraph_leiden,
)
from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.graph cimport hipgraph_graph_t
from pylibhipgraph._hipgraph_c.random cimport hipgraph_rng_state_t
from pylibhipgraph._hipgraph_c.resource_handle cimport (
    bool_t,
    hipgraph_resource_handle_t,
)
from pylibhipgraph.graphs cimport _GPUGraph
from pylibhipgraph.random cimport HipGraphRandomState
from pylibhipgraph.resource_handle cimport ResourceHandle
from pylibhipgraph.utils cimport assert_success, copy_to_cupy_array


def leiden(ResourceHandle resource_handle,
           random_state,
           _GPUGraph graph,
           size_t max_level,
           double resolution,
           double theta,
           bool_t do_expensive_check):
    """
    Compute the modularity optimizing partition of the input graph using the
    Leiden method.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    random_state : int , optional
        Random state to use when generating samples. Optional argument,
        defaults to a hash of process id, time, and hostname.
        (See pylibhipgraph.random.HipGraphRandomState)

    graph : SGGraph or MGGraph
        The input graph.

    max_level: size_t
        This controls the maximum number of levels/iterations of the leiden
        algorithm. When specified the algorithm will terminate after no more
        than the specified number of iterations. No error occurs when the
        algorithm terminates early in this manner.

    resolution: double
        Called gamma in the modularity formula, this changes the size
        of the communities.  Higher resolutions lead to more smaller
        communities, lower resolutions lead to fewer larger communities.
        Defaults to 1.

    theta: double
        Called theta in the Leiden algorithm, this is used to scale
        modularity gain in Leiden refinement phase, to compute
        the probability of joining a random leiden community.

    do_expensive_check : bool_t
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.

    Returns
    -------
    A tuple containing the hierarchical clustering vertices, clusters and
    modularity score

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
    >>> (vertices, clusters, modularity) = pylibhipgraph.Leiden(
                                resource_handle, G, 100, 1., False)
    >>> vertices
    [0, 1, 2]
    >>> clusters
    [0, 0, 0]
    >>> modularity
    0.0

    """
    cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef hipgraph_graph_t* c_graph_ptr = graph.c_graph_ptr
    cdef hipgraph_hierarchical_clustering_result_t* result_ptr
    cdef hipgraph_error_code_t error_code
    cdef hipgraph_error_t* error_ptr

    cg_rng_state = HipGraphRandomState(resource_handle, random_state)

    cdef hipgraph_rng_state_t* rng_state_ptr = cg_rng_state.rng_state_ptr

    error_code = hipgraph_leiden(c_resource_handle_ptr,
                                rng_state_ptr,
                                c_graph_ptr,
                                max_level,
                                resolution,
                                theta,
                                do_expensive_check,
                                &result_ptr,
                                &error_ptr)
    assert_success(error_code, error_ptr, "hipgraph_leiden")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef hipgraph_type_erased_device_array_view_t* vertices_ptr = \
        hipgraph_hierarchical_clustering_result_get_vertices(result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* clusters_ptr = \
        hipgraph_hierarchical_clustering_result_get_clusters(result_ptr)
    cdef double modularity = \
        hipgraph_hierarchical_clustering_result_get_modularity(result_ptr)

    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    cupy_clusters = copy_to_cupy_array(c_resource_handle_ptr, clusters_ptr)

    hipgraph_hierarchical_clustering_result_free(result_ptr)

    return (cupy_vertices, cupy_clusters, modularity)
