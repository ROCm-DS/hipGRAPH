# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
    hipgraph_centrality_result_converged,
    hipgraph_centrality_result_free,
    hipgraph_centrality_result_get_values,
    hipgraph_centrality_result_get_vertices,
    hipgraph_centrality_result_t,
    hipgraph_pagerank_allow_nonconvergence,
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

from pylibhipgraph.exceptions import FailedToConvergeError


def pagerank(ResourceHandle resource_handle,
            _GPUGraph graph,
            precomputed_vertex_out_weight_vertices,
            precomputed_vertex_out_weight_sums,
            initial_guess_vertices,
            initial_guess_values,
            double alpha,
            double epsilon,
            size_t max_iterations,
            bool_t do_expensive_check,
            fail_on_nonconvergence=True):
    """
    Find the PageRank score for every vertex in a graph by computing an
    approximation of the Pagerank eigenvector using the power method. The
    number of iterations depends on the properties of the network itself; it
    increases when the tolerance descreases and/or alpha increases toward the
    limiting value of 1.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    precomputed_vertex_out_weight_vertices: device array type
        Subset of vertices of graph for precomputed_vertex_out_weight
        (a performance optimization)

    precomputed_vertex_out_weight_sums : device array type
        Corresponding precomputed sum of outgoing vertices weight
        (a performance optimization)

    initial_guess_vertices : device array type
        Subset of vertices of graph for initial guess for pagerank values
        (a performance optimization)

    initial_guess_values : device array type
        Pagerank values for vertices
        (a performance optimization)

    alpha : double
        The damping factor alpha represents the probability to follow an
        outgoing edge, standard value is 0.85.
        Thus, 1.0-alpha is the probability to “teleport” to a random vertex.
        Alpha should be greater than 0.0 and strictly lower than 1.0.

    epsilon : double
        Set the tolerance the approximation, this parameter should be a small
        magnitude value.
        The lower the tolerance the better the approximation. If this value is
        0.0f, hipGRAPH will use the default value which is 1.0E-5.
        Setting too small a tolerance can lead to non-convergence due to
        numerical roundoff. Usually values between 0.01 and 0.00001 are
        acceptable.

    max_iterations : size_t
        The maximum number of iterations before an answer is returned. This can
        be used to limit the execution time and do an early exit before the
        solver reaches the convergence tolerance.
        If this value is lower or equal to 0 hipGRAPH will use the default
        value, which is 100.

    do_expensive_check : bool_t
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.

    fail_on_nonconvergence : bool (default=True)
        If the solver does not reach convergence, raise an exception if
        fail_on_nonconvergence is True. If fail_on_nonconvergence is False,
        the return value is a tuple of (pagerank, converged) where pagerank is
        a cudf.DataFrame as described below, and converged is a boolean
        indicating if the solver converged (True) or not (False).

    Returns
    -------
    The return value varies based on the value of the fail_on_nonconvergence
    paramter.  If fail_on_nonconvergence is True:

       A tuple of device arrays, where the first item in the tuple is a device
       array containing the vertex identifiers, and the second item is a device
       array containing the pagerank values for the corresponding vertices. For
       example, the vertex identifier at the ith element of the vertex array
       has the pagerank value of the ith element in the pagerank array.

    If fail_on_nonconvergence is False:

       A three-tuple where the first two items are the device arrays described
       above, and the third is a bool indicating if the solver converged (True)
       or not (False).

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
    >>> (vertices, pageranks) = pylibhipgraph.pagerank(
    ...     resource_handle, G, None, None, None, None, alpha=0.85,
    ...     epsilon=1.0e-6, max_iterations=500, do_expensive_check=False)
    >>> vertices
    array([0, 1, 2, 3], dtype=int32)
    >>> pageranks
    array([0.11615585, 0.21488841, 0.2988108 , 0.3701449 ], dtype=float32)
    """

    # FIXME: import these modules here for now until a better pattern can be
    # used for optional imports (perhaps 'import_optional()' from hipgraph), or
    # these are made hard dependencies.
    try:
        import cupy
    except ModuleNotFoundError:
        raise RuntimeError("pagerank requires the cupy package, which could "
                           "not be imported")
    try:
        import numpy
    except ModuleNotFoundError:
        raise RuntimeError("pagerank requires the numpy package, which could "
                           "not be imported")

    cdef hipgraph_type_erased_device_array_view_t* \
        initial_guess_vertices_view_ptr = \
            create_hipgraph_type_erased_device_array_view_from_py_obj(
                initial_guess_vertices)

    cdef hipgraph_type_erased_device_array_view_t* \
        initial_guess_values_view_ptr = \
            create_hipgraph_type_erased_device_array_view_from_py_obj(
                initial_guess_values)

    cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef hipgraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef hipgraph_type_erased_device_array_view_t* \
        precomputed_vertex_out_weight_vertices_view_ptr = \
            create_hipgraph_type_erased_device_array_view_from_py_obj(
                precomputed_vertex_out_weight_vertices)

    # FIXME: assert that precomputed_vertex_out_weight_sums
    # type == weight type
    cdef hipgraph_type_erased_device_array_view_t* \
        precomputed_vertex_out_weight_sums_view_ptr = \
            create_hipgraph_type_erased_device_array_view_from_py_obj(
                precomputed_vertex_out_weight_sums)

    cdef hipgraph_centrality_result_t* result_ptr
    cdef hipgraph_error_code_t error_code
    cdef hipgraph_error_t* error_ptr
    cdef bool_t converged
    cdef hipgraph_type_erased_device_array_view_t* vertices_ptr
    cdef hipgraph_type_erased_device_array_view_t* pageranks_ptr

    error_code = hipgraph_pagerank_allow_nonconvergence(
        c_resource_handle_ptr,
        c_graph_ptr,
        precomputed_vertex_out_weight_vertices_view_ptr,
        precomputed_vertex_out_weight_sums_view_ptr,
        initial_guess_vertices_view_ptr,
        initial_guess_values_view_ptr,
        alpha,
        epsilon,
        max_iterations,
        do_expensive_check,
        &result_ptr,
        &error_ptr)
    assert_success(error_code, error_ptr, "hipgraph_pagerank_allow_nonconvergence")

    converged = hipgraph_centrality_result_converged(result_ptr)

    # Only extract results if necessary
    if (fail_on_nonconvergence is False) or (converged is True):
        # Extract individual device array pointers from result and copy to cupy
        # arrays for returning.
        vertices_ptr = hipgraph_centrality_result_get_vertices(result_ptr)
        pageranks_ptr = hipgraph_centrality_result_get_values(result_ptr)
        cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
        cupy_pageranks = copy_to_cupy_array(c_resource_handle_ptr, pageranks_ptr)

    # Free all pointers
    hipgraph_centrality_result_free(result_ptr)
    if initial_guess_vertices is not None:
        hipgraph_type_erased_device_array_view_free(initial_guess_vertices_view_ptr)
    if initial_guess_values is not None:
        hipgraph_type_erased_device_array_view_free(initial_guess_values_view_ptr)
    if precomputed_vertex_out_weight_vertices is not None:
        hipgraph_type_erased_device_array_view_free(precomputed_vertex_out_weight_vertices_view_ptr)
    if precomputed_vertex_out_weight_sums is not None:
        hipgraph_type_erased_device_array_view_free(precomputed_vertex_out_weight_sums_view_ptr)

    if fail_on_nonconvergence is False:
        return (cupy_vertices, cupy_pageranks, bool(converged))
    else:
        if converged is True:
            return (cupy_vertices, cupy_pageranks)
        else:
            raise FailedToConvergeError
