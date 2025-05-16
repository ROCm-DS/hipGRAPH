# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

from libc.stdint cimport uintptr_t
from libc.stdio cimport printf
from pylibhipgraph._hipgraph_c.array cimport (
    hipgraph_type_erased_device_array_view_free,
    hipgraph_type_erased_device_array_view_t,
)
from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.graph cimport hipgraph_graph_t
from pylibhipgraph._hipgraph_c.graph_functions cimport (
    hipgraph_vertex_pairs_free,
    hipgraph_vertex_pairs_get_first,
    hipgraph_vertex_pairs_get_second,
    hipgraph_vertex_pairs_t,
)
from pylibhipgraph._hipgraph_c.resource_handle cimport (
    bool_t,
    hipgraph_resource_handle_t,
)
from pylibhipgraph._hipgraph_c.similarity_algorithms cimport (
    hipgraph_all_pairs_jaccard_coefficients,
    hipgraph_similarity_result_free,
    hipgraph_similarity_result_get_similarity,
    hipgraph_similarity_result_get_vertex_pairs,
    hipgraph_similarity_result_t,
)
from pylibhipgraph.graphs cimport _GPUGraph
from pylibhipgraph.resource_handle cimport ResourceHandle
from pylibhipgraph.utils cimport (
    SIZE_MAX,
    assert_success,
    copy_to_cupy_array,
    create_hipgraph_type_erased_device_array_view_from_py_obj,
)


def all_pairs_jaccard_coefficients(ResourceHandle resource_handle,
        _GPUGraph graph,
        vertices,
        bool_t use_weight,
        topk,
        bool_t do_expensive_check):
    """
    Perform All-Pairs Jaccard similarity computation.

    Note that Jaccard similarity must run on a symmetric graph.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    vertices : cudf.Series or None
        Vertex list to compute all-pairs. If None, then compute based
            on all vertices in the graph.

    use_weight : bool, optional
        If set to True, then compute weighted jaccard_coefficients(
            the input graph must be weighted in that case).
        Otherwise, compute non-weighted jaccard_coefficients

    topk : size_t
        Specify the number of answers to return otherwise will return all values.


    do_expensive_check : bool
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.

    Returns
    -------
    A tuple of device arrays containing the vertex pairs with
    their corresponding Jaccard coefficient scores.

    Examples
    --------
    # FIXME: No example yet

    """

    if topk is None:
        topk = SIZE_MAX

    cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef hipgraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef hipgraph_similarity_result_t* result_ptr
    cdef hipgraph_error_code_t error_code
    cdef hipgraph_error_t* error_ptr

    cdef hipgraph_type_erased_device_array_view_t* \
        vertices_view_ptr = \
            create_hipgraph_type_erased_device_array_view_from_py_obj(
                vertices)

    error_code = hipgraph_all_pairs_jaccard_coefficients(c_resource_handle_ptr,
                                              c_graph_ptr,
                                              vertices_view_ptr,
                                              use_weight,
                                              topk,
                                              do_expensive_check,
                                              &result_ptr,
                                              &error_ptr)
    assert_success(error_code, error_ptr, "hipgraph_all_pairs_jaccard_coefficients")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef hipgraph_type_erased_device_array_view_t* similarity_ptr = \
        hipgraph_similarity_result_get_similarity(result_ptr)

    cupy_similarity = copy_to_cupy_array(c_resource_handle_ptr, similarity_ptr)

    cdef hipgraph_vertex_pairs_t* vertex_pairs_ptr = \
        hipgraph_similarity_result_get_vertex_pairs(result_ptr)

    cdef hipgraph_type_erased_device_array_view_t* first_view_ptr = \
        hipgraph_vertex_pairs_get_first(vertex_pairs_ptr)

    cupy_first = copy_to_cupy_array(c_resource_handle_ptr, first_view_ptr)

    cdef hipgraph_type_erased_device_array_view_t* second_view_ptr = \
        hipgraph_vertex_pairs_get_second(vertex_pairs_ptr)

    cupy_second = copy_to_cupy_array(c_resource_handle_ptr, second_view_ptr)

    # Free all pointers
    hipgraph_similarity_result_free(result_ptr)
    hipgraph_vertex_pairs_free(vertex_pairs_ptr)

    hipgraph_type_erased_device_array_view_free(vertices_view_ptr)
    # No need to free 'first_view_ptr' and 'second_view_ptr' as their memory
    # are already deallocated when freeing 'result_ptr'

    return cupy_first, cupy_second, cupy_similarity
