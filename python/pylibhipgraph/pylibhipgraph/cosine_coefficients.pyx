# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

from cython.operator cimport dereference
from libc.stdint cimport uintptr_t
from libc.stdio cimport printf
from pylibhipgraph._hipgraph_c.array cimport (
    hipgraph_type_erased_device_array_view_free,
    hipgraph_type_erased_device_array_view_t,
)
from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.graph cimport hipgraph_graph_t
from pylibhipgraph._hipgraph_c.graph_functions cimport (
    hipgraph_create_vertex_pairs,
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
    hipgraph_cosine_similarity_coefficients,
    hipgraph_similarity_result_free,
    hipgraph_similarity_result_get_similarity,
    hipgraph_similarity_result_t,
)
from pylibhipgraph.graphs cimport _GPUGraph
from pylibhipgraph.resource_handle cimport ResourceHandle
from pylibhipgraph.utils cimport (
    assert_success,
    copy_to_cupy_array,
    create_hipgraph_type_erased_device_array_view_from_py_obj,
)


def cosine_coefficients(ResourceHandle resource_handle,
        _GPUGraph graph,
        first,
        second,
        bool_t use_weight,
        bool_t do_expensive_check):
    """
    Compute the Cosine coefficients for the specified vertex_pairs.

    Note that Cosine similarity must run on a symmetric graph.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    first :
        Source of the vertex pair.

    second :
        Destination of the vertex pair.

    use_weight : bool, optional
        If set to True, the  compute weighted cosine_coefficients(
            the input graph must be weighted in that case).
        Otherwise, computed un-weighted cosine_coefficients

    do_expensive_check : bool
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.

    Returns
    -------
    A tuple of device arrays containing the vertex pairs with
    their corresponding Cosine coefficient scores.

    Examples
    --------
    # FIXME: No example yet

    """

    cdef hipgraph_vertex_pairs_t* vertex_pairs_ptr

    cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef hipgraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef hipgraph_similarity_result_t* result_ptr
    cdef hipgraph_error_code_t error_code
    cdef hipgraph_error_t* error_ptr

    # 'first' is a required parameter
    cdef hipgraph_type_erased_device_array_view_t* \
        first_view_ptr = \
            create_hipgraph_type_erased_device_array_view_from_py_obj(
                first)

    # 'second' is a required parameter
    cdef hipgraph_type_erased_device_array_view_t* \
        second_view_ptr = \
            create_hipgraph_type_erased_device_array_view_from_py_obj(
                second)

    error_code = hipgraph_create_vertex_pairs(c_resource_handle_ptr,
                                             c_graph_ptr,
                                             first_view_ptr,
                                             second_view_ptr,
                                             &vertex_pairs_ptr,
                                             &error_ptr)
    assert_success(error_code, error_ptr, "vertex_pairs")

    error_code = hipgraph_cosine_similarity_coefficients(c_resource_handle_ptr,
                                              c_graph_ptr,
                                              vertex_pairs_ptr,
                                              use_weight,
                                              do_expensive_check,
                                              &result_ptr,
                                              &error_ptr)
    assert_success(error_code, error_ptr, "hipgraph_cosine_similarity_coefficients")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef hipgraph_type_erased_device_array_view_t* similarity_ptr = \
        hipgraph_similarity_result_get_similarity(result_ptr)

    cupy_similarity = copy_to_cupy_array(c_resource_handle_ptr, similarity_ptr)

    cdef hipgraph_type_erased_device_array_view_t* first_ptr = \
        hipgraph_vertex_pairs_get_first(vertex_pairs_ptr)

    cupy_first = copy_to_cupy_array(c_resource_handle_ptr, first_ptr)

    cdef hipgraph_type_erased_device_array_view_t* second_ptr = \
        hipgraph_vertex_pairs_get_second(vertex_pairs_ptr)

    cupy_second = copy_to_cupy_array(c_resource_handle_ptr, second_ptr)

    # Free all pointers
    hipgraph_similarity_result_free(result_ptr)
    hipgraph_vertex_pairs_free(vertex_pairs_ptr)

    hipgraph_type_erased_device_array_view_free(first_view_ptr)
    hipgraph_type_erased_device_array_view_free(second_view_ptr)

    return cupy_first, cupy_second, cupy_similarity
