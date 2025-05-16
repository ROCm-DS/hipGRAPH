# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibhipgraph._hipgraph_c.array cimport (
    hipgraph_type_erased_device_array_view_t,
    hipgraph_type_erased_host_array_view_t,
)
from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.graph cimport hipgraph_graph_t
from pylibhipgraph._hipgraph_c.graph_functions cimport hipgraph_vertex_pairs_t
from pylibhipgraph._hipgraph_c.resource_handle cimport (
    bool_t,
    hipgraph_resource_handle_t,
)


cdef extern from "hipgraph/hipgraph_c/similarity_algorithms.h":

    ###########################################################################
    ctypedef struct hipgraph_similarity_result_t:
        pass

    cdef hipgraph_vertex_pairs_t* \
        hipgraph_similarity_result_get_vertex_pairs(
            hipgraph_similarity_result_t* result);

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_similarity_result_get_similarity(
            hipgraph_similarity_result_t* result
        )

    cdef void \
        hipgraph_similarity_result_free(
            hipgraph_similarity_result_t* result
        )

    ###########################################################################
    # jaccard coefficients
    cdef hipgraph_error_code_t \
        hipgraph_jaccard_coefficients(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_vertex_pairs_t* vertex_pairs,
            bool_t use_weight,
            bool_t do_expensive_check,
            hipgraph_similarity_result_t** result,
            hipgraph_error_t** error
        )

    ###########################################################################
    # all-pairs jaccard coefficients
    cdef hipgraph_error_code_t \
        hipgraph_all_pairs_jaccard_coefficients(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_type_erased_device_array_view_t* vertices,
            bool_t use_weight,
            size_t topk,
            bool_t do_expensive_check,
            hipgraph_similarity_result_t** result,
            hipgraph_error_t** error
        )

    ###########################################################################
    # sorensen coefficients
    cdef hipgraph_error_code_t \
        hipgraph_sorensen_coefficients(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_vertex_pairs_t* vertex_pairs,
            bool_t use_weight,
            bool_t do_expensive_check,
            hipgraph_similarity_result_t** result,
            hipgraph_error_t** error
        )

    ###########################################################################
    # all-pairs sorensen coefficients
    cdef hipgraph_error_code_t \
        hipgraph_all_pairs_sorensen_coefficients(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_type_erased_device_array_view_t* vertices,
            bool_t use_weight,
            size_t topk,
            bool_t do_expensive_check,
            hipgraph_similarity_result_t** result,
            hipgraph_error_t** error
        )

    ###########################################################################
    # overlap coefficients
    cdef hipgraph_error_code_t \
        hipgraph_overlap_coefficients(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_vertex_pairs_t* vertex_pairs,
            bool_t use_weight,
            bool_t do_expensive_check,
            hipgraph_similarity_result_t** result,
            hipgraph_error_t** error
        )

    ###########################################################################
    # all-pairs overlap coefficients
    cdef hipgraph_error_code_t \
        hipgraph_all_pairs_overlap_coefficients(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_type_erased_device_array_view_t* vertices,
            bool_t use_weight,
            size_t topk,
            bool_t do_expensive_check,
            hipgraph_similarity_result_t** result,
            hipgraph_error_t** error
        )

    ###########################################################################
    # cosine coefficients
    cdef hipgraph_error_code_t \
        hipgraph_cosine_similarity_coefficients(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_vertex_pairs_t* vertex_pairs,
            bool_t use_weight,
            bool_t do_expensive_check,
            hipgraph_similarity_result_t** result,
            hipgraph_error_t** error
        )

    ###########################################################################
    # all-pairs cosine coefficients
    cdef hipgraph_error_code_t \
        hipgraph_all_pairs_cosine_similarity_coefficients(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_type_erased_device_array_view_t* vertices,
            bool_t use_weight,
            size_t topk,
            bool_t do_expensive_check,
            hipgraph_similarity_result_t** result,
            hipgraph_error_t** error
        )
