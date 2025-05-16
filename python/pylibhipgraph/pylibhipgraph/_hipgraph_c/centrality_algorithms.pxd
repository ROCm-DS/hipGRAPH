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
from pylibhipgraph._hipgraph_c.resource_handle cimport (
    bool_t,
    hipgraph_resource_handle_t,
)


cdef extern from "hipgraph/hipgraph_c/centrality_algorithms.h":
    ###########################################################################
    # pagerank
    ctypedef struct hipgraph_centrality_result_t:
        pass

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_centrality_result_get_vertices(
            hipgraph_centrality_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_centrality_result_get_values(
            hipgraph_centrality_result_t* result
        )

    cdef size_t \
        hipgraph_centrality_result_get_num_iterations(
            hipgraph_centrality_result_t* result
        )

    cdef bool_t \
        hipgraph_centrality_result_converged(
            hipgraph_centrality_result_t* result
        )

    cdef void \
        hipgraph_centrality_result_free(
            hipgraph_centrality_result_t* result
        )

    cdef hipgraph_error_code_t \
        hipgraph_pagerank(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices,
            const hipgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
            const hipgraph_type_erased_device_array_view_t* initial_guess_vertices,
            const hipgraph_type_erased_device_array_view_t* initial_guess_values,
            double alpha,
            double epsilon,
            size_t max_iterations,
            bool_t do_expensive_check,
            hipgraph_centrality_result_t** result,
            hipgraph_error_t** error
        )

    cdef hipgraph_error_code_t \
        hipgraph_pagerank_allow_nonconvergence(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices,
            const hipgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
            const hipgraph_type_erased_device_array_view_t* initial_guess_vertices,
            const hipgraph_type_erased_device_array_view_t* initial_guess_values,
            double alpha,
            double epsilon,
            size_t max_iterations,
            bool_t do_expensive_check,
            hipgraph_centrality_result_t** result,
            hipgraph_error_t** error
        )

    cdef hipgraph_error_code_t \
        hipgraph_personalized_pagerank(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices,
            const hipgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
            const hipgraph_type_erased_device_array_view_t* initial_guess_vertices,
            const hipgraph_type_erased_device_array_view_t* initial_guess_values,
            const hipgraph_type_erased_device_array_view_t* personalization_vertices,
            const hipgraph_type_erased_device_array_view_t* personalization_values,
            double alpha,
            double epsilon,
            size_t max_iterations,
            bool_t do_expensive_check,
            hipgraph_centrality_result_t** result,
            hipgraph_error_t** error
        )

    cdef hipgraph_error_code_t \
        hipgraph_personalized_pagerank_allow_nonconvergence(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices,
            const hipgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
            const hipgraph_type_erased_device_array_view_t* initial_guess_vertices,
            const hipgraph_type_erased_device_array_view_t* initial_guess_values,
            const hipgraph_type_erased_device_array_view_t* personalization_vertices,
            const hipgraph_type_erased_device_array_view_t* personalization_values,
            double alpha,
            double epsilon,
            size_t max_iterations,
            bool_t do_expensive_check,
            hipgraph_centrality_result_t** result,
            hipgraph_error_t** error
        )

    ###########################################################################
    # eigenvector centrality
    cdef hipgraph_error_code_t \
        hipgraph_eigenvector_centrality(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            double epsilon,
            size_t max_iterations,
            bool_t do_expensive_check,
            hipgraph_centrality_result_t** result,
            hipgraph_error_t** error
        )

    ###########################################################################
    # katz centrality
    cdef hipgraph_error_code_t \
        hipgraph_katz_centrality(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_type_erased_device_array_view_t* betas,
            double alpha,
            double beta,
            double epsilon,
            size_t max_iterations,
            bool_t do_expensive_check,
            hipgraph_centrality_result_t** result,
            hipgraph_error_t** error
        )

    ###########################################################################
    # hits
    ctypedef struct hipgraph_hits_result_t:
        pass

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_hits_result_get_vertices(
            hipgraph_hits_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_hits_result_get_hubs(
            hipgraph_hits_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_hits_result_get_authorities(
            hipgraph_hits_result_t* result
        )

    cdef void \
        hipgraph_hits_result_free(
            hipgraph_hits_result_t* result
        )

    cdef hipgraph_error_code_t \
        hipgraph_hits(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            double tol,
            size_t max_iter,
            const hipgraph_type_erased_device_array_view_t* initial_hubs_guess_vertices,
            const hipgraph_type_erased_device_array_view_t* initial_hubs_guess_values,
            bool_t normalized,
            bool_t do_expensive_check,
            hipgraph_hits_result_t** result,
            hipgraph_error_t** error
        )

    ###########################################################################
    # betweenness centrality

    cdef hipgraph_error_code_t \
        hipgraph_betweenness_centrality(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_type_erased_device_array_view_t* vertex_list,
            bool_t normalized,
            bool_t include_endpoints,
            bool_t do_expensive_check,
            hipgraph_centrality_result_t** result,
            hipgraph_error_t** error
        )

    ###########################################################################
    # edge betweenness centrality

    ctypedef struct hipgraph_edge_centrality_result_t:
        pass

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_edge_centrality_result_get_src_vertices(
            hipgraph_edge_centrality_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_edge_centrality_result_get_dst_vertices(
            hipgraph_edge_centrality_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_edge_centrality_result_get_edge_ids(
            hipgraph_edge_centrality_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_edge_centrality_result_get_values(
            hipgraph_edge_centrality_result_t* result
        )

    cdef void \
        hipgraph_edge_centrality_result_free(
            hipgraph_edge_centrality_result_t* result
        )

    cdef hipgraph_error_code_t \
        hipgraph_edge_betweenness_centrality(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_type_erased_device_array_view_t* vertex_list,
            bool_t normalized,
            bool_t do_expensive_check,
            hipgraph_edge_centrality_result_t** result,
            hipgraph_error_t** error
        )
