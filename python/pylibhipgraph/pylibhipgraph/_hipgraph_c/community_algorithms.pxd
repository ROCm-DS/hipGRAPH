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
from pylibhipgraph._hipgraph_c.graph_functions cimport (
    hipgraph_induced_subgraph_result_t,
)
from pylibhipgraph._hipgraph_c.random cimport hipgraph_rng_state_t
from pylibhipgraph._hipgraph_c.resource_handle cimport (
    bool_t,
    hipgraph_resource_handle_t,
)


cdef extern from "hipgraph/hipgraph_c/community_algorithms.h":
    ###########################################################################
    # triangle_count
    ctypedef struct hipgraph_triangle_count_result_t:
        pass

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_triangle_count_result_get_vertices(
            hipgraph_triangle_count_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_triangle_count_result_get_counts(
            hipgraph_triangle_count_result_t* result
        )

    cdef void \
        hipgraph_triangle_count_result_free(
            hipgraph_triangle_count_result_t* result
        )

    cdef hipgraph_error_code_t \
        hipgraph_triangle_count(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_type_erased_device_array_view_t* start,
            bool_t do_expensive_check,
            hipgraph_triangle_count_result_t** result,
            hipgraph_error_t** error
        )

    ###########################################################################
    # louvain
    ctypedef struct hipgraph_hierarchical_clustering_result_t:
        pass

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_hierarchical_clustering_result_get_vertices(
            hipgraph_hierarchical_clustering_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_hierarchical_clustering_result_get_clusters(
            hipgraph_hierarchical_clustering_result_t* result
        )

    cdef double hipgraph_hierarchical_clustering_result_get_modularity(
        hipgraph_hierarchical_clustering_result_t* result
        )

    cdef void \
        hipgraph_hierarchical_clustering_result_free(
            hipgraph_hierarchical_clustering_result_t* result
        )

    cdef hipgraph_error_code_t \
        hipgraph_louvain(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            size_t max_level,
            double threshold,
            double resolution,
            bool_t do_expensive_check,
            hipgraph_hierarchical_clustering_result_t** result,
            hipgraph_error_t** error
        )

    # extract_ego
    cdef hipgraph_error_code_t \
        hipgraph_extract_ego(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_type_erased_device_array_view_t* source_vertices,
            size_t radius,
            bool_t do_expensive_check,
            hipgraph_induced_subgraph_result_t** result,
            hipgraph_error_t** error
        )

    # leiden
    ctypedef struct hipgraph_hierarchical_clustering_result_t:
        pass

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_hierarchical_clustering_result_get_vertices(
            hipgraph_hierarchical_clustering_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_hierarchical_clustering_result_get_clusters(
            hipgraph_hierarchical_clustering_result_t* result
        )

    cdef double hipgraph_hierarchical_clustering_result_get_modularity(
        hipgraph_hierarchical_clustering_result_t* result
        )

    cdef void \
        hipgraph_hierarchical_clustering_result_free(
            hipgraph_hierarchical_clustering_result_t* result
        )

    cdef hipgraph_error_code_t \
        hipgraph_leiden(
            const hipgraph_resource_handle_t* handle,
            hipgraph_rng_state_t* rng_state,
            hipgraph_graph_t* graph,
            size_t max_level,
            double resolution,
            double theta,
            bool_t do_expensive_check,
            hipgraph_hierarchical_clustering_result_t** result,
            hipgraph_error_t** error
        )
    ###########################################################################
    # ECG
    cdef hipgraph_error_code_t \
        hipgraph_ecg(
            const hipgraph_resource_handle_t* handle,
            hipgraph_rng_state_t* rng_state,
            hipgraph_graph_t* graph,
            double min_weight,
            size_t ensemble_size,
            size_t max_level,
            double threshold,
            double resolution,
            bool_t do_expensive_check,
            hipgraph_hierarchical_clustering_result_t** result,
            hipgraph_error_t** error
        )

    ###########################################################################
    # Clustering
    ctypedef struct hipgraph_clustering_result_t:
        pass

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_clustering_result_get_vertices(
            hipgraph_clustering_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_clustering_result_get_clusters(
            hipgraph_clustering_result_t* result
        )

    cdef void \
        hipgraph_clustering_result_free(
            hipgraph_clustering_result_t* result
        )

    # Balanced cut clustering
    cdef hipgraph_error_code_t \
        hipgraph_balanced_cut_clustering(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            size_t n_clusters,
            size_t n_eigenvectors,
            double evs_tolerance,
            int evs_max_iterations,
            double k_means_tolerance,
            int k_means_max_iterations,
            bool_t do_expensive_check,
            hipgraph_clustering_result_t** result,
            hipgraph_error_t** error
        )

    # Spectral modularity maximization
    cdef hipgraph_error_code_t \
        hipgraph_spectral_modularity_maximization(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            size_t n_clusters,
            size_t n_eigenvectors,
            double evs_tolerance,
            int evs_max_iterations,
            double k_means_tolerance,
            int k_means_max_iterations,
            bool_t do_expensive_check,
            hipgraph_clustering_result_t** result,
            hipgraph_error_t** error
        )

    # Analyze clustering modularity
    cdef hipgraph_error_code_t \
        hipgraph_analyze_clustering_modularity(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            size_t n_clusters,
            const hipgraph_type_erased_device_array_view_t* vertices,
            const hipgraph_type_erased_device_array_view_t* clusters,
            double* score,
            hipgraph_error_t** error
        )

    # Analyze clustering edge cut
    cdef hipgraph_error_code_t \
        hipgraph_analyze_clustering_edge_cut(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            size_t n_clusters,
            const hipgraph_type_erased_device_array_view_t* vertices,
            const hipgraph_type_erased_device_array_view_t* clusters,
            double* score,
            hipgraph_error_t** error
        )

    # Analyze clustering ratio cut
    cdef hipgraph_error_code_t \
        hipgraph_analyze_clustering_ratio_cut(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            size_t n_clusters,
            const hipgraph_type_erased_device_array_view_t* vertices,
            const hipgraph_type_erased_device_array_view_t* clusters,
            double* score,
            hipgraph_error_t** error
        )

    ###########################################################################
    # K truss
    cdef hipgraph_error_code_t \
        hipgraph_k_truss_subgraph(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            size_t k,
            bool_t do_expensive_check,
            hipgraph_induced_subgraph_result_t** result,
            hipgraph_error_t** error)
