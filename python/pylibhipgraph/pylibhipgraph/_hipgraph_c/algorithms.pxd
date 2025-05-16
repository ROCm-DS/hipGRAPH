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


cdef extern from "hipgraph/hipgraph_c/algorithms.h":
    ###########################################################################
    # paths and path extraction
    ctypedef struct hipgraph_paths_result_t:
        pass

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_paths_result_get_vertices(
            hipgraph_paths_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_paths_result_get_distances(
            hipgraph_paths_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_paths_result_get_predecessors(
            hipgraph_paths_result_t* result
        )

    cdef void \
        hipgraph_paths_result_free(
            hipgraph_paths_result_t* result
        )

    ctypedef struct hipgraph_extract_paths_result_t:
        pass

    cdef hipgraph_error_code_t \
        hipgraph_extract_paths(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_type_erased_device_array_view_t* sources,
            const hipgraph_paths_result_t* paths_result,
            const hipgraph_type_erased_device_array_view_t* destinations,
            hipgraph_extract_paths_result_t** result,
            hipgraph_error_t** error
        )

    cdef size_t \
        hipgraph_extract_paths_result_get_max_path_length(
            hipgraph_extract_paths_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_extract_paths_result_get_paths(
            hipgraph_extract_paths_result_t* result
        )

    cdef void \
        hipgraph_extract_paths_result_free(
            hipgraph_extract_paths_result_t* result
        )

    ###########################################################################
    # bfs
    cdef hipgraph_error_code_t \
        hipgraph_bfs(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            # FIXME: this may become const
            hipgraph_type_erased_device_array_view_t* sources,
            bool_t direction_optimizing,
            size_t depth_limit,
            bool_t compute_predecessors,
            bool_t do_expensive_check,
            hipgraph_paths_result_t** result,
            hipgraph_error_t** error
        )

    ###########################################################################
    # sssp
    cdef hipgraph_error_code_t \
        hipgraph_sssp(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            size_t source,
            double cutoff,
            bool_t compute_predecessors,
            bool_t do_expensive_check,
            hipgraph_paths_result_t** result,
            hipgraph_error_t** error
        )

    ###########################################################################
    # random_walks
    ctypedef struct hipgraph_random_walk_result_t:
        pass

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_random_walk_result_get_paths(
            hipgraph_random_walk_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_random_walk_result_get_weights(
            hipgraph_random_walk_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_random_walk_result_get_path_sizes(
            hipgraph_random_walk_result_t* result
        )

    cdef size_t \
        hipgraph_random_walk_result_get_max_path_length(
            hipgraph_random_walk_result_t* result
        )

    cdef void \
        hipgraph_random_walk_result_free(
            hipgraph_random_walk_result_t* result
        )

    # node2vec
    cdef hipgraph_error_code_t \
        hipgraph_node2vec(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_type_erased_device_array_view_t* sources,
            size_t max_depth,
            bool_t compress_result,
            double p,
            double q,
            hipgraph_random_walk_result_t** result,
            hipgraph_error_t** error
        )


    ###########################################################################
    # sampling
    ctypedef struct hipgraph_sample_result_t:
        pass

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_sample_result_get_renumber_map(
            const hipgraph_sample_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_sample_result_get_renumber_map_offsets(
            const hipgraph_sample_result_t* result
        )

    # Deprecated, use hipgraph_sample_result_get_majors
    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_sample_result_get_sources(
            const hipgraph_sample_result_t* result
        )

    # Deprecated, use hipgraph_sample_result_get_minors
    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_sample_result_get_destinations(
            const hipgraph_sample_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_sample_result_get_majors(
            const hipgraph_sample_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_sample_result_get_minors(
            const hipgraph_sample_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_sample_result_get_major_offsets(
            const hipgraph_sample_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_sample_result_get_index(
            const hipgraph_sample_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_sample_result_get_edge_weight(
            const hipgraph_sample_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_sample_result_get_edge_id(
            const hipgraph_sample_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_sample_result_get_edge_type(
            const hipgraph_sample_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_sample_result_get_hop(
            const hipgraph_sample_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_sample_result_get_label_hop_offsets(
            const hipgraph_sample_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_sample_result_get_start_labels(
            const hipgraph_sample_result_t* result
        )

    # Deprecated
    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_sample_result_get_offsets(
            const hipgraph_sample_result_t* result
        )

    cdef void \
        hipgraph_sample_result_free(
            const hipgraph_sample_result_t* result
        )

    # testing API - hipgraph_sample_result_t instances are normally created only
    # by sampling algos
    cdef hipgraph_error_code_t \
        hipgraph_test_sample_result_create(
            const hipgraph_resource_handle_t* handle,
            const hipgraph_type_erased_device_array_view_t* srcs,
            const hipgraph_type_erased_device_array_view_t* dsts,
            const hipgraph_type_erased_device_array_view_t* edge_id,
            const hipgraph_type_erased_device_array_view_t* edge_type,
            const hipgraph_type_erased_device_array_view_t* wgt,
            const hipgraph_type_erased_device_array_view_t* hop,
            const hipgraph_type_erased_device_array_view_t* label,
            hipgraph_sample_result_t** result,
            hipgraph_error_t** error
        )

    ctypedef struct hipgraph_sampling_options_t:
        pass

    ctypedef enum hipgraph_prior_sources_behavior_t:
        DEFAULT=0
        CARRY_OVER
        EXCLUDE

    ctypedef enum hipgraph_compression_type_t:
        COO=0
        CSR
        CSC
        DCSR
        DCSC

    cdef hipgraph_error_code_t \
        hipgraph_sampling_options_create(
            hipgraph_sampling_options_t** options,
            hipgraph_error_t** error,
        )

    cdef void \
        hipgraph_sampling_set_renumber_results(
            hipgraph_sampling_options_t* options,
            bool_t value,
        )

    cdef void \
        hipgraph_sampling_set_retain_seeds(
            hipgraph_sampling_options_t* options,
            bool_t value,
        )

    cdef void \
        hipgraph_sampling_set_with_replacement(
            hipgraph_sampling_options_t* options,
            bool_t value,
        )

    cdef void \
        hipgraph_sampling_set_return_hops(
            hipgraph_sampling_options_t* options,
            bool_t value,
        )

    cdef void \
        hipgraph_sampling_set_prior_sources_behavior(
            hipgraph_sampling_options_t* options,
            hipgraph_prior_sources_behavior_t value,
        )

    cdef void \
        hipgraph_sampling_set_dedupe_sources(
            hipgraph_sampling_options_t* options,
            bool_t value,
        )

    cdef void \
        hipgraph_sampling_set_compress_per_hop(
            hipgraph_sampling_options_t* options,
            bool_t value,
        )

    cdef void \
        hipgraph_sampling_set_compression_type(
            hipgraph_sampling_options_t* options,
            hipgraph_compression_type_t value,
        )

    cdef void \
        hipgraph_sampling_options_free(
            hipgraph_sampling_options_t* options,
        )

    # uniform random walks
    cdef hipgraph_error_code_t \
        hipgraph_uniform_random_walks(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_type_erased_device_array_view_t* start_vertices,
            size_t max_length,
            hipgraph_random_walk_result_t** result,
            hipgraph_error_t** error
        )

    # biased random walks
    cdef hipgraph_error_code_t \
        hipgraph_based_random_walks(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_type_erased_device_array_view_t* start_vertices,
            size_t max_length,
            hipgraph_random_walk_result_t** result,
            hipgraph_error_t** error
        )
