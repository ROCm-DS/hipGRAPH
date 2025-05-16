# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibhipgraph._hipgraph_c.algorithms cimport (
    hipgraph_sample_result_t,
    hipgraph_sampling_options_t,
)
from pylibhipgraph._hipgraph_c.array cimport (
    hipgraph_type_erased_device_array_t,
    hipgraph_type_erased_device_array_view_t,
    hipgraph_type_erased_host_array_view_t,
)
from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.graph cimport hipgraph_graph_t
from pylibhipgraph._hipgraph_c.random cimport hipgraph_rng_state_t
from pylibhipgraph._hipgraph_c.resource_handle cimport (
    bool_t,
    hipgraph_resource_handle_t,
)

# from pylibhipgraph._hipgraph_c.properties cimport (
#     hipgraph_edge_property_view_t,
# )

cdef extern from "hipgraph/hipgraph_c/sampling_algorithms.h":
    ###########################################################################

    cdef hipgraph_error_code_t hipgraph_uniform_neighbor_sample(
        const hipgraph_resource_handle_t* handle,
        hipgraph_graph_t* graph,
        const hipgraph_type_erased_device_array_view_t* start_vertices,
        const hipgraph_type_erased_device_array_view_t* start_vertex_labels,
        const hipgraph_type_erased_device_array_view_t* label_list,
        const hipgraph_type_erased_device_array_view_t* label_to_comm_rank,
        const hipgraph_type_erased_device_array_view_t* label_offsets,
        const hipgraph_type_erased_host_array_view_t* fan_out,
        hipgraph_rng_state_t* rng_state,
        const hipgraph_sampling_options_t* options,
        bool_t do_expensive_check,
        hipgraph_sample_result_t** result,
        hipgraph_error_t** error
    )

    # cdef hipgraph_error_code_t hipgraph_biased_neighbor_sample(
    #     const hipgraph_resource_handle_t* handle,
    #     hipgraph_graph_t* graph,
    #     const hipgraph_edge_property_view_t* edge_biases,
    #     const hipgraph_type_erased_device_array_view_t* start_vertices,
    #     const hipgraph_type_erased_device_array_view_t* start_vertex_labels,
    #     const hipgraph_type_erased_device_array_view_t* label_list,
    #     const hipgraph_type_erased_device_array_view_t* label_to_comm_rank,
    #     const hipgraph_type_erased_device_array_view_t* label_offsets,
    #     const hipgraph_type_erased_host_array_view_t* fan_out,
    #     hipgraph_rng_state_t* rng_state,
    #     const hipgraph_sampling_options_t* options,
    #     bool_t do_expensive_check,
    #     hipgraph_sample_result_t** result,
    #     hipgraph_error_t** error
    # )

    cdef hipgraph_error_code_t hipgraph_test_uniform_neighborhood_sample_result_create(
        const hipgraph_resource_handle_t* handle,
        const hipgraph_type_erased_device_array_view_t* srcs,
        const hipgraph_type_erased_device_array_view_t* dsts,
        const hipgraph_type_erased_device_array_view_t* edge_id,
        const hipgraph_type_erased_device_array_view_t* edge_type,
        const hipgraph_type_erased_device_array_view_t* weight,
        const hipgraph_type_erased_device_array_view_t* hop,
        const hipgraph_type_erased_device_array_view_t* label,
        hipgraph_sample_result_t** result,
        hipgraph_error_t** error
    )

    # random vertices selection
    cdef hipgraph_error_code_t \
        hipgraph_select_random_vertices(
            const hipgraph_resource_handle_t* handle,
            const hipgraph_graph_t* graph,
            hipgraph_rng_state_t* rng_state,
            size_t num_vertices,
            hipgraph_type_erased_device_array_t** vertices,
            hipgraph_error_t** error
        )
