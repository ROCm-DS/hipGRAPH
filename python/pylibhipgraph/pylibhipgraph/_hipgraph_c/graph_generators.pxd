# Copyright (c) 2023, NVIDIA CORPORATION.
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
from pylibhipgraph._hipgraph_c.random cimport hipgraph_rng_state_t
from pylibhipgraph._hipgraph_c.resource_handle cimport (
    bool_t,
    hipgraph_data_type_id_t,
    hipgraph_resource_handle_t,
)


cdef extern from "hipgraph/hipgraph_c/graph_generators.h":
    ctypedef enum hipgraph_generator_distribution_t:
        POWER_LAW
        UNIFORM

    ctypedef struct hipgraph_coo_t:
        pass

    ctypedef struct hipgraph_coo_list_t:
        pass

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_coo_get_sources(
            hipgraph_coo_t* coo
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_coo_get_destinations(
            hipgraph_coo_t* coo
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_coo_get_edge_weights(
            hipgraph_coo_t* coo
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_coo_get_edge_id(
            hipgraph_coo_t* coo
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_coo_get_edge_type(
            hipgraph_coo_t* coo
        )

    cdef size_t \
        hipgraph_coo_list_size(
            const hipgraph_coo_list_t* coo_list
        )

    cdef hipgraph_coo_t* \
        hipgraph_coo_list_element(
            hipgraph_coo_list_t* coo_list,
            size_t index)

    cdef void \
        hipgraph_coo_free(
            hipgraph_coo_t* coo
        )

    cdef void \
        hipgraph_coo_list_free(
            hipgraph_coo_list_t* coo_list
        )

    cdef hipgraph_error_code_t \
        hipgraph_generate_rmat_edgelist(
            const hipgraph_resource_handle_t* handle,
            hipgraph_rng_state_t* rng_state,
            size_t scale,
            size_t num_edges,
            double a,
            double b,
            double c,
            bool_t clip_and_flip,
            bool_t scramble_vertex_ids,
            hipgraph_coo_t** result,
            hipgraph_error_t** error
        )

    cdef hipgraph_error_code_t \
        hipgraph_generate_rmat_edgelists(
            const hipgraph_resource_handle_t* handle,
            hipgraph_rng_state_t* rng_state,
            size_t n_edgelists,
            size_t min_scale,
            size_t max_scale,
            size_t edge_factor,
            hipgraph_generator_distribution_t size_distribution,
            hipgraph_generator_distribution_t edge_distribution,
            bool_t clip_and_flip,
            bool_t scramble_vertex_ids,
            hipgraph_coo_list_t** result,
            hipgraph_error_t** error
        )

    cdef hipgraph_error_code_t \
        hipgraph_generate_edge_weights(
            const hipgraph_resource_handle_t* handle,
            hipgraph_rng_state_t* rng_state,
            hipgraph_coo_t* coo,
            hipgraph_data_type_id_t dtype,
            double minimum_weight,
            double maximum_weight,
            hipgraph_error_t** error
        )

    cdef hipgraph_error_code_t \
        hipgraph_generate_edge_ids(
            const hipgraph_resource_handle_t* handle,
            hipgraph_coo_t* coo,
            bool_t multi_gpu,
            hipgraph_error_t** error
        )

    cdef hipgraph_error_code_t \
        hipgraph_generate_edge_types(
            const hipgraph_resource_handle_t* handle,
            hipgraph_rng_state_t* rng_state,
            hipgraph_coo_t* coo,
            int min_edge_type,
            int max_edge_type,
            hipgraph_error_t** error
        )
