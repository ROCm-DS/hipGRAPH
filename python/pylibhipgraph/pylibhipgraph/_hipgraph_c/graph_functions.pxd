# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibhipgraph._hipgraph_c.array cimport hipgraph_type_erased_device_array_view_t
from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.graph cimport hipgraph_graph_t
from pylibhipgraph._hipgraph_c.resource_handle cimport (
    bool_t,
    hipgraph_resource_handle_t,
)
from pylibhipgraph._hipgraph_c.similarity_algorithms cimport (
    hipgraph_similarity_result_t,
)


cdef extern from "hipgraph/hipgraph_c/graph_functions.h":
    #"""
    #ctypedef struct hipgraph_similarity_result_t:
    #    pass
    #"""
    ctypedef struct hipgraph_vertex_pairs_t:
        pass


from pylibhipgraph._hipgraph_c.array cimport hipgraph_type_erased_device_array_view_t
from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.graph cimport hipgraph_graph_t


cdef extern from "hipgraph/hipgraph_c/graph_functions.h":
    ###########################################################################
    # vertex_pairs
    ctypedef struct hipgraph_vertex_pairs_t:
        pass

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_vertex_pairs_get_first(
            hipgraph_vertex_pairs_t* vertex_pairs
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_vertex_pairs_get_second(
            hipgraph_vertex_pairs_t* vertex_pairs
        )

    cdef void \
        hipgraph_vertex_pairs_free(
            hipgraph_vertex_pairs_t* vertex_pairs
        )

    cdef hipgraph_error_code_t \
        hipgraph_create_vertex_pairs(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_type_erased_device_array_view_t* first,
            const hipgraph_type_erased_device_array_view_t* second,
            hipgraph_vertex_pairs_t** vertex_pairs,
            hipgraph_error_t** error
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_vertex_pairs_get_first(
            hipgraph_vertex_pairs_t* vertex_pairs
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_vertex_pairs_get_second(
            hipgraph_vertex_pairs_t* vertex_pairs
        )

    cdef void hipgraph_vertex_pairs_free(
        hipgraph_vertex_pairs_t* vertex_pairs
        )

    cdef hipgraph_error_code_t hipgraph_two_hop_neighbors(
        const hipgraph_resource_handle_t* handle,
        const hipgraph_graph_t* graph,
        const hipgraph_type_erased_device_array_view_t* start_vertices,
        bool_t do_expensive_check,
        hipgraph_vertex_pairs_t** result,
        hipgraph_error_t** error)

    cdef hipgraph_error_code_t \
        hipgraph_two_hop_neighbors(
            const hipgraph_resource_handle_t* handle,
            const hipgraph_graph_t* graph,
            const hipgraph_type_erased_device_array_view_t* start_vertices,
            hipgraph_vertex_pairs_t** result,
            hipgraph_error_t** error
        )

    ###########################################################################
    # induced_subgraph
    ctypedef struct hipgraph_induced_subgraph_result_t:
        pass

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_induced_subgraph_get_sources(
            hipgraph_induced_subgraph_result_t* induced_subgraph
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_induced_subgraph_get_destinations(
            hipgraph_induced_subgraph_result_t* induced_subgraph
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_induced_subgraph_get_edge_weights(
            hipgraph_induced_subgraph_result_t* induced_subgraph
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_induced_subgraph_get_edge_ids(
            hipgraph_induced_subgraph_result_t* induced_subgraph
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_induced_subgraph_get_edge_type_ids(
            hipgraph_induced_subgraph_result_t* induced_subgraph
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_induced_subgraph_get_subgraph_offsets(
            hipgraph_induced_subgraph_result_t* induced_subgraph
        )

    cdef void \
        hipgraph_induced_subgraph_result_free(
            hipgraph_induced_subgraph_result_t* induced_subgraph
        )

    cdef hipgraph_error_code_t \
        hipgraph_extract_induced_subgraph(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_type_erased_device_array_view_t* subgraph_offsets,
            const hipgraph_type_erased_device_array_view_t* subgraph_vertices,
            bool_t do_expensive_check,
            hipgraph_induced_subgraph_result_t** result,
            hipgraph_error_t** error
        )

    ###########################################################################
    # allgather
    cdef hipgraph_error_code_t \
        hipgraph_allgather(
            const hipgraph_resource_handle_t* handle,
            const hipgraph_type_erased_device_array_view_t* src,
            const hipgraph_type_erased_device_array_view_t* dst,
            const hipgraph_type_erased_device_array_view_t* weights,
            const hipgraph_type_erased_device_array_view_t* edge_ids,
            const hipgraph_type_erased_device_array_view_t* edge_type_ids,
            hipgraph_induced_subgraph_result_t** result,
            hipgraph_error_t** error
        )

    ###########################################################################
    # count multi-edges
    cdef hipgraph_error_code_t \
        hipgraph_count_multi_edges(
            const hipgraph_resource_handle_t *handle,
            hipgraph_graph_t* graph,
            bool_t do_expenive_check,
            size_t *result,
            hipgraph_error_t** error
        )

    ###########################################################################
    # degrees
    ctypedef struct hipgraph_degrees_result_t:
        pass

    cdef hipgraph_error_code_t \
        hipgraph_in_degrees(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_type_erased_device_array_view_t* source_vertices,
            bool_t do_expensive_check,
            hipgraph_degrees_result_t** result,
            hipgraph_error_t** error
        )

    cdef hipgraph_error_code_t \
        hipgraph_out_degrees(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_type_erased_device_array_view_t* source_vertices,
            bool_t do_expensive_check,
            hipgraph_degrees_result_t** result,
            hipgraph_error_t** error
        )

    cdef hipgraph_error_code_t \
        hipgraph_degrees(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            const hipgraph_type_erased_device_array_view_t* source_vertices,
            bool_t do_expensive_check,
            hipgraph_degrees_result_t** result,
            hipgraph_error_t** error
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_degrees_result_get_vertices(
            hipgraph_degrees_result_t* degrees_result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_degrees_result_get_in_degrees(
            hipgraph_degrees_result_t* degrees_result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_degrees_result_get_out_degrees(
            hipgraph_degrees_result_t* degrees_result
        )

    cdef void \
        hipgraph_degrees_result_free(
            hipgraph_degrees_result_t* degrees_result
        )
