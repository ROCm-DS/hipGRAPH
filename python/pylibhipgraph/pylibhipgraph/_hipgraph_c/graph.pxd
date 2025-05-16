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
from pylibhipgraph._hipgraph_c.resource_handle cimport (
    bool_t,
    hipgraph_resource_handle_t,
)


cdef extern from "hipgraph/hipgraph_c/graph.h":

    ctypedef struct hipgraph_graph_t:
        pass

    ctypedef struct hipgraph_graph_properties_t:
        bool_t is_symmetric
        bool_t is_multigraph

    cdef hipgraph_error_code_t \
         hipgraph_sg_graph_create(
             const hipgraph_resource_handle_t* handle,
             const hipgraph_graph_properties_t* properties,
             const hipgraph_type_erased_device_array_view_t* src,
             const hipgraph_type_erased_device_array_view_t* dst,
             const hipgraph_type_erased_device_array_view_t* weights,
             const hipgraph_type_erased_device_array_view_t* edge_ids,
             const hipgraph_type_erased_device_array_view_t* edge_types,
             bool_t store_transposed,
             bool_t renumber,
             bool_t check,
             hipgraph_graph_t** graph,
             hipgraph_error_t** error)

    # Supports isolated vertices
    cdef hipgraph_error_code_t \
         hipgraph_graph_create_sg(
             const hipgraph_resource_handle_t* handle,
             const hipgraph_graph_properties_t* properties,
             const hipgraph_type_erased_device_array_view_t* vertices,
             const hipgraph_type_erased_device_array_view_t* src,
             const hipgraph_type_erased_device_array_view_t* dst,
             const hipgraph_type_erased_device_array_view_t* weights,
             const hipgraph_type_erased_device_array_view_t* edge_ids,
             const hipgraph_type_erased_device_array_view_t* edge_types,
             bool_t store_transposed,
             bool_t renumber,
             bool_t drop_self_loops,
             bool_t drop_multi_edges,
             bool_t check,
             hipgraph_graph_t** graph,
             hipgraph_error_t** error)

    # This may get renamed to hipgraph_graph_free()
    cdef void \
        hipgraph_sg_graph_free(
            hipgraph_graph_t* graph
        )

    # FIXME: Might want to delete 'hipgraph_sg_graph_free' and replace
    # 'hipgraph_mg_graph_free' by 'hipgraph_graph_free'
    cdef void \
        hipgraph_graph_free(
            hipgraph_graph_t* graph
        )

    # cdef hipgraph_error_code_t \
    #     hipgraph_mg_graph_create(
    #         const hipgraph_resource_handle_t* handle,
    #         const hipgraph_graph_properties_t* properties,
    #         const hipgraph_type_erased_device_array_view_t* src,
    #         const hipgraph_type_erased_device_array_view_t* dst,
    #         const hipgraph_type_erased_device_array_view_t* weights,
    #         const hipgraph_type_erased_device_array_view_t* edge_ids,
    #         const hipgraph_type_erased_device_array_view_t* edge_types,
    #         bool_t store_transposed,
    #         size_t num_edges,
    #         bool_t check,
    #         hipgraph_graph_t** graph,
    #         hipgraph_error_t** error
    #     )
    #
    # # This may get renamed to or replaced with hipgraph_graph_free()
    # cdef void \
    #     hipgraph_mg_graph_free(
    #         hipgraph_graph_t* graph
    #     )

    cdef hipgraph_error_code_t \
        hipgraph_sg_graph_create_from_csr(
            const hipgraph_resource_handle_t* handle,
            const hipgraph_graph_properties_t* properties,
            const hipgraph_type_erased_device_array_view_t* offsets,
            const hipgraph_type_erased_device_array_view_t* indices,
            const hipgraph_type_erased_device_array_view_t* weights,
            const hipgraph_type_erased_device_array_view_t* edge_ids,
            const hipgraph_type_erased_device_array_view_t* edge_type_ids,
            bool_t store_transposed,
            bool_t renumber,
            bool_t check,
            hipgraph_graph_t** graph,
            hipgraph_error_t** error
        )

    cdef hipgraph_error_code_t \
        hipgraph_graph_create_sg_from_csr(
            const hipgraph_resource_handle_t* handle,
            const hipgraph_graph_properties_t* properties,
            const hipgraph_type_erased_device_array_view_t* offsets,
            const hipgraph_type_erased_device_array_view_t* indices,
            const hipgraph_type_erased_device_array_view_t* weights,
            const hipgraph_type_erased_device_array_view_t* edge_ids,
            const hipgraph_type_erased_device_array_view_t* edge_type_ids,
            bool_t store_transposed,
            bool_t renumber,
            bool_t check,
            hipgraph_graph_t** graph,
            hipgraph_error_t** error
        )

    cdef void \
        hipgraph_sg_graph_free(
            hipgraph_graph_t* graph
        )

    # cdef hipgraph_error_code_t \
    #     hipgraph_mg_graph_create(
    #         const hipgraph_resource_handle_t* handle,
    #         const hipgraph_graph_properties_t* properties,
    #         const hipgraph_type_erased_device_array_view_t* src,
    #         const hipgraph_type_erased_device_array_view_t* dst,
    #         const hipgraph_type_erased_device_array_view_t* weights,
    #         const hipgraph_type_erased_device_array_view_t* edge_ids,
    #         const hipgraph_type_erased_device_array_view_t* edge_type_ids,
    #         bool_t store_transposed,
    #         size_t num_edges,
    #         bool_t check,
    #         hipgraph_graph_t** graph,
    #         hipgraph_error_t** error
    #     )
    #
    # cdef hipgraph_error_code_t \
    #     hipgraph_graph_create_mg(
    #         const hipgraph_resource_handle_t* handle,
    #         const hipgraph_graph_properties_t* properties,
    #         const hipgraph_type_erased_device_array_view_t** vertices,
    #         const hipgraph_type_erased_device_array_view_t** src,
    #         const hipgraph_type_erased_device_array_view_t** dst,
    #         const hipgraph_type_erased_device_array_view_t** weights,
    #         const hipgraph_type_erased_device_array_view_t** edge_ids,
    #         const hipgraph_type_erased_device_array_view_t** edge_type_ids,
    #         bool_t store_transposed,
    #         size_t num_arrays,
    #         bool_t drop_self_loops,
    #         bool_t drop_multi_edges,
    #         bool_t do_expensive_check,
    #         hipgraph_graph_t** graph,
    #         hipgraph_error_t** error)
    #
    # cdef void \
    #     hipgraph_mg_graph_free(
    #         hipgraph_graph_t* graph
    #     )
