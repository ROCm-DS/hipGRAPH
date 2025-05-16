# Copyright (c) 2024, NVIDIA CORPORATION.
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


cdef extern from "hipgraph/hipgraph_c/lookup_src_dst.h":
    ###########################################################################

    ctypedef struct hipgraph_lookup_container_t:
       pass

    ctypedef struct hipgraph_lookup_result_t:
       pass

    cdef hipgraph_error_code_t hipgraph_build_edge_id_and_type_to_src_dst_lookup_map(
        const hipgraph_resource_handle_t* handle,
        hipgraph_graph_t* graph,
        hipgraph_lookup_container_t** lookup_container,
        hipgraph_error_t** error)

    cdef hipgraph_error_code_t hipgraph_lookup_endpoints_from_edge_ids_and_single_type(
        const hipgraph_resource_handle_t* handle,
        hipgraph_graph_t* graph,
        const hipgraph_lookup_container_t* lookup_container,
        const hipgraph_type_erased_device_array_view_t* edge_ids_to_lookup,
        int edge_type_to_lookup,
        hipgraph_lookup_result_t** result,
        hipgraph_error_t** error)

    cdef hipgraph_error_code_t hipgraph_lookup_endpoints_from_edge_ids_and_types(
        const hipgraph_resource_handle_t* handle,
        hipgraph_graph_t* graph,
        const hipgraph_lookup_container_t* lookup_container,
        const hipgraph_type_erased_device_array_view_t* edge_ids_to_lookup,
        const hipgraph_type_erased_device_array_view_t* edge_types_to_lookup,
        hipgraph_lookup_result_t** result,
        hipgraph_error_t** error)

    cdef hipgraph_type_erased_device_array_view_t* hipgraph_lookup_result_get_srcs(
        const hipgraph_lookup_result_t* result)

    cdef hipgraph_type_erased_device_array_view_t* hipgraph_lookup_result_get_dsts(
        const hipgraph_lookup_result_t* result)

    cdef void hipgraph_lookup_result_free(hipgraph_lookup_result_t* result)
