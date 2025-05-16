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


cdef extern from "hipgraph/hipgraph_c/core_algorithms.h":
    ###########################################################################
    # core number
    ctypedef struct hipgraph_core_result_t:
        pass

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_core_result_get_vertices(
            hipgraph_core_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_core_result_get_core_numbers(
            hipgraph_core_result_t* result
        )

    cdef void \
        hipgraph_core_result_free(
            hipgraph_core_result_t* result
        )

    ctypedef enum hipgraph_k_core_degree_type_t:
        K_CORE_DEGREE_TYPE_IN=0,
        K_CORE_DEGREE_TYPE_OUT=1,
        K_CORE_DEGREE_TYPE_INOUT=2

    cdef hipgraph_error_code_t \
        hipgraph_core_number(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            hipgraph_k_core_degree_type_t degree_type,
            bool_t do_expensive_check,
            hipgraph_core_result_t** result,
            hipgraph_error_t** error
        )

    ###########################################################################
    # k-core
    ctypedef struct hipgraph_k_core_result_t:
        pass

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_k_core_result_get_src_vertices(
            hipgraph_k_core_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_k_core_result_get_dst_vertices(
            hipgraph_k_core_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_k_core_result_get_weights(
            hipgraph_k_core_result_t* result
        )

    cdef void \
        hipgraph_k_core_result_free(
            hipgraph_k_core_result_t* result
        )

    cdef hipgraph_error_code_t \
        hipgraph_core_result_create(
            const hipgraph_resource_handle_t* handle,
            hipgraph_type_erased_device_array_view_t* vertices,
            hipgraph_type_erased_device_array_view_t* core_numbers,
            hipgraph_core_result_t** core_result,
            hipgraph_error_t** error
        )

    cdef hipgraph_error_code_t \
        hipgraph_k_core(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            size_t k,
            hipgraph_k_core_degree_type_t degree_type,
            const hipgraph_core_result_t* core_result,
            bool_t do_expensive_check,
            hipgraph_k_core_result_t** result,
            hipgraph_error_t** error
        )
