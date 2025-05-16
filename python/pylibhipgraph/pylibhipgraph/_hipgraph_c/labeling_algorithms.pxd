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


cdef extern from "hipgraph/hipgraph_c/labeling_algorithms.h":
    ###########################################################################
    # weakly connected components
    ctypedef struct hipgraph_labeling_result_t:
        pass

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_labeling_result_get_vertices(
            hipgraph_labeling_result_t* result
        )

    cdef hipgraph_type_erased_device_array_view_t* \
        hipgraph_labeling_result_get_labels(
            hipgraph_labeling_result_t* result
        )

    cdef void \
        hipgraph_labeling_result_free(
            hipgraph_labeling_result_t* result
        )

    cdef hipgraph_error_code_t \
        hipgraph_weakly_connected_components(
            const hipgraph_resource_handle_t* handle,
            hipgraph_graph_t* graph,
            bool_t do_expensive_check,
            hipgraph_labeling_result_t** result,
            hipgraph_error_t** error
        )
