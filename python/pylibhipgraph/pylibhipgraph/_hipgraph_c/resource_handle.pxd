# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

from libc.stdint cimport int8_t


cdef extern from "hipgraph/hipgraph_c/resource_handle.h":

    ctypedef enum bool_t:
        FALSE
        TRUE

    ctypedef enum data_type_id_t:
        INT32
        INT64
        FLOAT32
        FLOAT64
        SIZE_T

    ctypedef data_type_id_t hipgraph_data_type_id_t

    ctypedef int8_t byte_t

    ctypedef struct hipgraph_resource_handle_t:
        pass

    # FIXME: the void* raft_handle arg will change in a future release
    cdef hipgraph_resource_handle_t* \
        hipgraph_create_resource_handle(
	    void* raft_handle
	)

    cdef int \
        hipgraph_resource_handle_get_rank(
	    const hipgraph_resource_handle_t* handle
	)

    cdef void \
        hipgraph_free_resource_handle(
            hipgraph_resource_handle_t* p_handle
        )
