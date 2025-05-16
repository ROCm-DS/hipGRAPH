# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

cdef extern from "hipgraph/hipgraph_c/error.h":

    ctypedef enum hipgraph_error_code_t:
        HIPGRAPH_SUCCESS
        HIPGRAPH_UNKNOWN_ERROR
        HIPGRAPH_INVALID_HANDLE
        HIPGRAPH_ALLOC_ERROR
        HIPGRAPH_INVALID_INPUT
        HIPGRAPH_NOT_IMPLEMENTED
        HIPGRAPH_UNSUPPORTED_TYPE_COMBINATION

    ctypedef struct hipgraph_error_t:
       pass

    const char* \
        hipgraph_error_message(
            const hipgraph_error_t* error
        )

    void \
        hipgraph_error_free(
            hipgraph_error_t* error
        )
