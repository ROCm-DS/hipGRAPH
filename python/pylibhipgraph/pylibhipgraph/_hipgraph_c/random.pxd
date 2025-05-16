# Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.resource_handle cimport hipgraph_resource_handle_t


cdef extern from "hipgraph/hipgraph_c/random.h":
    ctypedef struct hipgraph_rng_state_t:
        pass

    cdef hipgraph_error_code_t hipgraph_rng_state_create(
        const hipgraph_resource_handle_t* handle,
        size_t seed,
        hipgraph_rng_state_t** state,
        hipgraph_error_t** error,
    )

    cdef void hipgraph_rng_state_free(hipgraph_rng_state_t* p)
