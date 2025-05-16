# Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.random cimport (
    hipgraph_rng_state_create,
    hipgraph_rng_state_free,
    hipgraph_rng_state_t,
)
from pylibhipgraph._hipgraph_c.resource_handle cimport hipgraph_resource_handle_t
from pylibhipgraph.resource_handle cimport ResourceHandle
from pylibhipgraph.utils cimport assert_success

import os
import socket
import time


def generate_default_seed():
    h = hash(
            (
                socket.gethostname(),
                os.getpid(),
                time.perf_counter_ns()
            )
        )

    return h

cdef class HipGraphRandomState:
    """
        This class wraps a hipgraph_rng_state_t instance, which represents a
        random state.
    """

    def __cinit__(self, ResourceHandle resource_handle, seed=None):
        """
        Constructs a new HipGraphRandomState instance.

        Parameters
        ----------
        resource_handle: pylibhipgraph.ResourceHandle (Required)
            The hipgraph resource handle for this process.
        seed: int (Optional)
            The random seed of this random state object.
            Defaults to the hash of the hostname, pid, and time.

        """

        cdef hipgraph_error_code_t error_code
        cdef hipgraph_error_t* error_ptr

        cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
            resource_handle.c_resource_handle_ptr

        cdef hipgraph_rng_state_t* new_rng_state_ptr

        if seed is None:
            seed = generate_default_seed()

        # reinterpret as unsigned
        seed &= (2**64 - 1)

        error_code = hipgraph_rng_state_create(
            c_resource_handle_ptr,
            <size_t>seed,
            &new_rng_state_ptr,
            &error_ptr
        )
        assert_success(error_code, error_ptr, "hipgraph_rng_state_create")

        self.rng_state_ptr = new_rng_state_ptr

    def __dealloc__(self):
        """
        Destroys this HipGraphRandomState instance.  Properly calls
        free to destroy the underlying C++ object.
        """
        hipgraph_rng_state_free(self.rng_state_ptr)
