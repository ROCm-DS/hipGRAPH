# Copyright (c) 2021-2022, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.memory cimport shared_ptr
from rmm._lib.cuda_stream_view cimport cuda_stream_view


cdef extern from "raft/core/interruptible.hpp" namespace "raft" nogil:
    cdef cppclass interruptible:
        void cancel()

cdef extern from "raft/core/interruptible.hpp" \
        namespace "raft::interruptible" nogil:
    cdef void inter_synchronize \
        "raft::interruptible::synchronize"(cuda_stream_view stream) except+
    cdef void inter_yield "raft::interruptible::yield"() except+
    cdef shared_ptr[interruptible] get_token() except+
