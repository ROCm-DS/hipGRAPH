# Copyright (c) 2022-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from libcpp.memory cimport shared_ptr, unique_ptr
from rmm._lib.cuda_stream_pool cimport cuda_stream_pool
from rmm._lib.cuda_stream_view cimport cuda_stream_view


# Keeping `handle_t` around for backwards compatibility at the
# cython layer but users are encourage to switch to device_resources
cdef extern from "raft/core/handle.hpp" namespace "raft" nogil:
    cdef cppclass handle_t:
        handle_t() except +
        handle_t(cuda_stream_view stream_view) except +
        handle_t(cuda_stream_view stream_view,
                 shared_ptr[cuda_stream_pool] stream_pool) except +
        cuda_stream_view get_stream() except +
        void sync_stream() except +


cdef extern from "raft/core/device_resources.hpp" namespace "raft" nogil:
    cdef cppclass device_resources:
        device_resources() except +
        device_resources(cuda_stream_view stream_view) except +
        device_resources(cuda_stream_view stream_view,
                         shared_ptr[cuda_stream_pool] stream_pool) except +
        cuda_stream_view get_stream() except +
        void sync_stream() except +

cdef class DeviceResources:
    cdef unique_ptr[device_resources] c_obj
    cdef shared_ptr[cuda_stream_pool] stream_pool
    cdef int n_streams
