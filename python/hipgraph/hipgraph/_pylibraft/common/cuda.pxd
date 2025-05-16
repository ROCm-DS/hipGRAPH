# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from cuda.ccudart cimport cudaStream_t


cdef class Stream:
    cdef cudaStream_t s

    cdef cudaStream_t getStream(self)
