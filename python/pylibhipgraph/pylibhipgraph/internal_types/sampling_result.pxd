# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3


from pylibhipgraph._hipgraph_c.algorithms cimport hipgraph_sample_result_t


cdef class SamplingResult:
    cdef hipgraph_sample_result_t* c_sample_result_ptr
    cdef set_ptr(self, hipgraph_sample_result_t* sample_result_ptr)
