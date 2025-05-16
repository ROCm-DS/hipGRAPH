# Copyright (c) 2020-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from pylibraft.common.handle cimport *


cdef extern from "hipgraph/cpp/partition_manager.hpp" namespace "hipgraph::partition_manager":
   cdef void init_subcomm(handle_t &handle,
                          size_t row_comm_size)
