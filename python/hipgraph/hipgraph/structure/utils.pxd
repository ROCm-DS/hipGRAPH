# Copyright (c) 2019-2022, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from hipgraph.structure.graph_primtypes cimport *
from libcpp.memory cimport unique_ptr


cdef extern from "hipgraph/cpp/legacy/functions.hpp" namespace "hipgraph":

    cdef unique_ptr[GraphCSR[VT,ET,WT]] coo_to_csr[VT,ET,WT](
            const GraphCOOView[VT,ET,WT] &graph) except +

    cdef void comms_bcast[value_t](
            const handle_t &handle,
            value_t *dst,
            size_t size) except +
