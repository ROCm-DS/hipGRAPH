# Copyright (c) 2019-2021, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from hipgraph.structure.graph_primtypes cimport *


cdef extern from "hipgraph/cpp/algorithms.hpp" namespace "hipgraph":

    cdef unique_ptr[GraphCOO[VT,ET,WT]] minimum_spanning_tree[VT,ET,WT](const handle_t &handle,
        const GraphCSRView[VT,ET,WT] &graph) except +
