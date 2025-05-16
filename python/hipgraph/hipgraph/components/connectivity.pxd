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
from hipgraph.structure.graph_utilities cimport *


cdef extern from "hipgraph/cpp/algorithms.hpp" namespace "hipgraph":

    ctypedef enum hipgraph_cc_t:
        HIPGRAPH_STRONG "hipgraph::hipgraph_cc_t::HIPGRAPH_STRONG"

    cdef void connected_components[VT,ET,WT](
        const GraphCSRView[VT,ET,WT] &graph,
        hipgraph_cc_t connect_type,
        VT *labels) except +
