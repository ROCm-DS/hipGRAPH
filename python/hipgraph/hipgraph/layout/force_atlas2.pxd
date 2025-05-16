# Copyright (c) 2020-2022, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from hipgraph.structure.graph_primtypes cimport *
from libcpp cimport bool


cdef extern from "hipgraph/cpp/legacy/internals.hpp" namespace "hipgraph::internals":
    cdef cppclass GraphBasedDimRedCallback

cdef extern from "hipgraph/cpp/algorithms.hpp" namespace "hipgraph":

    cdef void force_atlas2[vertex_t, edge_t, weight_t](
        const handle_t &handle,
        GraphCOOView[vertex_t, edge_t, weight_t] &graph,
        float *pos,
        const int max_iter,
        float *x_start,
        float *y_start,
        bool outbound_attraction_distribution,
        bool lin_log_mode,
        bool prevent_overlapping,
        const float edge_weight_influence,
        const float jitter_tolerance,
        bool barnes_hut_optimize,
        const float barnes_hut_theta,
        const float scaling_ratio,
        bool strong_gravity_mode,
        const float gravity,
        bool verbose,
        GraphBasedDimRedCallback *callback) except +
