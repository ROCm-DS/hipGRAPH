# Copyright (c) 2021, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from hipgraph.structure.graph_primtypes cimport *


cdef extern from "hipgraph/cpp/utilities/path_retrieval.hpp" namespace "hipgraph":

    cdef void get_traversed_cost[vertex_t, weight_t](const handle_t &handle,
            const vertex_t *vertices,
            const vertex_t *preds,
            const weight_t *info_weights,
            weight_t *out,
            vertex_t stop_vertex,
            vertex_t num_vertices) except +
