# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

    cdef weight_t hungarian[vertex_t,edge_t,weight_t](
        const handle_t &handle,
        const GraphCOOView[vertex_t,edge_t,weight_t] &graph,
        vertex_t num_workers,
        const vertex_t *workers,
        vertex_t *assignments,
        weight_t epsilon) except +

    cdef weight_t hungarian[vertex_t,edge_t,weight_t](
        const handle_t &handle,
        const GraphCOOView[vertex_t,edge_t,weight_t] &graph,
        vertex_t num_workers,
        const vertex_t *workers,
        vertex_t *assignments) except +

cdef extern from "hipgraph/cpp/algorithms.hpp":

    cdef weight_t dense_hungarian "hipgraph::dense::hungarian" [vertex_t,weight_t](
        const handle_t &handle,
        const weight_t *costs,
        vertex_t num_rows,
        vertex_t num_columns,
        vertex_t *assignments,
        weight_t epsilon) except +

    cdef weight_t dense_hungarian "hipgraph::dense::hungarian" [vertex_t,weight_t](
        const handle_t &handle,
        const weight_t *costs,
        vertex_t num_rows,
        vertex_t num_columns,
        vertex_t *assignments) except +
