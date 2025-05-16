# Copyright (c) 2019-2022, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from hipgraph.components.connectivity cimport *
from hipgraph.structure.graph_primtypes cimport *
from hipgraph.structure.graph_utilities cimport *

from hipgraph.structure import graph_primtypes_wrapper, utils_wrapper

from libc.stdint cimport uintptr_t

import cudf
import numpy as np
from hipgraph.structure.graph_classes import Graph as type_Graph
from hipgraph.structure.symmetrize import symmetrize


def strongly_connected_components(input_graph):
    """
    Call connected_components
    """
    if not input_graph.adjlist:
        input_graph.view_adj_list()

    [offsets, indices] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])

    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)

    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['labels'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))

    cdef uintptr_t c_offsets    = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices    = indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_identifier = df['vertex'].__cuda_array_interface__['data'][0];
    cdef uintptr_t c_labels_val = df['labels'].__cuda_array_interface__['data'][0];

    cdef GraphCSRView[int,int,float] g

    g = GraphCSRView[int,int,float](<int*>c_offsets, <int*>c_indices, <float*>NULL, num_verts, num_edges)

    cdef hipgraph_cc_t connect_type=HIPGRAPH_STRONG
    connected_components(g, <hipgraph_cc_t>connect_type, <int *>c_labels_val)

    g.get_vertex_identifiers(<int*>c_identifier)

    return df
