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
from hipgraph.tree.minimum_spanning_tree cimport minimum_spanning_tree as c_mst

from hipgraph.structure import graph_primtypes_wrapper

from libc.stdint cimport uintptr_t

import cudf
import cupy as cp

# FIXME: these are transitive dependencies and are not currently listed in the
# conda recipe. Either make them optional or add them to the conda recipe.
import numpy as np


def mst_float(num_verts, num_edges, offsets, indices, weights):
    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())
    handle_ = handle_ptr.get();
    cdef uintptr_t c_offsets = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices = indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = weights.__cuda_array_interface__['data'][0]
    cdef GraphCSRView[int,int,float] graph_float
    graph_float = GraphCSRView[int,int,float](<int*>c_offsets, <int*>c_indices, <float*>c_weights, num_verts, num_edges)
    return coo_to_df(move(c_mst[int,int,float](handle_[0], graph_float)))


def mst_double(num_verts, num_edges, offsets, indices, weights):
    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())
    handle_ = handle_ptr.get();
    cdef uintptr_t c_offsets = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices = indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = weights.__cuda_array_interface__['data'][0]
    cdef GraphCSRView[int,int,double] graph_double
    graph_double = GraphCSRView[int,int,double](<int*>c_offsets, <int*>c_indices, <double*>c_weights, num_verts, num_edges)
    return coo_to_df(move(c_mst[int,int,double](handle_[0], graph_double)))


def minimum_spanning_tree(input_graph):
    if not input_graph.adjlist:
        input_graph.view_adj_list()
    [offsets, indices] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])
    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)

    if input_graph.adjlist.weights is not None:
        [weights] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.weights], [np.float32, np.float64])
    else:
        weights = cudf.Series(cp.full(num_edges, 1.0, dtype=np.float32))

    if graph_primtypes_wrapper.weight_type(input_graph) == np.float32:
         df = mst_float(num_verts, num_edges, offsets, indices, weights)
         return df
    else:
        return mst_double(num_verts, num_edges, offsets, indices, weights)

def maximum_spanning_tree(input_graph):
    if not input_graph.adjlist:
        input_graph.view_adj_list()
    [offsets, indices] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])
    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)

    if input_graph.adjlist.weights is not None:
        [weights] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.weights], [np.float32, np.float64])
    else:
        weights = cudf.Series(cp.full(num_edges, 1.0, dtype=np.float32))

    if graph_primtypes_wrapper.weight_type(input_graph) == np.float32:
         df = mst_float(num_verts, num_edges, offsets, indices, weights)
         return df
    else:
        return mst_double(num_verts, num_edges, offsets, indices, weights)
