# Copyright (c) 2020-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from hipgraph.layout.force_atlas2 cimport force_atlas2 as c_force_atlas2

from hipgraph.structure import graph_primtypes_wrapper

from hipgraph.structure.graph_primtypes cimport *
from libc.stdint cimport uintptr_t
from libcpp cimport bool

import cudf
import numpy as np
from numba import cuda


cdef extern from "hipgraph/cpp/legacy/internals.hpp" namespace "hipgraph::internals":
    cdef cppclass GraphBasedDimRedCallback


def force_atlas2(input_graph,
                 max_iter=500,
                 pos_list=None,
                 outbound_attraction_distribution=True,
                 lin_log_mode=False,
                 prevent_overlapping=False,
                 edge_weight_influence=1.0,
                 jitter_tolerance=1.0,
                 barnes_hut_optimize=True,
                 barnes_hut_theta=0.5,
                 scaling_ratio=1.0,
                 strong_gravity_mode=False,
                 gravity=1.0,
                 verbose=False,
                 callback=None):

    """
    Call force_atlas2
    """

    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())
    handle_ = handle_ptr.get();

    if not input_graph.edgelist:
        input_graph.view_edge_list()

    # this code allows handling of renumbered graphs
    if input_graph.is_renumbered():
        num_verts = input_graph.renumber_map.df_internal_to_external['id'].max()+1
    else:
        num_verts = input_graph.nodes().max() + 1
    num_edges = len(input_graph.edgelist.edgelist_df['src'])

    cdef GraphCOOView[int,int,float] graph_float
    cdef GraphCOOView[int,int,double] graph_double

    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.arange(num_verts, dtype=np.int32))

    src = input_graph.edgelist.edgelist_df['src']
    dst = input_graph.edgelist.edgelist_df['dst']

    [src, dst] = graph_primtypes_wrapper.datatype_cast([src, dst], [np.int32])

    cdef uintptr_t c_src_indices = src.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst_indices = dst.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = <uintptr_t> NULL

    if input_graph.edgelist.weights:
        weights = input_graph.edgelist.edgelist_df["weights"]
        [weights] = graph_primtypes_wrapper.datatype_cast([weights], [np.float32, np.float64])
        c_weights = weights.__cuda_array_interface__['data'][0]

    cdef uintptr_t x_start = <uintptr_t>NULL
    cdef uintptr_t y_start = <uintptr_t>NULL
    cdef uintptr_t pos_ptr = <uintptr_t>NULL

    if pos_list is not None:
        if len(pos_list) != num_verts:
            raise ValueError('pos_list must have initial positions for all vertices')
        pos_list['x'] = pos_list['x'].astype(np.float32)
        pos_list['y'] = pos_list['y'].astype(np.float32)
        pos_list['x'][pos_list['vertex']] = pos_list['x']
        pos_list['y'][pos_list['vertex']] = pos_list['y']
        x_start = pos_list['x'].__cuda_array_interface__['data'][0]
        y_start = pos_list['y'].__cuda_array_interface__['data'][0]

    cdef uintptr_t callback_ptr = 0
    if callback:
        callback_ptr = callback.get_native_callback()


    # We keep np.float32 as results for both cases
    pos = cuda.device_array(
            (num_verts, 2),
            order="F",
            dtype=np.float32)

    pos_ptr = pos.device_ctypes_pointer.value

    if input_graph.edgelist.weights \
            and input_graph.edgelist.edgelist_df['weights'].dtype == np.float64:
        graph_double = GraphCOOView[int,int, double](<int*>c_src_indices,
                        <int*>c_dst_indices, <double*>c_weights, num_verts, num_edges)

        c_force_atlas2[int, int, double](handle_[0],
                        graph_double,
                        <float*>pos_ptr,
                        <int>max_iter,
                        <float*>x_start,
                        <float*>y_start,
                        <bool>outbound_attraction_distribution,
                        <bool>lin_log_mode,
                        <bool>prevent_overlapping,
                        <float>edge_weight_influence,
                        <float>jitter_tolerance,
                        <bool>barnes_hut_optimize,
                        <float>barnes_hut_theta,
                        <float>scaling_ratio,
                        <bool> strong_gravity_mode,
                        <float>gravity,
                        <bool> verbose,
                        <GraphBasedDimRedCallback*>callback_ptr)
    else:
        graph_float = GraphCOOView[int,int,float](<int*>c_src_indices,
                <int*>c_dst_indices, <float*>c_weights, num_verts,
                num_edges)
        c_force_atlas2[int, int, float](handle_[0],
                graph_float,
                <float*>pos_ptr,
                <int>max_iter,
                <float*>x_start,
                <float*>y_start,
                <bool>outbound_attraction_distribution,
                <bool>lin_log_mode,
                <bool>prevent_overlapping,
                <float>edge_weight_influence,
                <float>jitter_tolerance,
                <bool>barnes_hut_optimize,
                <float>barnes_hut_theta,
                <float>scaling_ratio,
                <bool> strong_gravity_mode,
                <float>gravity,
                <bool> verbose,
                <GraphBasedDimRedCallback*>callback_ptr)

    pos_df = cudf.DataFrame(pos, columns=['x', 'y'])
    df['x'] = pos_df['x']
    df['y'] = pos_df['y']

    return df
