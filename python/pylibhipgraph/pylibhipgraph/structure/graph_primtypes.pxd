# Copyright (c) 2021-2022, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp cimport bool

# AMD: We "cheat" and just use a generic pointer to an undefined struct to break
# the *only* direct RAFT dependency in pylibhipgraph.
# from pylibraft.common.handle cimport *

cdef extern from "hipgraph/legacy/graph.hpp" namespace "hipgraph::legacy":

    ctypedef enum PropType:
        PROP_UNDEF "hipgraph::legacy::PROP_UNDEF"
        PROP_FALSE "hipgraph::legacy::PROP_FALSE"
        PROP_TRUE "hipgraph::legacy::PROP_TRUE"

    ctypedef enum DegreeDirection:
        DIRECTION_IN_PLUS_OUT "hipgraph::legacy::DegreeDirection::IN_PLUS_OUT"
        DIRECTION_IN "hipgraph::legacy::DegreeDirection::IN"
        DIRECTION_OUT "hipgraph::legacy::DegreeDirection::OUT"

    struct GraphProperties:
        bool directed
        bool weighted
        bool multigraph
        bool bipartite
        bool tree
        PropType has_negative_edges

    ctypedef struct handle_t:
        pass

    cdef cppclass GraphViewBase[VT,ET,WT]:
        WT *edge_data
        handle_t *handle;
        GraphProperties prop
        VT number_of_vertices
        ET number_of_edges
        VT* local_vertices
        ET* local_edges
        VT* local_offsets
        void set_handle(handle_t*)
        void set_local_data(VT* local_vertices_, ET* local_edges_, VT* local_offsets_)
        void get_vertex_identifiers(VT *) const

        GraphViewBase(WT*,VT,ET)

    cdef cppclass GraphCOOView[VT,ET,WT](GraphViewBase[VT,ET,WT]):
        VT *src_indices
        VT *dst_indices

        void degree(ET *,DegreeDirection) const

        GraphCOOView()
        GraphCOOView(const VT *, const ET *, const WT *, size_t, size_t)

    cdef cppclass GraphCompressedSparseBaseView[VT,ET,WT](GraphViewBase[VT,ET,WT]):
        ET *offsets
        VT *indices

        void get_source_indices(VT *) const
        void degree(ET *,DegreeDirection) const

        GraphCompressedSparseBaseView(const VT *, const ET *, const WT *, size_t, size_t)

    cdef cppclass GraphCSRView[VT,ET,WT](GraphCompressedSparseBaseView[VT,ET,WT]):
        GraphCSRView()
        GraphCSRView(const VT *, const ET *, const WT *, size_t, size_t)

    cdef cppclass GraphCSCView[VT,ET,WT](GraphCompressedSparseBaseView[VT,ET,WT]):
        GraphCSCView()
        GraphCSCView(const VT *, const ET *, const WT *, size_t, size_t)
