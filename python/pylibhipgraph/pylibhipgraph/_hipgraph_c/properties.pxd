# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

cdef extern from "hipgraph/hipgraph_c/properties.h":

    ctypedef struct hipgraph_vertex_property_t:
        pass

    ctypedef struct hipgraph_edge_property_t:
        pass

    ctypedef struct hipgraph_vertex_property_view_t:
        pass

    ctypedef struct hipgraph_edge_property_view_t:
        pass
