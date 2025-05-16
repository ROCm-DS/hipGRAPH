# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibhipgraph._hipgraph_c.graph cimport (
    hipgraph_graph_t,
    hipgraph_type_erased_device_array_view_t,
)


# Base class allowing functions to accept either SGGraph or MGGraph
# This is not visible in python
cdef class _GPUGraph:
    cdef hipgraph_graph_t* c_graph_ptr
    cdef hipgraph_type_erased_device_array_view_t* edge_id_view_ptr
    cdef hipgraph_type_erased_device_array_view_t** edge_id_view_ptr_ptr
    cdef hipgraph_type_erased_device_array_view_t* weights_view_ptr
    cdef hipgraph_type_erased_device_array_view_t** weights_view_ptr_ptr

cdef class SGGraph(_GPUGraph):
    pass

# Disabling multi-gpu graphs for now.
# cdef class MGGraph(_GPUGraph):
#     pass
