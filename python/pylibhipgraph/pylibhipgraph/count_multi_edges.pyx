# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.graph cimport hipgraph_graph_t
from pylibhipgraph._hipgraph_c.graph_functions cimport hipgraph_count_multi_edges
from pylibhipgraph._hipgraph_c.resource_handle cimport (
    bool_t,
    data_type_id_t,
    hipgraph_resource_handle_t,
)
from pylibhipgraph.graphs cimport _GPUGraph
from pylibhipgraph.resource_handle cimport ResourceHandle


def count_multi_edges(ResourceHandle resource_handle,
                      _GPUGraph graph,
                      bool_t do_expensive_check):
    """
    Count the number of multi-edges in the graph.  This returns
    the number of duplicates.  If the edge (u, v) appears k times
    in the graph, then that edge will contribute (k-1) toward the
    total number of duplicates.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    do_expensive_check : bool_t
        A flag to run expensive checks for input arguments if True.

    Returns
    -------
    Total count of duplicate edges in the graph

    Examples
    --------
    >>> import pylibhipgraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 0, 0], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 1, 1], dtype=numpy.int32)
    >>> weights = cupy.asarray([1.0, 1.0, 1.0], dtype=numpy.float32)
    >>> resource_handle = pylibhipgraph.ResourceHandle()
    >>> graph_props = pylibhipgraph.GraphProperties(
    ...     is_symmetric=False, is_multigraph=False)
    >>> G = pylibhipgraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=True, renumber=False, do_expensive_check=False)
    >>> count = pylibhipgraph.count_multi_edges(resource_handle, G, False)

    """

    cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef hipgraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef size_t result
    cdef hipgraph_error_code_t error_code
    cdef hipgraph_error_t* error_ptr

    error_code = hipgraph_count_multi_edges(c_resource_handle_ptr,
                                           c_graph_ptr,
                                           do_expensive_check,
                                           &result,
                                           &error_ptr)
    assert_success(error_code, error_ptr, "hipgraph_count_multi_edges")

    return result;
