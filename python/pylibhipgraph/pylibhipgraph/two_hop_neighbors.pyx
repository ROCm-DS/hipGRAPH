# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3


from pylibhipgraph._hipgraph_c.array cimport (
    hipgraph_type_erased_device_array_view_free,
    hipgraph_type_erased_device_array_view_t,
)
from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.graph cimport hipgraph_graph_t
from pylibhipgraph._hipgraph_c.graph_functions cimport (
    hipgraph_two_hop_neighbors,
    hipgraph_vertex_pairs_free,
    hipgraph_vertex_pairs_get_first,
    hipgraph_vertex_pairs_get_second,
    hipgraph_vertex_pairs_t,
)
from pylibhipgraph._hipgraph_c.resource_handle cimport (
    bool_t,
    hipgraph_resource_handle_t,
)
from pylibhipgraph.graphs cimport _GPUGraph
from pylibhipgraph.resource_handle cimport ResourceHandle
from pylibhipgraph.utils cimport (
    assert_success,
    copy_to_cupy_array,
    create_hipgraph_type_erased_device_array_view_from_py_obj,
)


def get_two_hop_neighbors(ResourceHandle resource_handle,
                          _GPUGraph graph,
                          start_vertices,
                          bool_t do_expensive_check):
    """
        Compute vertex pairs that are two hops apart. The resulting pairs are
        sorted before returning.

        Parameters
        ----------
        resource_handle : ResourceHandle
            Handle to the underlying device resources needed for referencing data
            and running algorithms.

        graph : SGGraph or MGGraph
            The input graph, for either Single or Multi-GPU operations.

        start_vertices : Optional array of starting vertices
                         If None use all, if specified compute two-hop
                         neighbors for these starting vertices

        Returns
        -------
        return a cupy arrays of 'first' and 'second' or a 'hipgraph_vertex_pairs_t'
        which can be directly passed to the similarity algorithm?
    """

    cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef hipgraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef hipgraph_vertex_pairs_t* result_ptr
    cdef hipgraph_error_code_t error_code
    cdef hipgraph_error_t* error_ptr

    cdef hipgraph_type_erased_device_array_view_t* start_vertices_ptr

    cdef hipgraph_type_erased_device_array_view_t* \
        start_vertices_view_ptr = \
            create_hipgraph_type_erased_device_array_view_from_py_obj(
                start_vertices)

    error_code = hipgraph_two_hop_neighbors(c_resource_handle_ptr,
                                           c_graph_ptr,
                                           start_vertices_view_ptr,
                                           do_expensive_check,
                                           &result_ptr,
                                           &error_ptr)
    assert_success(error_code, error_ptr, "two_hop_neighbors")

    cdef hipgraph_type_erased_device_array_view_t* first_ptr = \
        hipgraph_vertex_pairs_get_first(result_ptr)

    cdef hipgraph_type_erased_device_array_view_t* second_ptr = \
        hipgraph_vertex_pairs_get_second(result_ptr)

    cupy_first = copy_to_cupy_array(c_resource_handle_ptr, first_ptr)
    cupy_second = copy_to_cupy_array(c_resource_handle_ptr, second_ptr)

    # Free all pointers
    hipgraph_vertex_pairs_free(result_ptr)
    if start_vertices is not None:
        hipgraph_type_erased_device_array_view_free(start_vertices_view_ptr)

    return cupy_first, cupy_second
