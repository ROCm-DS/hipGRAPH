# Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3


from pylibhipgraph._hipgraph_c.array cimport (
    hipgraph_type_erased_device_array_t,
    hipgraph_type_erased_device_array_view,
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
from pylibhipgraph._hipgraph_c.random cimport hipgraph_rng_state_t
from pylibhipgraph._hipgraph_c.resource_handle cimport hipgraph_resource_handle_t
from pylibhipgraph._hipgraph_c.sampling_algorithms cimport (
    hipgraph_select_random_vertices,
)
from pylibhipgraph.graphs cimport _GPUGraph
from pylibhipgraph.random cimport HipGraphRandomState
from pylibhipgraph.resource_handle cimport ResourceHandle
from pylibhipgraph.utils cimport assert_success, copy_to_cupy_array


def select_random_vertices(ResourceHandle resource_handle,
                           _GPUGraph graph,
                           random_state,
                           size_t num_vertices,
                           ):
    """
    Select random vertices from the graph

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    random_state : int , optional
        Random state to use when generating samples. Optional argument,
        defaults to a hash of process id, time, and hostname.
        (See pylibhipgraph.random.HipGraphRandomState)

    num_vertices : size_t , optional
        Number of vertices to sample. Optional argument, defaults to the
        total number of vertices.

    Returns
    -------
    return random vertices from the graph
    """

    cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef hipgraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef hipgraph_type_erased_device_array_t* vertices_ptr
    cdef hipgraph_error_code_t error_code
    cdef hipgraph_error_t* error_ptr

    cg_rng_state = HipGraphRandomState(resource_handle, random_state)

    cdef hipgraph_rng_state_t* rng_state_ptr = \
        cg_rng_state.rng_state_ptr

    error_code = hipgraph_select_random_vertices(c_resource_handle_ptr,
                                                c_graph_ptr,
                                                rng_state_ptr,
                                                num_vertices,
                                                &vertices_ptr,
                                                &error_ptr)
    assert_success(error_code, error_ptr, "select_random_vertices")

    cdef hipgraph_type_erased_device_array_view_t* \
        vertices_view_ptr = \
            hipgraph_type_erased_device_array_view(
                vertices_ptr)

    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_view_ptr)

    return cupy_vertices
