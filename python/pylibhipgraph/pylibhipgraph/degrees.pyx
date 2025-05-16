# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

from libc.stdint cimport uintptr_t
from pylibhipgraph._hipgraph_c.array cimport hipgraph_type_erased_device_array_view_t
from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.graph cimport hipgraph_graph_t
from pylibhipgraph._hipgraph_c.graph_functions cimport (
    hipgraph_degrees,
    hipgraph_degrees_result_free,
    hipgraph_degrees_result_get_in_degrees,
    hipgraph_degrees_result_get_out_degrees,
    hipgraph_degrees_result_get_vertices,
    hipgraph_degrees_result_t,
    hipgraph_in_degrees,
    hipgraph_out_degrees,
)
from pylibhipgraph._hipgraph_c.resource_handle cimport (
    bool_t,
    data_type_id_t,
    hipgraph_resource_handle_t,
)
from pylibhipgraph.graphs cimport _GPUGraph
from pylibhipgraph.resource_handle cimport ResourceHandle
from pylibhipgraph.utils cimport (
    assert_CAI_type,
    assert_success,
    copy_to_cupy_array,
    create_hipgraph_type_erased_device_array_view_from_py_obj,
)


def in_degrees(ResourceHandle resource_handle,
               _GPUGraph graph,
               source_vertices,
               bool_t do_expensive_check):
    """
    Compute the in degrees for the nodes of the graph.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    source_vertices : cupy array
        The nodes for which we will compute degrees.

    do_expensive_check : bool_t
        A flag to run expensive checks for input arguments if True.

    Returns
    -------
    A tuple of device arrays, where the first item in the tuple is a device
    array containing the vertices, the second item in the tuple is a device
    array containing the in degrees for the vertices.

    Examples
    --------
    >>> import pylibhipgraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 2], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 2, 3], dtype=numpy.int32)
    >>> weights = cupy.asarray([1.0, 1.0, 1.0], dtype=numpy.float32)
    >>> resource_handle = pylibhipgraph.ResourceHandle()
    >>> graph_props = pylibhipgraph.GraphProperties(
    ...     is_symmetric=False, is_multigraph=False)
    >>> G = pylibhipgraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=True, renumber=False, do_expensive_check=False)
    >>> (vertices, in_degrees) = pylibhipgraph.in_degrees(
                                   resource_handle, G, None, False)

    """

    cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef hipgraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef hipgraph_degrees_result_t* result_ptr
    cdef hipgraph_error_code_t error_code
    cdef hipgraph_error_t* error_ptr

    assert_CAI_type(source_vertices, "source_vertices", True)

    cdef hipgraph_type_erased_device_array_view_t* \
        source_vertices_ptr = \
            create_hipgraph_type_erased_device_array_view_from_py_obj(
                source_vertices)

    error_code = hipgraph_in_degrees(c_resource_handle_ptr,
                                    c_graph_ptr,
                                    source_vertices_ptr,
                                    do_expensive_check,
                                    &result_ptr,
                                    &error_ptr)
    assert_success(error_code, error_ptr, "hipgraph_in_degrees")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef hipgraph_type_erased_device_array_view_t* vertices_ptr = \
        hipgraph_degrees_result_get_vertices(result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* in_degrees_ptr = \
        hipgraph_degrees_result_get_in_degrees(result_ptr)

    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    cupy_in_degrees = copy_to_cupy_array(c_resource_handle_ptr, in_degrees_ptr)

    hipgraph_degrees_result_free(result_ptr)

    return (cupy_vertices, cupy_in_degrees)

def out_degrees(ResourceHandle resource_handle,
                _GPUGraph graph,
                source_vertices,
                bool_t do_expensive_check):
    """
    Compute the out degrees for the nodes of the graph.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    source_vertices : cupy array
        The nodes for which we will compute degrees.

    do_expensive_check : bool_t
        A flag to run expensive checks for input arguments if True.

    Returns
    -------
    A tuple of device arrays, where the first item in the tuple is a device
    array containing the vertices, the second item in the tuple is a device
    array containing the out degrees for the vertices.

    Examples
    --------
    >>> import pylibhipgraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 2], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 2, 3], dtype=numpy.int32)
    >>> weights = cupy.asarray([1.0, 1.0, 1.0], dtype=numpy.float32)
    >>> resource_handle = pylibhipgraph.ResourceHandle()
    >>> graph_props = pylibhipgraph.GraphProperties(
    ...     is_symmetric=False, is_multigraph=False)
    >>> G = pylibhipgraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=True, renumber=False, do_expensive_check=False)
    >>> (vertices, out_degrees) = pylibhipgraph.out_degrees(
                                    resource_handle, G, None, False)

    """

    cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef hipgraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef hipgraph_degrees_result_t* result_ptr
    cdef hipgraph_error_code_t error_code
    cdef hipgraph_error_t* error_ptr

    assert_CAI_type(source_vertices, "source_vertices", True)

    cdef hipgraph_type_erased_device_array_view_t* \
        source_vertices_ptr = \
            create_hipgraph_type_erased_device_array_view_from_py_obj(
                source_vertices)

    error_code = hipgraph_out_degrees(c_resource_handle_ptr,
                                     c_graph_ptr,
                                     source_vertices_ptr,
                                     do_expensive_check,
                                     &result_ptr,
                                     &error_ptr)
    assert_success(error_code, error_ptr, "hipgraph_out_degrees")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef hipgraph_type_erased_device_array_view_t* vertices_ptr = \
        hipgraph_degrees_result_get_vertices(result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* out_degrees_ptr = \
        hipgraph_degrees_result_get_out_degrees(result_ptr)

    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    cupy_out_degrees = copy_to_cupy_array(c_resource_handle_ptr, out_degrees_ptr)

    hipgraph_degrees_result_free(result_ptr)

    return (cupy_vertices, cupy_out_degrees)


def degrees(ResourceHandle resource_handle,
            _GPUGraph graph,
            source_vertices,
            bool_t do_expensive_check):
    """
    Compute the degrees for the nodes of the graph.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    source_vertices : cupy array
        The nodes for which we will compute degrees.

    do_expensive_check : bool_t
        A flag to run expensive checks for input arguments if True.

    Returns
    -------
    A tuple of device arrays, where the first item in the tuple is a device
    array containing the vertices, the second item in the tuple is a device
    array containing the in degrees for the vertices, the third item in the
    tuple is a device array containing the out degrees for the vertices.

    Examples
    --------
    >>> import pylibhipgraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 2], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 2, 3], dtype=numpy.int32)
    >>> weights = cupy.asarray([1.0, 1.0, 1.0], dtype=numpy.float32)
    >>> resource_handle = pylibhipgraph.ResourceHandle()
    >>> graph_props = pylibhipgraph.GraphProperties(
    ...     is_symmetric=False, is_multigraph=False)
    >>> G = pylibhipgraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=True, renumber=False, do_expensive_check=False)
    >>> (vertices, in_degrees, out_degrees) = pylibhipgraph.degrees(
                                                resource_handle, G, None, False)

    """

    cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef hipgraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef hipgraph_degrees_result_t* result_ptr
    cdef hipgraph_error_code_t error_code
    cdef hipgraph_error_t* error_ptr

    assert_CAI_type(source_vertices, "source_vertices", True)

    cdef hipgraph_type_erased_device_array_view_t* \
        source_vertices_ptr = \
            create_hipgraph_type_erased_device_array_view_from_py_obj(
                source_vertices)

    error_code = hipgraph_degrees(c_resource_handle_ptr,
                                 c_graph_ptr,
                                 source_vertices_ptr,
                                 do_expensive_check,
                                 &result_ptr,
                                 &error_ptr)
    assert_success(error_code, error_ptr, "hipgraph_degrees")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef hipgraph_type_erased_device_array_view_t* vertices_ptr = \
        hipgraph_degrees_result_get_vertices(result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* in_degrees_ptr = \
        hipgraph_degrees_result_get_in_degrees(result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* out_degrees_ptr = \
        hipgraph_degrees_result_get_out_degrees(result_ptr)

    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    cupy_in_degrees = copy_to_cupy_array(c_resource_handle_ptr, in_degrees_ptr)
    cupy_out_degrees = copy_to_cupy_array(c_resource_handle_ptr, out_degrees_ptr)

    hipgraph_degrees_result_free(result_ptr)

    return (cupy_vertices, cupy_in_degrees, cupy_out_degrees)
