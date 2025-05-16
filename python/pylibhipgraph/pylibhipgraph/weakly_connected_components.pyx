# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibhipgraph import GraphProperties, SGGraph

from pylibhipgraph._hipgraph_c.array cimport (
    hipgraph_type_erased_device_array_view_copy,
    hipgraph_type_erased_device_array_view_t,
)
from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.graph cimport hipgraph_graph_t
from pylibhipgraph._hipgraph_c.labeling_algorithms cimport (
    hipgraph_labeling_result_free,
    hipgraph_labeling_result_get_labels,
    hipgraph_labeling_result_get_vertices,
    hipgraph_labeling_result_t,
    hipgraph_weakly_connected_components,
)
from pylibhipgraph._hipgraph_c.resource_handle cimport (
    bool_t,
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


def _ensure_args(graph, offsets, indices, weights):
    i = 0
    if graph is not None:
        # ensure the remaining parametes are None
        invalid_input = [i for p in [offsets, indices, weights] if p is not None]
        input_type = "graph"
    else:
        invalid_input = [i for p in [offsets, indices] if p is None]
        input_type = "csr_arrays"


    if len(invalid_input) != 0:
        raise TypeError("Invalid input combination: Must set either 'graph' or "
                        "a combination of 'offsets', 'indices' and 'weights', not both")
    else:
        if input_type == "csr_arrays":
            assert_CAI_type(offsets, "offsets")
            assert_CAI_type(indices, "indices")
            assert_CAI_type(weights, "weights", True)

    return input_type


def weakly_connected_components(ResourceHandle resource_handle,
                                _GPUGraph graph,
                                offsets,
                                indices,
                                weights,
                                labels,
                                bool_t do_expensive_check):
    """
    Generate the Weakly Connected Components from either an input graph or
    or CSR arrays('offsets', 'indices', 'weights') and attach a component label
    to each vertex.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph.

    offsets : object supporting a __cuda_array_interface__ interface
        Array containing the offsets values of a Compressed Sparse Row matrix
        that represents the graph.

    indices : object supporting a __cuda_array_interface__ interface
        Array containing the indices values of a Compressed Sparse Row matrix
        that represents the graph.

    weights : object supporting a __cuda_array_interface__ interface
        Array containing the weights values of a Compressed Sparse Row matrix
        that represents the graph

    do_expensive_check : bool_t
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.

    Returns
    -------
    A tuple containing containing two device arrays which are respectively
    vertices and their corresponding labels

    Examples
    --------
    >>> import pylibhipgraph, cupy, numpy
    >>> from pylibhipgraph import weakly_connected_components
    >>> srcs = cupy.asarray([0, 1, 1, 2, 2, 0], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 0, 2, 1, 0, 2], dtype=numpy.int32)
    >>> weights = cupy.asarray(
    ...     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=numpy.float32)
    >>> resource_handle = pylibhipgraph.ResourceHandle()
    >>> graph_props = pylibhipgraph.GraphProperties(
    ...      is_symmetric=True, is_multigraph=False)
    >>> G = pylibhipgraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=False, renumber=True, do_expensive_check=False)
    >>> (vertices, labels) = weakly_connected_components(
    ...     resource_handle, G, None, None, None, None, False)

    >>> vertices
    [0, 1, 2]
    >>> labels
    [2, 2, 2]

    >>> import cupy as cp
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>>
    >>> graph = [
    ... [0, 1, 1, 0, 0],
    ... [0, 0, 1, 0, 0],
    ... [0, 0, 0, 0, 0],
    ... [0, 0, 0, 0, 1],
    ... [0, 0, 0, 0, 0],
    ... ]
    >>> scipy_csr = csr_matrix(graph)
    >>> rows, cols = scipy_csr.nonzero()
    >>> scipy_csr[cols, rows] = scipy_csr[rows, cols]
    >>>
    >>> cp_offsets = cp.asarray(scipy_csr.indptr)
    >>> cp_indices = cp.asarray(scipy_csr.indices, dtype=np.int32)
    >>>
    >>> resource_handle = pylibhipgraph.ResourceHandle()
    >>> weakly_connected_components(resource_handle=resource_handle,
                                    graph=None,
    ...                             offsets=cp_offsets,
    ...                             indices=cp_indices,
    ...                             weights=None,
    ...                             False)
    >>> print(f"{len(set(cp_labels.tolist()))} - {cp_labels}")
    2 - [2 2 2 4 4]

    """

    # FIXME: Remove this function once the deprecation is completed
    input_type = _ensure_args(graph, offsets, indices, weights)

    if input_type == "csr_arrays":
        if resource_handle is None:
            # Get a default handle
            resource_handle = ResourceHandle()

        graph_props = GraphProperties(
        is_symmetric=True, is_multigraph=False)
        graph = SGGraph(
                resource_handle,
                graph_props,
                offsets,
                indices,
                weights,
                store_transposed=False,
                renumber=False,
                do_expensive_check=True,
                input_array_format="CSR"
            )

    cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr

    cdef hipgraph_graph_t* c_graph_ptr = graph.c_graph_ptr
    cdef hipgraph_labeling_result_t* result_ptr
    cdef hipgraph_error_code_t error_code
    cdef hipgraph_error_t* error_ptr

    error_code = hipgraph_weakly_connected_components(c_resource_handle_ptr,
                                                     c_graph_ptr,
                                                     do_expensive_check,
                                                     &result_ptr,
                                                     &error_ptr)
    assert_success(error_code, error_ptr, "hipgraph_weakly_connected_components")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef hipgraph_type_erased_device_array_view_t* vertices_ptr = \
        hipgraph_labeling_result_get_vertices(result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* labels_ptr = \
        hipgraph_labeling_result_get_labels(result_ptr)

    cdef hipgraph_type_erased_device_array_view_t* labels_view_ptr
    if labels is not None:
        labels_view_ptr = create_hipgraph_type_erased_device_array_view_from_py_obj(
            labels)
        hipgraph_type_erased_device_array_view_copy(
            c_resource_handle_ptr,
            labels_view_ptr,
            labels_ptr,
            &error_ptr
        )
        assert_success(
            error_code, error_ptr, "hipgraph_type_erased_device_array_view_copy")
        hipgraph_labeling_result_free(result_ptr)
    else:
        cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
        cupy_labels = copy_to_cupy_array(c_resource_handle_ptr, labels_ptr)
        hipgraph_labeling_result_free(result_ptr)
        return (cupy_vertices, cupy_labels)
