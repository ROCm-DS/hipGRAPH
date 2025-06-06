# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Have cython use python 3 syntax
# cython: language_level = 3

from libc.stdint cimport uintptr_t
from pylibhipgraph._hipgraph_c.algorithms cimport (
    hipgraph_node2vec,
    hipgraph_random_walk_result_free,
    hipgraph_random_walk_result_get_path_sizes,
    hipgraph_random_walk_result_get_paths,
    hipgraph_random_walk_result_get_weights,
    hipgraph_random_walk_result_t,
)
from pylibhipgraph._hipgraph_c.array cimport (
    hipgraph_type_erased_device_array_view_create,
    hipgraph_type_erased_device_array_view_free,
    hipgraph_type_erased_device_array_view_t,
)
from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.graph cimport hipgraph_graph_t
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
    get_c_type_from_numpy_type,
)


def node2vec(ResourceHandle resource_handle,
            _GPUGraph graph,
            seed_array,
            size_t max_depth,
            bool_t compress_result,
            double p,
            double q):
    """
    Computes random walks under node2vec sampling procedure.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph
        The input graph.

    seed_array: device array type
        Device array containing the pointer to the array of seed vertices.

    max_depth : size_t
        Maximum number of vertices in generated path

    compress_result : bool_t
        If true, the paths are unpadded and a third return device array contains
        the sizes for each path, otherwise the paths are padded and the third
        return device array is empty.

    p : double
        The return factor p represents the likelihood of backtracking to a node
        in the walk. A higher value (> max(q, 1)) makes it less likely to sample
        a previously visited node, while a lower value (< min(q, 1)) would make it
        more likely to backtrack, making the walk more "local".

    q : double
        The in-out factor q represents the likelihood of visiting nodes closer or
        further from the outgoing node. If q > 1, the random walk is likelier to
        visit nodes closer to the outgoing node. If q < 1, the random walk is
        likelier to visit nodes further from the outgoing node.

    Returns
    -------
    A tuple of device arrays, where the first item in the tuple is a device
    array containing the compressed paths, the second item is a device
    array containing the corresponding weights for each edge traversed in
    each path, and the third item is a device array containing the sizes
    for each of the compressed paths, if compress_result is True.

    Examples
    --------
    >>> import pylibhipgraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 2], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 2, 3], dtype=numpy.int32)
    >>> seeds = cupy.asarray([0, 0, 1], dtype=numpy.int32)
    >>> weights = cupy.asarray([1.0, 1.0, 1.0], dtype=numpy.float32)
    >>> resource_handle = pylibhipgraph.ResourceHandle()
    >>> graph_props = pylibhipgraph.GraphProperties(
    ...     is_symmetric=False, is_multigraph=False)
    >>> G = pylibhipgraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=False, renumber=False, do_expensive_check=False)
    >>> (paths, weights, sizes) = pylibhipgraph.node2vec(
    ...                             resource_handle, G, seeds, 3, True, 1.0, 1.0)

    """

    # FIXME: import these modules here for now until a better pattern can be
    # used for optional imports (perhaps 'import_optional()' from hipgraph), or
    # these are made hard dependencies.
    try:
        import cupy
    except ModuleNotFoundError:
        raise RuntimeError("node2vec requires the cupy package, which could not "
                           "be imported")
    assert_CAI_type(seed_array, "seed_array")

    cdef hipgraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef hipgraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef hipgraph_random_walk_result_t* result_ptr
    cdef hipgraph_error_code_t error_code
    cdef hipgraph_error_t* error_ptr

    cdef uintptr_t cai_seed_ptr = \
        seed_array.__cuda_array_interface__["data"][0]
    cdef hipgraph_type_erased_device_array_view_t* seed_view_ptr = \
        hipgraph_type_erased_device_array_view_create(
            <void*>cai_seed_ptr,
            len(seed_array),
            get_c_type_from_numpy_type(seed_array.dtype))

    error_code = hipgraph_node2vec(c_resource_handle_ptr,
                                  c_graph_ptr,
                                  seed_view_ptr,
                                  max_depth,
                                  compress_result,
                                  p,
                                  q,
                                  &result_ptr,
                                  &error_ptr)
    assert_success(error_code, error_ptr, "hipgraph_node2vec")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef hipgraph_type_erased_device_array_view_t* paths_ptr = \
        hipgraph_random_walk_result_get_paths(result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* weights_ptr = \
        hipgraph_random_walk_result_get_weights(result_ptr)
    cdef hipgraph_type_erased_device_array_view_t* path_sizes_ptr = \
        hipgraph_random_walk_result_get_path_sizes(result_ptr)

    cupy_paths = copy_to_cupy_array(c_resource_handle_ptr, paths_ptr)
    cupy_weights = copy_to_cupy_array(c_resource_handle_ptr, weights_ptr)
    cupy_path_sizes = copy_to_cupy_array(c_resource_handle_ptr,
                                           path_sizes_ptr)

    hipgraph_random_walk_result_free(result_ptr)
    hipgraph_type_erased_device_array_view_free(seed_view_ptr)

    return (cupy_paths, cupy_weights, cupy_path_sizes)
