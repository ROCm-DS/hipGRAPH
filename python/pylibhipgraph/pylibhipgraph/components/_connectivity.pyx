# Copyright (c) 2021-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

from libc.stdint cimport uintptr_t
from libcpp.memory cimport unique_ptr
from pylibhipgraph._hipgraph_c.error cimport hipgraph_error_code_t, hipgraph_error_t
from pylibhipgraph._hipgraph_c.resource_handle cimport (
    bool_t,
    hipgraph_resource_handle_t,
)
from pylibhipgraph.components._connectivity cimport *
from pylibhipgraph.graphs cimport _GPUGraph
from pylibhipgraph.utils cimport assert_success


def _ensure_arg_types(**kwargs):
    """
    Ensure all args have a __cuda_array_interface__ attr
    """
    for (arg_name, arg_val) in kwargs.items():
        # FIXME: remove this special case when weights are supported: weights
        # can only be None
        if arg_name is "weights":
            if arg_val is not None:
                raise TypeError("weights are currently not supported and must "
                                "be None")

        elif not(hasattr(arg_val, "__cuda_array_interface__")):
            raise TypeError(f"{arg_name} must support __cuda_array_interface__")

        if arg_name in ["offsets", "indices", "labels"]:
            typestr = arg_val.__cuda_array_interface__.get("typestr")
            if typestr != "<i4":
                raise TypeError(f"{arg_name} array must have a dtype of int32")


def strongly_connected_components(offsets, indices, weights, num_verts, num_edges, labels):
    """
    Generate the Strongly Connected Components and attach a component label to
    each vertex.

    Parameters
    ----------
    offsets : object supporting a __cuda_array_interface__ interface
        Array containing the offsets values of a Compressed Sparse Row matrix
        that represents the graph.

    indices : object supporting a __cuda_array_interface__ interface
        Array containing the indices values of a Compressed Sparse Row matrix
        that represents the graph.

    weights : object supporting a __cuda_array_interface__ interface
        Array containing the weights values of a Compressed Sparse Row matrix
        that represents the graph.

        NOTE: weighted graphs are currently unsupported, and because of this the
        weights parameter can only be set to None.

    num_verts : int
        The number of vertices present in the graph represented by the CSR
        arrays above.

    num_edges : int
        The number of edges present in the graph represented by the CSR arrays
        above.

    labels : object supporting a __cuda_array_interface__ interface
        Array of size num_verts that will be populated with component label
        values. The component lables in the array are ordered based on the
        sorted vertex ID values of the graph.  For example, labels [9, 9, 7]
        mean vertex 0 is labelled 9, vertex 1 is labelled 9, and vertex 2 is
        labelled 7.

    Returns
    -------
    None

    Examples
    --------
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
    >>> num_verts = scipy_csr.get_shape()[0]
    >>> num_edges = scipy_csr.nnz
    >>>
    >>> cp_offsets = cp.asarray(scipy_csr.indptr)
    >>> cp_indices = cp.asarray(scipy_csr.indices, dtype=np.int32)
    >>> cp_labels = cp.asarray(np.zeros(num_verts, dtype=np.int32))
    >>>
    >>> strongly_connected_components(offsets=cp_offsets,
    ...                               indices=cp_indices,
    ...                               weights=None,
    ...                               num_verts=num_verts,
    ...                               num_edges=num_edges,
    ...                               labels=cp_labels)
    >>> print(f"{len(set(cp_labels.tolist()))} - {cp_labels}")
    5 - [0 1 2 3 4]
    """
    _ensure_arg_types(offsets=offsets, indices=indices,
                      weights=weights, labels=labels)

    cdef unique_ptr[hipgraph_resource_handle_t] handle_ptr
    handle_ptr.reset(new hipgraph_resource_handle_t())
    handle_ = handle_ptr.get()

    cdef hipgraph_error_t* error_ptr
    cdef hipgraph_error_code_t error_code

    # Necessary for matching the "legacy" interface of this call.
    SGGraph graph(handle_, XXX, offsets.__cuda_array_interface__['data'][0],
        indices.__cuda_array_interface__['data'][0], NULL, False, False, False,
        NULL, NULL, 'CSR')

    error_code = hipgraph_strongly_connected_components(handle_, graph, <bool_t>False, <int *>c_labels, error_ptr)
    assert_success(error_code, error_ptr, "hipgraph_strongly_connected_components()")
