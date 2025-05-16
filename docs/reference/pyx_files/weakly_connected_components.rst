.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.weakly_connected_components, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-weakly_connected_components:

*******************************************
pylibhipgraph.weakly_connected_components
*******************************************

**weakly_connected_components** (*ResourceHandle resource_handle, _GPUGraph graph, offsets, indices, weights, labels, bool_t do_expensive_check*)

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

.. code:: Python

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
