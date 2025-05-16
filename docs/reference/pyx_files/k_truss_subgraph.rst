.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.k_truss_subgraph, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-k_truss_subgraph:

*******************************************
pylibhipgraph.k_truss_subgraph
*******************************************

**k_truss_subgraph** (*ResourceHandle resource_handle, _GPUGraph graph, size_t k, bool_t do_expensive_check*)

Extract k truss of a graph for a specific k.

Parameters
----------

resource_handle : ResourceHandle
    Handle to the underlying device resources needed for referencing data
    and running algorithms.

graph : SGGraph or MGGraph
    The input graph.

k: size_t
    The desired k to be used for extracting the k-truss subgraph.

do_expensive_check : bool_t
    If True, performs more extensive tests on the inputs to ensure
    validitity, at the expense of increased run time.

Returns
-------

A tuple of device arrays containing the sources, destinations,
edge_weights and edge_offsets.

Examples
--------

.. code:: Python

    >>> import pylibhipgraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 1, 3, 1, 4, 2, 0, 2, 1, 2,
    ...     3, 3, 4, 3, 5, 4, 5], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 0, 3, 1, 4, 1, 0, 2, 1, 2, 3,
    ...     2, 4, 3, 5, 3, 5, 4], dtype=numpy.int32)
    >>> weights = cupy.asarray(
    ...     [0.1, 0.1, 2.1, 2.1, 1.1, 1.1, 7.2, 7.2, 2.1, 2.1,
    ...     1.1, 1.1, 7.2, 7.2, 3.2, 3.2, 6.1, 6.1]
    ...     ,dtype=numpy.float32)
    >>> k = 2
    >>> resource_handle = pylibhipgraph.ResourceHandle()
    >>> graph_props = pylibhipgraph.GraphProperties(
    ...     is_symmetric=True, is_multigraph=False)
    >>> G = pylibhipgraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=False, renumber=False, do_expensive_check=False)
    >>> (sources, destinations, edge_weights, subgraph_offsets) =
    ...     pylibhipgraph.k_truss_subgraph(resource_handle, G, k, False)
    >>> sources
    [0 0 1 1 1 1 2 2 2 3 3 3 3 4 4 4 5 5]
    >>> destinations
    [1 2 0 2 3 4 0 1 3 1 2 4 5 1 3 5 3 4]
    >>> edge_weights
    [0.1 7.2 0.1 2.1 2.1 1.1 7.2 2.1 1.1 2.1 1.1 7.2 3.2 1.1 7.2 6.1 3.2 6.1]
    >>> subgraph_offsets
    [0 18]
