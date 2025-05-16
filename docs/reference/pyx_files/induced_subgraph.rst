.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.induced_subgraph, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-induced_subgraph:

*******************************************
pylibhipgraph.induced_subgraph
*******************************************

**induced_subgraph** (*ResourceHandle resource_handle, _GPUGraph graph, subgraph_vertices, subgraph_offsets, bool_t do_expensive_check*)

extract a list of edges that represent the subgraph
containing only the specified vertex ids.

Parameters
----------

resource_handle : ResourceHandle
    Handle to the underlying device resources needed for referencing data
    and running algorithms.

graph : SGGraph or MGGraph
    The input graph.

subgraph_vertices : cupy array
    array of vertices to include in extracted subgraph.

subgraph_offsets : cupy array
    array of subgraph offsets into subgraph_vertices.

do_expensive_check : bool_t
    If True, performs more extensive tests on the inputs to ensure
    validitity, at the expense of increased run time.

Returns
-------

A tuple of device arrays containing the sources, destinations, edge_weights
and the subgraph_offsets(if there are more than one seeds)

Examples
--------

.. code:: Python

    >>> import pylibhipgraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 1, 2, 2, 2, 3, 4], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 3, 4, 0, 1, 3, 5, 5], dtype=numpy.int32)
    >>> weights = cupy.asarray(
    ...     [0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2], dtype=numpy.float32)
    >>> subgraph_vertices = cupy.asarray([0, 1, 2, 3], dtype=numpy.int32)
    >>> subgraph_offsets = cupy.asarray([0, 4], dtype=numpy.int32)
    >>> resource_handle = pylibhipgraph.ResourceHandle()
    >>> graph_props = pylibhipgraph.GraphProperties(
    ...     is_symmetric=False, is_multigraph=False)
    >>> G = pylibhipgraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=False, renumber=False, do_expensive_check=False)
    >>> (sources, destinations, edge_weights, subgraph_offsets) =
    ...     pylibhipgraph.induced_subgraph(
    ...         resource_handle, G, subgraph_vertices, subgraph_offsets, False)
    >>> sources
    [0, 1, 2, 2, 2]
    >>> destinations
    [1, 3, 0, 1, 3]
    >>> edge_weights
    [0.1, 2.1, 5.1, 3.1, 4.1]
    >>> subgraph_offsets
    [0, 5]
