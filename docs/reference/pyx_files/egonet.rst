.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.ego_graph, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-ego_graph:

*******************************************
pylibhipgraph.ego_graph
*******************************************

**ego_graph** (*ResourceHandle resource_handle, _GPUGraph graph, source_vertices, size_t radius, bool_t do_expensive_check*)

Compute the induced subgraph of neighbors centered at nodes
source_vertices, within a given radius.

Parameters
----------

resource_handle : ResourceHandle
    Handle to the underlying device resources needed for referencing data
    and running algorithms.

graph : SGGraph or MGGraph
    The input graph.

source_vertices : cupy array
    The centered nodes from which the induced subgraph will be extracted

radius: size_t
    The number of hops to go out from each source vertex

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
    >>> srcs = cupy.asarray([0, 1, 1, 2, 2, 2, 3, 3, 4], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 3, 4, 0, 1, 3, 4, 5, 5], dtype=numpy.int32)
    >>> weights = cupy.asarray(
    ...     [0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2, 6.1], dtype=numpy.float32)
    >>> source_vertices = cupy.asarray([0, 1], dtype=numpy.int32)
    >>> resource_handle = pylibhipgraph.ResourceHandle()
    >>> graph_props = pylibhipgraph.GraphProperties(
    ...     is_symmetric=False, is_multigraph=False)
    >>> G = pylibhipgraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=False, renumber=False, do_expensive_check=False)
    >>> (sources, destinations, edge_weights, subgraph_offsets) =
    ...     pylibhipgraph.ego_graph(resource_handle, G, source_vertices, 2, False)
    # FIXME: update results
    >>> sources
    [0, 1, 1, 3, 1, 1, 3, 3, 4]
    >>> destinations
    [1, 3, 4, 4, 3, 4, 4, 5, 5]
    >>> edge_weights
    [0.1, 2.1, 1.1, 7.2, 2.1, 1.1, 7.2, 3.2, 6.1]
    >>> subgraph_offsets
    [0, 4, 9]
