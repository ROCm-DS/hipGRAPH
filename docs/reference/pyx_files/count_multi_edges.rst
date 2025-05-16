.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.count_multi_edges, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-count_multi_edges:

*******************************************
pylibhipgraph.count_multi_edges
*******************************************

**count_multi_edges** (*ResourceHandle resource_handle, _GPUGraph graph, bool_t do_expensive_check*)

Count the number of multi-edges in the graph.  This returns
the number of duplicates.  If the edge (u, v) appears k times
in the graph, then that edge will contribute (k-1) toward the
total number of duplicates.

Parameters
----------

resource_handle : ResourceHandle
    Handle to the underlying device resources needed for referencing data
    and running algorithms.

graph : SGGraph or MGGraph
    The input graph, for either Single or Multi-GPU operations.

do_expensive_check : bool_t
    A flag to run expensive checks for input arguments if True.

Returns
-------

Total count of duplicate edges in the graph

Examples
--------

.. code:: Python

    >>> import pylibhipgraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 0, 0], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 1, 1], dtype=numpy.int32)
    >>> weights = cupy.asarray([1.0, 1.0, 1.0], dtype=numpy.float32)
    >>> resource_handle = pylibhipgraph.ResourceHandle()
    >>> graph_props = pylibhipgraph.GraphProperties(
    ...     is_symmetric=False, is_multigraph=False)
    >>> G = pylibhipgraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=True, renumber=False, do_expensive_check=False)
    >>> count = pylibhipgraph.count_multi_edges(resource_handle, G, False)
