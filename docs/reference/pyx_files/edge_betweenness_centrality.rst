.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.edge_betweenness_centrality, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-edge_betweenness_centrality:

*******************************************
pylibhipgraph.edge_betweenness_centrality
*******************************************

**edge_betweenness_centrality** (*ResourceHandle resource_handle _GPUGraph graph, k random_state, bool_t normalized, bool_t do_expensive_check*)

Compute the edge betweenness centrality for all edges of the graph G.
Betweenness centrality is a measure of the number of shortest paths
that pass over an edge.  An edge with a high betweenness centrality
score has more paths passing over it and is therefore believed to be
more important.

Parameters
----------

resource_handle : ResourceHandle
    Handle to the underlying device resources needed for referencing data
    and running algorithms.

graph : SGGraph or MGGraph
    The input graph, for either Single or Multi-GPU operations.

k : int or device array type or None, optional (default=None)
    If k is not None, use k node samples to estimate the edge betweenness.
    Higher values give better approximation.  If k is a device array type,
    the contents are assumed to be vertex identifiers to be used for estimation.
    If k is None (the default), all the vertices are used to estimate the edge
    betweenness.  Vertices obtained through sampling or defined as a list will
    be used as sources for traversals inside the algorithm.

random_state : int, optional (default=None)
    if k is specified and k is an integer, use random_state to initialize the
    random number generator.
    Using None defaults to a hash of process id, time, and hostname
    If k is either None or list or cudf objects: random_state parameter is
    ignored.

normalized : bool_t
    Normalization will ensure that values are in [0, 1].

do_expensive_check : bool_t
    A flag to run expensive checks for input arguments if True.

Returns
-------

A tuple of device arrays corresponding to the sources, destinations, edge
betweenness centrality scores and edge ids (if provided).

array containing the vertices and the second item in the tuple is a device
array containing the eigenvector centrality scores for the corresponding
vertices.

Examples
--------

.. code:: Python

    >>> import pylibhipgraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5],
    ...     dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4],
    ...     dtype=numpy.int32)
    >>> edge_ids = cupy.asarray(
    ...     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    ...     dtype=numpy.int32)
    >>> resource_handle = pylibhipgraph.ResourceHandle()
    >>> graph_props = pylibhipgraph.GraphProperties(
    ...     is_symmetric=False, is_multigraph=False)
    >>> G = pylibhipgraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, store_transposed=False,
    ...     renumber=False, do_expensive_check=False, edge_id_array=edge_ids)
    >>> (srcs, dsts, values, edge_ids) = pylibhipgraph.edge_betweenness_centrality(
            resource_handle, G, None, None, True, False)
    >>> srcs
    [0 0 1 1 1 1 2 2 2 3 3 3 4 4 5 5]
    >>> dsts
    [1 2 0 2 3 4 0 1 3 1 2 5 1 5 3 4]
    >>> values
    [0.10555556 0.06111111 0.10555556 0.06666667 0.09444445 0.14444445
     0.06111111 0.06666667 0.09444445 0.09444445 0.09444445 0.12222222
     0.14444445 0.07777778 0.12222222 0.07777778]
    >>> edge_ids
    [ 0 11  8 12  1  2  3  4  5  9 13  6 10  7 14 15]
