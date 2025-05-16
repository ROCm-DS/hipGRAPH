.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.betweenness_centrality, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-betweenness_centrality:

*******************************************
pylibhipgraph.betweenness_centrality
*******************************************

**betweenness_centrality** (*ResourceHandle resource_handle, _GPUGraph graph, k, random_state, bool_t normalized bool_t include_endpoints, bool_t do_expensive_check*)

Compute the betweenness centrality for all vertices of the graph G.
Betweenness centrality is a measure of the number of shortest paths that
pass through a vertex.  A vertex with a high betweenness centrality score
has more paths passing through it and is therefore believed to be more
important.

Parameters
----------
resource_handle : ResourceHandle
    Handle to the underlying device resources needed for referencing data
    and running algorithms.

graph : SGGraph or MGGraph
    The input graph, for either Single or Multi-GPU operations.

k : int or device array type or None, optional (default=None)
    If k is not None, use k node samples to estimate betweenness.  Higher
    values give better approximation.  If k is a device array type,
    use the content of the list for estimation: the list should contain
    vertex identifiers. If k is None (the default), all the vertices are
    used to estimate betweenness.  Vertices obtained through sampling or
    defined as a list will be used as sources for traversals inside the
    algorithm.

random_state : int, optional (default=None)
    if k is specified and k is an integer, use random_state to initialize the
    random number generator.
    Using None defaults to a hash of process id, time, and hostname
    If k is either None or list or cudf objects: random_state parameter is
    ignored.

normalized : bool_t
    Normalization will ensure that values are in [0, 1].

include_endpoints : bool_t
    If true, include the endpoints in the shortest path counts.

do_expensive_check : bool_t
    A flag to run expensive checks for input arguments if True.
