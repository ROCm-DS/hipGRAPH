.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.bfs, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-bfs:

*******************************************
pylibhipgraph.bfs
*******************************************

**bfs** (*ResourceHandle handle, _GPUGraph graph, sources, bool_t direction_optimizing, int32_t depth_limit, bool_t compute_predecessors, bool_t do_expensive_check*)

Performs a Breadth-first search starting from the provided sources.
Returns the distances, and predecessors if requested.

Parameters
----------

handle: ResourceHandle
    The resource handle responsible for managing device resources
    that this algorithm will use

graph: SGGraph or MGGraph
    The graph to operate upon

sources: cudf.Series
    The vertices to start the breadth-first search from.  Should
    match the numbering of the provided graph.  All workers must
    have a unique set of sources. Empty sets are allowed as long
    as at least one worker has a source.

direction_optimizing: bool_t
    Whether to treat the graph as undirected (should only be called
    on a symmetric graph)

depth_limit: int32_t
    The depth limit at which the traversal will be stopped.  If this
    is a negative number, the traversal will run without a depth limit.

compute_predecessors: bool_t
    Whether to compute the predecessors.  If left blank, -1 will be
    returned instead of the correct predecessor of each vertex.

do_expensive_check : bool_t
    If True, performs more extensive tests on the inputs to ensure
    validitity, at the expense of increased run time.


Returns
-------

A tuple of device arrays (cupy arrays) of the form
(distances, predecessors, vertices)

Examples
--------

.. code:: Python

    >>> M = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = hipgraph.Graph()
    >>>  G.from_cudf_edgelist(M, source='0', destination='1', edge_attr='2')
    >>>
    >>> handle = ResourceHandle()
    >>>
    >>> srcs = G.edgelist.edgelist_df['src']
    >>> dsts = G.edgelist.edgelist_df['dst']
    >>> weights = G.edgelist.edgelist_df['weights']
    >>>
    >>> sg = SGGraph(
    >>>     resource_handle = handle,
    >>>     graph_properties = GraphProperties(is_multigraph=G.is_multigraph()),
    >>>     src_array = srcs,
    >>>     dst_array = dsts,
    >>>     weight_array = weights,
    >>>     store_transposed=False,
    >>>     renumber=False,
    >>>     do_expensive_check=do_expensive_check
    >>> )
    >>>
    >>> res = pylibhipgraph_bfs(
    >>>         handle,
    >>>         sg,
    >>>         cudf.Series([0], dtype='int32'),
    >>>         False,
    >>>         10,
    >>>         True,
    >>>         False
    >>> )
    >>>
    >>> distances, predecessors, vertices = res
    >>>
    f>>> inal_results = cudf.DataFrame({
    >>>     'distance': cudf.Series(distances),
    >>>     'vertex': cudf.Series(vertices),
    >>>     'predecessor': cudf.Series(predecessors),
    >>> })
