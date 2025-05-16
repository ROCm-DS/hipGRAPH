.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.select_random_vertices, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-select_random_vertices:

*******************************************
pylibhipgraph.select_random_vertices
*******************************************

**select_random_vertices** (*ResourceHandle resource_handle, _GPUGraph graph, random_state, size_t num_vertices*)

Select random vertices from the graph

Parameters
----------

resource_handle : ResourceHandle
    Handle to the underlying device resources needed for referencing data
    and running algorithms.

graph : SGGraph or MGGraph
    The input graph, for either Single or Multi-GPU operations.

random_state : int , optional
    Random state to use when generating samples. Optional argument,
    defaults to a hash of process id, time, and hostname.
    (See pylibhipgraph.random.HipGraphRandomState)

num_vertices : size_t , optional
    Number of vertices to sample. Optional argument, defaults to the
    total number of vertices.

Returns
-------

return random vertices from the graph
