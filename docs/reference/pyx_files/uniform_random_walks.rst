.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.uniform_random_walks, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-uniform_random_walks:

*******************************************
pylibhipgraph.uniform_random_walks
*******************************************

**uniform_random_walks** (*ResourceHandle resource_handle, _GPUGraph input_graph, start_vertices, size_t max_length*)

Compute uniform random walks for each nodes in 'start_vertices'

Parameters
----------

resource_handle: ResourceHandle
    Handle to the underlying device and host resources needed for
    referencing data and running algorithms.

input_graph : SGGraph or MGGraph
    The input graph, for either Single or Multi-GPU operations.

start_vertices: device array type
    Device array containing the list of starting vertices from which
    to run the uniform random walk

max_length: size_t
    The maximum depth of the uniform random walks


Returns
-------

A tuple containing two device arrays and an size_t which are respectively
the vertices path, the edge path weights and the maximum path length
