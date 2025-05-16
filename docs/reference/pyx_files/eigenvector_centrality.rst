.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.eigenvector_centrality, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-eigenvector_centrality:

*******************************************
pylibhipgraph.eigenvector_centrality
*******************************************

**eigenvector_centrality** (*ResourceHandle resource_handle, _GPUGraph graph, double epsilon, size_t max_iterations, bool_t do_expensive_check*)

Compute the Eigenvector centrality for the nodes of the graph.

Parameters
----------

resource_handle : ResourceHandle
    Handle to the underlying device resources needed for referencing data
    and running algorithms.

graph : SGGraph or MGGraph
    The input graph, for either Single or Multi-GPU operations.

epsilon : double
    Error tolerance to check convergence

max_iterations: size_t
    Maximum number of Eignevector Centrality iterations

do_expensive_check : bool_t
    A flag to run expensive checks for input arguments if True.

Returns
-------

A tuple of device arrays, where the first item in the tuple is a device
array containing the vertices and the second item in the tuple is a device
array containing the eigenvector centrality scores for the corresponding
vertices.

Examples
--------

.. code:: Python

    >>> import pylibhipgraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 2], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 2, 3], dtype=numpy.int32)
    >>> weights = cupy.asarray([1.0, 1.0, 1.0], dtype=numpy.float32)
    >>> resource_handle = pylibhipgraph.ResourceHandle()
    >>> graph_props = pylibhipgraph.GraphProperties(
    ...     is_symmetric=False, is_multigraph=False)
    >>> G = pylibhipgraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=True, renumber=False, do_expensive_check=False)
    >>> (vertices, values) = pylibhipgraph.eigenvector_centrality(
                                resource_handle, G, 1e-6, 1000, False)
