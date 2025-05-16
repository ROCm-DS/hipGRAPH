.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.katz_centrality, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-katz_centrality:

*******************************************
pylibhipgraph.katz_centrality
*******************************************

**katz_centrality** (*ResourceHandle resource_handle, _GPUGraph graph, betas, double alpha, double beta, double epsilon, size_t max_iterations, bool_t do_expensive_check*)

Compute the Katz centrality for the nodes of the graph. This implementation
is based on a relaxed version of Katz defined by Foster with a reduced
computational complexity of O(n+m)

Parameters
----------

resource_handle : ResourceHandle
    Handle to the underlying device resources needed for referencing data
    and running algorithms.

graph : SGGraph or MGGraph
    The input graph, for either Single or Multi-GPU operations.

betas : device array type
    Device array containing the values to be added to each vertex's new
    Katz Centrality score in every iteration. If set to None then beta is
    used for all vertices.

alpha : double
    The attenuation factor, should be smaller than the inverse of the
    maximum eigenvalue of the graph

beta : double
    Constant value to be added to each vertex's new Katz Centrality score
    in every iteration. Relevant only when betas is None

epsilon : double
    Error tolerance to check convergence

max_iterations: size_t
    Maximum number of Katz Centrality iterations

do_expensive_check : bool_t
    A flag to run expensive checks for input arguments if True.
