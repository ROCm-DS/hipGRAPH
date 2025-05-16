.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.hits, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-hits:

*******************************************
pylibhipgraph.hits
*******************************************

**hits** (*ResourceHandle resource_handle, _GPUGraph graph, double tol, size_t max_iter, initial_hubs_guess_vertices, initial_hubs_guess_values, bool_t normalized, bool_t do_expensive_check*)

Compute HITS hubs and authorities values for each vertex

The HITS algorithm computes two numbers for a node.  Authorities
estimates the node value based on the incoming links.  Hubs estimates
the node value based on outgoing links.

Parameters
----------

resource_handle : ResourceHandle
    Handle to the underlying device resources needed for referencing data
    and running algorithms.

graph : SGGraph or MGGraph
    The input graph, for either Single or Multi-GPU operations.

tol : float, optional (default=1.0e-5)
    Set the tolerance the approximation, this parameter should be a small
    magnitude value.  This parameter is not currently supported.

max_iter : int, optional (default=100)
    The maximum number of iterations before an answer is returned.

initial_hubs_guess_vertices : device array type, optional (default=None)
    Device array containing the pointer to the array of initial hub guess vertices

initial_hubs_guess_values : device array type, optional (default=None)
    Device array containing the pointer to the array of initial hub guess values

normalized : bool, optional (default=True)

do_expensive_check : bool
    If True, performs more extensive tests on the inputs to ensure
    validitity, at the expense of increased run time.

Returns
-------

A tuple of device arrays, where the third item in the tuple is a device
array containing the vertex identifiers, the first and second items are device
arrays containing respectively the hubs and authorities values for the corresponding
vertices
