.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.k_core, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-k_core:

*******************************************
pylibhipgraph.k_core
*******************************************

**k_core** (*ResourceHandle resource_handle, _GPUGraph graph, size_t k, degree_type, core_result, bool_t do_expensive_check*)

Compute the k-core of the graph G
A k-core of a graph is a maximal subgraph that
contains nodes of degree k or more. This call does not support a graph
with self-loops and parallel edges.

Parameters
----------

resource_handle: ResourceHandle
    Handle to the underlying device and host resource needed for
    referencing data and running algorithms.

graph : SGGraph or MGGraph
    The input graph, for either Single or Multi-GPU operations.

k : size_t (default=None)
    Order of the core. This value must not be negative. If set to None
    the main core is returned.

degree_type: str
    This option determines if the core number computation should be based
    on input, output, or both directed edges, with valid values being
    "incoming", "outgoing", and "bidirectional" respectively.
    This option is currently ignored in this release, and setting it will
    result in a warning.

core_result : device array type
    Precomputed core number of the nodes of the graph G
    If set to None, the core numbers of the nodes are calculated
    internally.

do_expensive_check: bool
    If True, performs more extensive tests on the inputs to ensure
    validity, at the expense of increased run time.

Returns
-------

A tuple of device arrays contaning the sources, destinations vertices
and the weights.
