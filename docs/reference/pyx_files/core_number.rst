.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.core_number, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-core_number:

*******************************************
pylibhipgraph.core_number
*******************************************

**core_number** (*ResourceHandle resource_handle, _GPUGraph graph, degree_type, bool_t do_expensive_check*)

Computes core number.

Parameters
----------

resource_handle: ResourceHandle
    Handle to the underlying device and host resource needed for
    referencing data and running algorithms.

graph : SGGraph or MGGraph
    The input graph, for either Single or Multi-GPU operations.

degree_type: str
    This option determines if the core number computation should be based
    on input, output, or both directed edges, with valid values being
    "incoming", "outgoing", and "bidirectional" respectively.
    This option is currently ignored in this release, and setting it will
    result in a warning.

do_expensive_check: bool
    If True, performs more extensive tests on the inputs to ensure
    validity, at the expense of increased run time.

Returns
-------

A tuple of device arrays, where the first item in the tuple is a device
array containing the vertices and the second item in the tuple is a device
array containing the core numbers for the corresponding vertices.
