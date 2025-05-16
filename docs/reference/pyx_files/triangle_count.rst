.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.triangle_count, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-triangle_count:

*******************************************
pylibhipgraph.triangle_count
*******************************************

**triangle_count** (*ResourceHandle resource_handle, _GPUGraph graph, start_list, bool_t do_expensive_check*)

Computes the number of triangles (cycles of length three) and the number
per vertex in the input graph.

Parameters
----------

resource_handle: ResourceHandle
    Handle to the underlying device and host resources needed for
    referencing data and running algorithms.

graph : SGGraph or MGGraph
    The input graph, for either Single or Multi-GPU operations.

start_list: device array type
    Device array containing the list of vertices for triangle counting.
    If 'None' the entire set of vertices in the graph is processed

do_expensive_check: bool
    If True, performs more extensive tests on the inputs to ensure
    validitity, at the expense of increased run time.

Returns
-------

A tuple of device arrays, where the first item in the tuple is a device
array containing the vertex identifiers and the second item contains the
triangle counting counts

Examples
--------

    # FIXME: No example yet
