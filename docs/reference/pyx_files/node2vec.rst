.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.node2vec, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-node2vec:

*******************************************
pylibhipgraph.node2vec
*******************************************

**node2vec** (*ResourceHandle resource_handle, _GPUGraph graph, seed_array, size_t max_depth, bool_t compress_result, double p, double q*)

Computes random walks under node2vec sampling procedure.

Parameters
----------

resource_handle : ResourceHandle
    Handle to the underlying device resources needed for referencing data
    and running algorithms.

graph : SGGraph
    The input graph.

seed_array: device array type
    Device array containing the pointer to the array of seed vertices.

max_depth : size_t
    Maximum number of vertices in generated path

compress_result : bool_t
    If true, the paths are unpadded and a third return device array contains
    the sizes for each path, otherwise the paths are padded and the third
    return device array is empty.

p : double
    The return factor p represents the likelihood of backtracking to a node
    in the walk. A higher value (> max(q, 1)) makes it less likely to sample
    a previously visited node, while a lower value (< min(q, 1)) would make it
    more likely to backtrack, making the walk more "local".

q : double
    The in-out factor q represents the likelihood of visiting nodes closer or
    further from the outgoing node. If q > 1, the random walk is likelier to
    visit nodes closer to the outgoing node. If q < 1, the random walk is
    likelier to visit nodes further from the outgoing node.

Returns
-------

A tuple of device arrays, where the first item in the tuple is a device
array containing the compressed paths, the second item is a device
array containing the corresponding weights for each edge traversed in
each path, and the third item is a device array containing the sizes
for each of the compressed paths, if compress_result is True.

Examples
--------

.. code:: Python

    >>> import pylibhipgraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 2], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 2, 3], dtype=numpy.int32)
    >>> seeds = cupy.asarray([0, 0, 1], dtype=numpy.int32)
    >>> weights = cupy.asarray([1.0, 1.0, 1.0], dtype=numpy.float32)
    >>> resource_handle = pylibhipgraph.ResourceHandle()
    >>> graph_props = pylibhipgraph.GraphProperties(
    ...     is_symmetric=False, is_multigraph=False)
    >>> G = pylibhipgraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=False, renumber=False, do_expensive_check=False)
    >>> (paths, weights, sizes) = pylibhipgraph.node2vec(
    ...                             resource_handle, G, seeds, 3, True, 1.0, 1.0)
