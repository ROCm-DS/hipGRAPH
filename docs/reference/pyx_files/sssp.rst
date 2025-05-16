.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.sssp, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-sssp:

*******************************************
pylibhipgraph.sssp
*******************************************

**sssp** (*ResourceHandle resource_handle, _GPUGraph graph, size_t source, double cutoff, bool_t compute_predecessors, bool_t do_expensive_check*)

Compute the distance and predecessors for shortest paths from the specified
source to all the vertices in the graph. The returned distances array will
contain the distance from the source to each vertex in the returned vertex
array at the same index. The returned predecessors array will contain the
previous vertex in the shortest path for each vertex in the vertex array at
the same index. Vertices that are unreachable will have a distance of
infinity denoted by the maximum value of the data type and the predecessor
set as -1. The source vertex predecessor will be set to -1. Graphs with
negative weight cycles are not supported.

Parameters
----------

resource_handle : ResourceHandle
    Handle to the underlying device resources needed for referencing data
    and running algorithms.

graph : SGGraph or MGGraph
    The input graph, for either Single or Multi-GPU operations.

source :
    The vertex identifier of the source vertex.

cutoff :
    Maximum edge weight sum to consider.

compute_predecessors : bool
    This parameter must be set to True for this release.

do_expensive_check : bool
    If True, performs more extensive tests on the inputs to ensure
    validitity, at the expense of increased run time.

Returns
-------

A 3-tuple, where the first item in the tuple is a device array containing
the vertex identifiers, the second item is a device array containing the
distance for each vertex from the source vertex, and the third item is a
device array containing the vertex identifier of the preceding vertex in the
path for that vertex. For example, the vertex identifier at the ith element
of the vertex array has a distance from the source vertex of the ith element
in the distance array, and the preceding vertex in the path is the ith
element in the predecessor array.

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
    ...     store_transposed=False, renumber=False, do_expensive_check=False)
    >>> (vertices, distances, predecessors) = pylibhipgraph.sssp(
    ...     resource_handle, G, source=1, cutoff=999,
    ...     compute_predecessors=True, do_expensive_check=False)
    >>> vertices
    array([0, 1, 2, 3], dtype=int32)
    >>> distances
    array([3.4028235e+38, 0.0000000e+00, 1.0000000e+00, 2.0000000e+00],
          dtype=float32)
    >>> predecessors
    array([-1, -1,  1,  2], dtype=int32)
