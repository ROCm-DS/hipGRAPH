.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.leiden, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-leiden:

*******************************************
pylibhipgraph.leiden
*******************************************

**leiden** (*ResourceHandle resource_handle, random_state, _GPUGraph graph, size_t max_level, double resolution, double theta, bool_t do_expensive_check*)

Compute the modularity optimizing partition of the input graph using the
Leiden method.

Parameters
----------

resource_handle : ResourceHandle
    Handle to the underlying device resources needed for referencing data
    and running algorithms.

random_state : int , optional
    Random state to use when generating samples. Optional argument,
    defaults to a hash of process id, time, and hostname.
    (See pylibhipgraph.random.HipGraphRandomState)

graph : SGGraph or MGGraph
    The input graph.

max_level: size_t
    This controls the maximum number of levels/iterations of the leiden
    algorithm. When specified the algorithm will terminate after no more
    than the specified number of iterations. No error occurs when the
    algorithm terminates early in this manner.

resolution: double
    Called gamma in the modularity formula, this changes the size
    of the communities.  Higher resolutions lead to more smaller
    communities, lower resolutions lead to fewer larger communities.
    Defaults to 1.

theta: double
    Called theta in the Leiden algorithm, this is used to scale
    modularity gain in Leiden refinement phase, to compute
    the probability of joining a random leiden community.

do_expensive_check : bool_t
    If True, performs more extensive tests on the inputs to ensure
    validitity, at the expense of increased run time.

Returns
-------

A tuple containing the hierarchical clustering vertices, clusters and
modularity score

Examples
--------

.. code:: Python

    >>> import pylibhipgraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 2], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 2, 0], dtype=numpy.int32)
    >>> weights = cupy.asarray([1.0, 1.0, 1.0], dtype=numpy.float32)
    >>> resource_handle = pylibhipgraph.ResourceHandle()
    >>> graph_props = pylibhipgraph.GraphProperties(
    ...     is_symmetric=True, is_multigraph=False)
    >>> G = pylibhipgraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=True, renumber=False, do_expensive_check=False)
    >>> (vertices, clusters, modularity) = pylibhipgraph.Leiden(
                                resource_handle, G, 100, 1., False)
    >>> vertices
    [0, 1, 2]
    >>> clusters
    [0, 0, 0]
    >>> modularity
    0.0
