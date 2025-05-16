.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.ecg, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-ecg:

*******************************************
pylibhipgraph.ecg
*******************************************

**ecg** (*ResourceHandle resource_handle, random_state, _GPUGraph graph, double min_weight, size_t ensemble_size, size_t max_level, double threshold, double resolution, bool_t do_expensive_check*)

Compute the Ensemble Clustering for Graphs (ECG) partition of the input
graph. ECG runs truncated Louvain on an ensemble of permutations of the
input graph, then uses the ensemble partitions to determine weights for
the input graph. The final result is found by running full Louvain on
the input graph using the determined weights.

See https://arxiv.org/abs/1809.05578 for further information.

Parameters
----------

resource_handle : ResourceHandle
    Handle to the underlying device resources needed for referencing data
    and running algorithms.

random_state : int , optional
    Random state to use when generating samples. Optional argument,
    defaults to a hash of process id, time, and hostname.
    (See pylibhipgraph.random.HipGraphRandomState)

graph : SGGraph
    The input graph.

min_weight : double, optional (default=0.5)
    The minimum value to assign as an edgeweight in the ECG algorithm.
    It should be a value in the range [0,1] usually left as the default
    value of .05

ensemble_size : size_t, optional (default=16)
    The number of graph permutations to use for the ensemble.
    The default value is 16, larger values may produce higher quality
    partitions for some graphs.

max_level: size_t
    This controls the maximum number of levels/iterations of the leiden
    algorithm. When specified the algorithm will terminate after no more
    than the specified number of iterations. No error occurs when the
    algorithm terminates early in this manner.

threshold: float
    Modularity gain threshold for each level. If the gain of
    modularity between 2 levels of the algorithm is less than the
    given threshold then the algorithm stops and returns the
    resulting communities.

resolution: double
    Called gamma in the modularity formula, this changes the size
    of the communities.  Higher resolutions lead to more smaller
    communities, lower resolutions lead to fewer larger communities.
    Defaults to 1.

do_expensive_check : bool_t
    If True, performs more extensive tests on the inputs to ensure
    validitity, at the expense of increased run time.

Returns
-------

A tuple containing the hierarchical clustering vertices, clusters

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
    >>> (vertices, clusters) = pylibhipgraph.ecg(resource_handle, G)
    # FIXME: Check this docstring example
    >>> vertices
    [0, 1, 2]
    >>> clusters
    [0, 0, 0]
