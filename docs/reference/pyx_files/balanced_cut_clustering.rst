.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.balanced_cut_clustering, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-balanced_cut_clustering:

*******************************************
pylibhipgraph.balanced_cut_clustering
*******************************************

**balanced_cut_clustering** (*ResourceHandle resource_handle, _GPUGraph graph, num_clusters, num_eigen_vects, evs_tolerance, evs_max_iter, kmean_tolerance, kmean_max_iter, bool_t do_expensive_check*)

Compute a clustering/partitioning of the given graph using the spectral
balanced cut method.

Parameters
----------

resource_handle : ResourceHandle
    Handle to the underlying device resources needed for referencing data
    and running algorithms.

graph : SGGraph
    The input graph.

num_clusters : size_t
    Specifies the number of clusters to find, must be greater than 1.

num_eigen_vects : size_t
    Specifies the number of eigenvectors to use. Must be lower or equal to
    num_clusters.

evs_tolerance: double
    Specifies the tolerance to use in the eigensolver.

evs_max_iter: size_t
    Specifies the maximum number of iterations for the eigensolver.

kmean_tolerance: double
    Specifies the tolerance to use in the k-means solver.

kmean_max_iter: size_t
    Specifies the maximum number of iterations for the k-means solver.

do_expensive_check : bool_t
    If True, performs more extensive tests on the inputs to ensure
    validitity, at the expense of increased run time.

Returns
-------

A tuple containing the clustering vertices, clusters

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
    >>> (vertices, clusters) = pylibhipgraph.balanced_cut_clustering(
    ...     resource_handle, G, num_clusters=5, num_eigen_vects=2, evs_tolerance=0.00001
    ...     evs_max_iter=100, kmean_tolerance=0.00001, kmean_max_iter=100)
    # FIXME: Fix docstring results.
    >>> vertices
    ############
    >>> clusters
    ############
