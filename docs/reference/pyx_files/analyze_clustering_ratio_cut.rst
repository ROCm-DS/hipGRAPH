.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.analyze_clustering_ratio_cut, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-analyze_clustering_ratio_cut:

*******************************************
pylibhipgraph.analyze_clustering_ratio_cut
*******************************************

**analyze_clustering_ratio_cut** (*ResourceHandle resource_handle, _GPUGraph graph, size_t num_clusters, vertex, cluster*)

Compute ratio cut score of the specified clustering.

Parameters
----------

resource_handle : ResourceHandle
    Handle to the underlying device resources needed for referencing data
    and running algorithms.

graph : SGGraph
    The input graph.

num_clusters : size_t
    Specifies the number of clusters to find, must be greater than 1.

vertex : device array type
    Vertex ids from the clustering to analyze.

cluster : device array type
    Cluster ids from the clustering to analyze.

Returns
-------

The ratio cut score of the specified clustering.

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
    >>> (vertex, cluster) = pylibhipgraph.spectral_modularity_maximization(
    ...     resource_handle, G, num_clusters=5, num_eigen_vects=2, evs_tolerance=0.00001
    ...     evs_max_iter=100, kmean_tolerance=0.00001, kmean_max_iter=100)
        # FIXME: Fix docstring result.
    >>> vertices
    ############
    >>> clusters
    ############
    >>> score = pylibhipgraph.analyze_clustering_ratio_cut(
    ...     resource_handle, G, num_clusters=5, vertex=vertex, cluster=cluster)
    >>> score
    ############
