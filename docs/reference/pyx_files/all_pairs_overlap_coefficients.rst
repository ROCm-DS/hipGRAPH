.. meta::
  :description: hipGRAPH Python API reference library
  :keywords: hipGRAPH, hipgraph, pylibhipgraph.all_pairs_overlap_coefficients, ROCm-DS, API, documentation

.. _hipgraph-all_pairs_overlap:

********************************************
pylibhipgraph.all_pairs_overlap_coefficients
********************************************

**all_pairs_overlap_coefficients** (*ResourceHandle resource_handle, _GPUGraph graph, vertices, bool_t use_weight, topk, bool_t do_expensive_check*)

Perform All-Pairs Overlap similarity computation.

.. note::
    Overlap similarity must run on a symmetric graph.

Parameters
----------

resource_handle : ResourceHandle
    Handle to the underlying device resources needed for referencing data
    and running algorithms.

graph : SGGraph or MGGraph
    The input graph, for either Single or Multi-GPU operations.

vertices : cudf.Series or None
    Vertex list to compute all-pairs. If None, then compute based
        on all vertices in the graph.

use_weight : bool, optional
    If set to True, then compute the weighted overlap_coefficients (the input graph must be weighted in that case).
    Otherwise, compute the non-weighted overlap_coefficients.

topk : size_t
    Specify the number of answers to return otherwise will return all values.

do_expensive_check : bool
    If True, performs more extensive tests on the inputs to ensure
    validitity, at the expense of increased run time.

Returns
-------

A tuple of device arrays containing the vertex pairs with
their corresponding Overlap coefficient scores.
