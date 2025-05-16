.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.sorensen_coefficients, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-sorensen_coefficients:

*******************************************
pylibhipgraph.sorensen_coefficients
*******************************************

**sorensen_coefficients** (*ResourceHandle resource_handle, _GPUGraph graph, first, second, bool_t use_weight, bool_t do_expensive_check*)

Compute the Sorensen coefficients for the specified vertex_pairs.

.. note::
    Sorensen similarity must run on a symmetric graph.

Parameters
----------

resource_handle : ResourceHandle
    Handle to the underlying device resources needed for referencing data
    and running algorithms.

graph : SGGraph or MGGraph
    The input graph, for either Single or Multi-GPU operations.

first :
    Source of the vertex pair.

second :
    Destination of the vertex pair.

use_weight : bool, optional
    If set to True, compute the weighted jaccard_coefficients (the input graph must be weighted in that case).
    Otherwise, compute the un-weighted jaccard_coefficients.

do_expensive_check : bool
    If True, performs more extensive tests on the inputs to ensure
    validitity, at the expense of increased run time.

Returns
-------

A tuple of device arrays containing the vertex pairs with
their corresponding Sorensen coefficient scores.
