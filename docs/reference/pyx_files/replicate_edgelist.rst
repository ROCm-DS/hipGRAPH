.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.replicate_edgelist, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-replicate_edgelist:

*******************************************
pylibhipgraph.replicate_edgelist
*******************************************

**replicate_edgelist** (*ResourceHandle resource_handle, src_array, dst_array, weight_array, edge_id_array, edge_type_id_array*)

Replicate edges across all GPUs

Parameters
----------

resource_handle : ResourceHandle
    Handle to the underlying device resources needed for referencing data
    and running algorithms.

src_array : device array type, optional
    Device array containing the vertex identifiers of the source of each
    directed edge. The order of the array corresponds to the ordering of the
    ``dst_array``, where the ith item in ``src_array`` and the ith item in ``dst_array``
    define the ith edge of the graph.

dst_array : device array type, optional
    Device array containing the vertex identifiers of the destination of
    each directed edge. The order of the array corresponds to the ordering
    of the ``src_array``, where the ith item in ``src_array`` and the ith item in
    ``dst_array`` define the ith edge of the graph.

weight_array : device array type, optional
    Device array containing the weight values of each directed edge. The
    order of the array corresponds to the ordering of the ``src_array`` and
    ``dst_array`` arrays, where the ith item in ``weight_array`` is the weight value
    of the ith edge of the graph.

edge_id_array : device array type, optional
    Device array containing the edge id values of each directed edge. The
    order of the array corresponds to the ordering of the ``src_array`` and
    ``dst_array`` arrays, where the ith item in ``edge_id_array`` is the id value
    of the ith edge of the graph.

edge_type_id_array : device array type, optional
    Device array containing the edge type id values of each directed edge. The
    order of the array corresponds to the ordering of the ``src_array`` and
    ``dst_array`` arrays, where the ith item in ``edge_type_id_array`` is the type id
    value of the ith edge of the graph.

Returns
-------

return cupy arrays of ``src`` and/or ``dst`` and/or ``weight`` and/or ``edge_id``
and/or ``edge_type_id``.
