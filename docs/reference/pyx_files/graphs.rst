.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.SGGraph, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-SGGraph:

*******************************************
pylibhipgraph.SGGraph
*******************************************

**class SGGraph** (*_GPUGraph*)

RAII-stye Graph class for use with single-GPU APIs that manages the
individual create/free calls and the corresponding hipgraph_graph_t pointer.

Parameters
----------

resource_handle : ResourceHandle
    Handle to the underlying device resources needed for referencing data
    and running algorithms.

graph_properties : GraphProperties
    Object defining intended properties for the graph.

src_or_offset_array : device array type
    Device array containing either the vertex identifiers of the source of
    each directed edge if represented in COO format or the offset if
    CSR format. In the case of a COO, the order of the array corresponds to
    the ordering of the dst_or_index_array, where the ith item in
    src_offset_array and the ith item in dst_index_array define the ith edge
    of the graph.

dst_or_index_array : device array type
    Device array containing the vertex identifiers of the destination of
    each directed edge if represented in COO format or the index if
    CSR format. In the case of a COO, The order of the array corresponds to
    the ordering of the src_offset_array, where the ith item in src_offset_array
    and the ith item in dst_index_array define the ith edge of the graph.

vertices_array : device array type
    Device array containing all vertices of the graph. This array is
    optional, but must be used if the graph contains isolated vertices
    which cannot be represented in the src_or_offset_array and
    dst_index_array arrays.  If specified, this array must contain every
    vertex identifier, including vertex identifiers that are already
    included in the src_or_offset_array and dst_index_array arrays.

weight_array : device array type
    Device array containing the weight values of each directed edge. The
    order of the array corresponds to the ordering of the src_array and
    dst_array arrays, where the ith item in weight_array is the weight value
    of the ith edge of the graph.

store_transposed : bool
    Set to True if the graph should be transposed. This is required for some
    algorithms, such as pagerank.

renumber : bool
    Set to True to indicate the vertices used in src_array and dst_array are
    not appropriate for use as internal array indices, and should be mapped
    to continuous integers starting from 0.

do_expensive_check : bool
    If True, performs more extensive tests on the inputs to ensure
    validity, at the expense of increased run time.

edge_id_array : device array type
    Device array containing the edge ids of each directed edge.  Must match
    the ordering of the src/dst arrays.  Optional (may be null).  If
    provided, edge_type_array must also be provided.

edge_type_array : device array type
    Device array containing the edge types of each directed edge.  Must
    match the ordering of the src/dst/edge_id arrays.  Optional (may be
    null).  If provided, edge_id_array must be provided.

input_array_format: str, optional (default='COO')
    Input representation used to construct a graph
        COO: arrays represent src_array and dst_array
        CSR: arrays represent offset_array and index_array

drop_self_loops : bool, optional (default='False')
    If true, drop any self loops that exist in the provided edge list.

drop_multi_edges: bool, optional (default='False')
    If true, drop any multi edges that exist in the provided edge list

Examples
---------

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
