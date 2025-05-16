.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.generate_rmat_edgelist, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-generate_rmat_edgelist:

*******************************************
pylibhipgraph.generate_rmat_edgelist
*******************************************

**generate_rmat_edgelist** (*ResourceHandle resource_handle, random_state, size_t scale, size_t num_edges, double a, double b, double c, bool_t clip_and_flip, bool_t scramble_vertex_ids, bool_t include_edge_weights, minimum_weight, maximum_weight dtype, bool_t include_edge_ids, bool_t include_edge_types, min_edge_type_value, max_edge_type_value, bool_t multi_gpu*)

Generate RMAT edge list

Parameters
----------

resource_handle : ResourceHandle
    Handle to the underlying device resources needed for referencing data
    and running algorithms.

random_state : int , optional
    Random state to use when generating samples. Optional argument,
    defaults to a hash of process id, time, and hostname.
    (See pylibhipgraph.random.HipGraphRandomState)

scale : size_t
    Scale factor to set the number of vertices in the graph Vertex IDs have
    values in [0, V), where V = 1 << ``scale``

num_edges : size_t
    Number of edges to generate

a : double
    Probability of the edge being in the first partition
    The Graph 500 spec sets this value to 0.57

b : double
    Probability of the edge being in the second partition
    The Graph 500 spec sets this value to 0.19

c : double
    Probability of the edge being in the third partition
    The Graph 500 spec sets this value to 0.19

clip_and_flip : bool
    Flag controlling whether to generate edges only in the lower triangular
    part (including the diagonal) of the graph adjacency matrix
    (if set to ``true``) or not (if set to ``false``).

scramble_vertex_ids : bool
    Flag controlling whether to scramble vertex ID bits (if set to ``true``)
    or not (if set to ``false``); scrambling vertex ID bits breaks
    correlation between vertex ID values and vertex degrees.

include_edge_weights : bool
    Flag controlling whether to generate edges with weights
    (if set to ``true``) or not (if set to ``false``).

minimum_weight : double
    Minimum weight value to generate (if ``include_edge_weights`` is ``true``)

maximum_weight : double
    Maximum weight value to generate (if ``include_edge_weights`` is ``true``)

dtype : string
    The type of weight to generate (``FLOAT32`` or ``FLOAT64``), ignored unless
    ``include_weights`` is ``true``

include_edge_ids : bool
    Flag controlling whether to generate edges with ids
    (if set to ``true``) or not (if set to ``false``).

include_edge_types : bool
    Flag controlling whether to generate edges with types
    (if set to ``true``) or not (if set to ``false``).

min_edge_type_value : int
    Minimum edge type to generate if ``include_edge_types`` is ``true``
    otherwise, this parameter is ignored.

max_edge_type_value : int
    Maximum edge type to generate if ``include_edge_types`` is ``true``
    otherwise, this parameter is ignored.

multi_gpu : bool
    Flag if the COO is being created on multiple GPUs

Returns
-------

Return a tuple containing the sources and destinations with their corresponding
weights, ids and types if the flags ``include_edge_weights``, ``include_edge_ids``
and ``include_edge_types`` are respectively set to ``true``
