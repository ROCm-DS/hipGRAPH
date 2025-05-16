.. meta::
  :description: ROCm-DS pylibhipgraph API reference library
  :keywords: hipGRAPH, pylibhipgraph, pylibhipgraph.uniform_neighbor_sample, rocGRAPH, ROCm-DS, API, documentation

.. _pylibhipgraph-uniform_neighbor_sample:

*******************************************
pylibhipgraph.uniform_neighbor_sample
*******************************************

**uniform_neighbor_sample** (ResourceHandle resource_handle, _GPUGraph input_graph, start_list, h_fan_out, \*, bool_t with_replacement, bool_t do_expensive_check, with_edge_properties=False, batch_id_list=None, label_list=None, label_to_output_comm_rank=None, label_offsets=None, prior_sources_behavior=None, deduplicate_sources=False, return_hops=False, renumber=False, retain_seeds=False, compression='COO', compress_per_hop=False, random_state=None, return_dict=False)

Does neighborhood sampling, which samples nodes from a graph based on the
current node's neighbors, with a corresponding fanout value at each hop.

Parameters
----------

resource_handle: ResourceHandle
    Handle to the underlying device and host resources needed for
    referencing data and running algorithms.

input_graph : SGGraph or MGGraph
    The input graph, for either Single or Multi-GPU operations.

start_list: device array type
    Device array containing the list of starting vertices for sampling.

h_fan_out: numpy array type
    Device array containing the brancing out (fan-out) degrees per
    starting vertex for each hop level.

with_replacement: bool
    If true, sampling procedure is done with replacement (the same vertex
    can be selected multiple times in the same step).

do_expensive_check: bool
    If True, performs more extensive tests on the inputs to ensure
    validitity, at the expense of increased run time.

with_edge_properties: bool
    If True, returns the edge properties of each edges along with the
    edges themselves.  Will result in an error if the provided graph
    does not have edge properties.

batch_id_list: list[int32] (Optional)
    List of int32 batch ids that is returned with each edge.  Optional
    argument, defaults to NULL, returning nothing.

label_list: list[int32] (Optional)
    List of unique int32 batch ids.  Required if also passing the
    label_to_output_comm_rank flag.  Default to NULL (does nothing)

label_to_output_comm_rank: list[int32] (Optional)
    Maps the unique batch ids in label_list to the rank of the
    worker that should hold results for that batch id.
    Defaults to NULL (does nothing)

label_offsets: list[int] (Optional)
    Offsets of each label within the start vertex list.

prior_sources_behavior: str (Optional)
    Options are "carryover", and "exclude".
    Default will leave the source list as-is.
    Carryover will carry over sources from previous hops to the
    current hop.
    Exclude will exclude sources from previous hops from reappearing
    as sources in future hops.

deduplicate_sources: bool (Optional)
    If True, will deduplicate the source list before sampling.
    Defaults to False.

renumber: bool (Optional)
    If True, will renumber the sources and destinations on a
    per-batch basis and return the renumber map and batch offsets
    in additional to the standard returns.

retain_seeds: bool (Optional)
    If True, will retain the original seeds (original source vertices)
    in the output even if they do not have outgoing neighbors.
    Defaults to False.

compression: str (Optional)
    Options: COO (default), CSR, CSC, DCSR, DCSR
    Sets the compression format for the returned samples.

compress_per_hop: bool (Optional)
    If False (default), will create a compressed edgelist for the
    entire batch.
    If True, will create a separate compressed edgelist per hop within
    a batch.

random_state: int (Optional)
    Random state to use when generating samples.  Optional argument,
    defaults to a hash of process id, time, and hostname.
    (See pylibhipgraph.random.HipGraphRandomState)

return_dict: bool (Optional)
    Whether to return a dictionary instead of a tuple.
    Optional argument, defaults to False, returning a tuple.
    This argument will eventually be deprecated in favor
    of always returning a dictionary.

Returns
-------

A tuple of device arrays, where the first and second items in the tuple
are device arrays containing the starting and ending vertices of each
walk respectively, the third item in the tuple is a device array
containing the start labels, and the fourth item in the tuple is a device
array containing the indices for reconstructing paths.

If renumber was set to True, then the fifth item in the tuple is a device
array containing the renumber map, and the sixth item in the tuple is a
device array containing the renumber map offsets (which delineate where
the renumber map for each batch starts).
