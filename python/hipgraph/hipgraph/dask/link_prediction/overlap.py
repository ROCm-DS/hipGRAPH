# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import cudf
import dask_cudf
import hipgraph.dask.comms.comms as Comms
from dask.distributed import default_client, wait
from hipgraph.dask import get_n_workers
from hipgraph.dask.common.input_utils import get_distributed_data
from hipgraph.dask.common.part_utils import (
    get_persisted_df_worker_map,
    persist_dask_df_equal_parts_per_worker,
)
from hipgraph.utilities import renumber_vertex_pair
from pylibhipgraph import ResourceHandle
from pylibhipgraph import (
    all_pairs_overlap_coefficients as pylibhipgraph_all_pairs_overlap_coefficients,
)
from pylibhipgraph import overlap_coefficients as pylibhipgraph_overlap_coefficients


def convert_to_cudf(cp_arrays):
    """
    Creates a cudf DataFrame from cupy arrays from pylibhipgraph wrapper
    """

    cupy_first, cupy_second, cupy_similarity = cp_arrays

    df = cudf.DataFrame()
    df["first"] = cupy_first
    df["second"] = cupy_second
    df["overlap_coeff"] = cupy_similarity

    return df


def _call_plc_all_pairs_overlap(
    sID, mg_graph_x, vertices, use_weight, topk, do_expensive_check
):

    return pylibhipgraph_all_pairs_overlap_coefficients(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        vertices=vertices,
        use_weight=use_weight,
        topk=topk,
        do_expensive_check=do_expensive_check,
    )


def _call_plc_overlap(
    sID, mg_graph_x, vertex_pair, use_weight, do_expensive_check, vertex_pair_col_name
):

    first = vertex_pair[vertex_pair_col_name[0]]
    second = vertex_pair[vertex_pair_col_name[1]]

    return pylibhipgraph_overlap_coefficients(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        first=first,
        second=second,
        use_weight=use_weight,
        do_expensive_check=do_expensive_check,
    )


def overlap(input_graph, vertex_pair=None, use_weight=False):
    """
    Compute the Overlap Coefficient between each pair of vertices connected by
    an edge, or between arbitrary pairs of vertices specified by the user.
    Overlap Coefficient is defined between two sets as the ratio of the volume
    of their intersection over the smaller of their two volumes. In the
    context of graphs, the neighborhood of a vertex is seen as a set. The
    Overlap Coefficient weight of each edge represents the strength of
    connection between vertices based on the relative similarity of their
    neighbors. If first is specified but second is not, or vice versa, an
    exception will be thrown.

    hipgraph.overlap, in the absence of a specified vertex pair list, will
    compute the two_hop_neighbors of the entire graph to construct a vertex pair
    list and will return the Overlap coefficient for those vertex pairs. This is
    not advisable as the vertex_pairs can grow exponentially with respect to the
    size of the datasets


    Parameters
    ----------
    input_graph : hipgraph.Graph
        hipGRAPH Graph instance, should contain the connectivity information
        as an edge list (edge weights are not supported yet for this algorithm). The
        graph should be undirected where an undirected edge is represented by a
        directed edge in both direction. The adjacency list will be computed if
        not already present.

        This implementation only supports undirected, non-multi Graphs.

    vertex_pair : cudf.DataFrame, optional (default=None)
        A GPU dataframe consisting of two columns representing pairs of
        vertices. If provided, the Overlap coefficient is computed for the
        given vertex pairs.  If the vertex_pair is not provided then the
        current implementation computes the Overlap coefficient for all
        adjacent vertices in the graph.

    use_weight : bool, optional (default=False)
        Flag to indicate whether to compute weighted overlap (if use_weight==True)
        or un-weighted overlap (if use_weight==False).
        'input_graph' must be weighted if 'use_weight=True'.

    Returns
    -------
    result : dask_cudf.DataFrame
        GPU distributed data frame containing 3 dask_cudf.Series

        ddf['first']: dask_cudf.Series
            The first vertex ID of each pair(will be identical to first if specified).
        ddf['second']: dask_cudf.Series
            The second vertex ID of each pair(will be identical to second if
            specified).
        ddf['overlap_coeff']: dask_cudf.Series
            The computed overlap coefficient between the first and the second
            vertex ID.
    """

    if input_graph.is_directed():
        raise ValueError("input graph must be undirected")

    if vertex_pair is None:
        # Call two_hop neighbor of the entire graph
        vertex_pair = input_graph.get_two_hop_neighbors()

    vertex_pair_col_name = vertex_pair.columns

    if isinstance(vertex_pair, (dask_cudf.DataFrame, cudf.DataFrame)):
        vertex_pair = renumber_vertex_pair(input_graph, vertex_pair)

    elif vertex_pair is not None:
        raise ValueError("vertex_pair must be a dask_cudf or cudf dataframe")

    if not isinstance(vertex_pair, (dask_cudf.DataFrame)):
        vertex_pair = dask_cudf.from_cudf(
            vertex_pair, npartitions=len(Comms.get_workers())
        )
    vertex_pair = get_distributed_data(vertex_pair)
    wait(vertex_pair)
    vertex_pair = vertex_pair.worker_to_parts

    # Initialize dask client
    client = default_client()

    do_expensive_check = False

    result = [
        client.submit(
            _call_plc_overlap,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            vertex_pair[w][0],
            use_weight,
            do_expensive_check,
            vertex_pair_col_name,
            workers=[w],
            allow_other_workers=False,
        )
        for w in Comms.get_workers()
    ]

    wait(result)

    cudf_result = [client.submit(convert_to_cudf, cp_arrays) for cp_arrays in result]

    wait(cudf_result)

    ddf = dask_cudf.from_delayed(cudf_result).persist()
    wait(ddf)

    # Wait until the inactive futures are released
    wait([(r.release(), c_r.release()) for r, c_r in zip(result, cudf_result)])

    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "first")
        ddf = input_graph.unrenumber(ddf, "second")

    return ddf


def all_pairs_overlap(
    input_graph,
    vertices: cudf.Series = None,
    use_weight: bool = False,
    topk: int = None,
):
    """
    Compute the All Pairs Overlap similarity between all pairs of vertices specified.
    All pairs Overlap Coefficient is defined between two sets as the ratio of the volume
    of their intersection over the smaller of their two volumes. In the context
    of graphs, the neighborhood of a vertex is seen as a set. The Overlap
    similarity weight of each edge represents the strength of connection
    between vertices based on the relative similarity of their neighbors.

    hipgraph.all_pairs_overlap, in the absence of specified vertices, will
    compute the two_hop_neighbors of the entire graph to construct a vertex pair
    list and will return the overlap coefficient for all the vertex pairs in the graph.
    This is not advisable as the vertex_pairs can grow exponentially with respect to
    the size of the datasets.

    If the topk parameter is specified then the result will only contain the top k
    highest scoring results.

    Parameters
    ----------
    input_graph : hipgraph.Graph
        hipGRAPH Graph instance, should contain the connectivity information
        as an edge list (edge weights are not supported yet for this algorithm). The
        graph should be undirected where an undirected edge is represented by a
        directed edge in both direction. The adjacency list will be computed if
        not already present.

       This implementation only supports undirected, non-multi Graphs.

    vertices : int or list or cudf.Series, dask_cudf.Series, optional (default=None)
        A GPU Series containing the input vertex list.  If the vertex list is not
        provided then the current implementation computes the overlap coefficient for
        all adjacent vertices in the graph.

    use_weight : bool, optional (default=False)
        Flag to indicate whether to compute weighted overlap (if use_weight==True)
        or un-weighted overlap (if use_weight==False).
        'input_graph' must be weighted if 'use_weight=True'.

    topk : int, optional (default=None)
        Specify the number of answers to return otherwise returns the entire
        solution

    Returns
    -------
    result : dask_cudf.DataFrame
        GPU distributed data frame containing 3 dask_cudf.Series

        ddf['first']: dask_cudf.Series
            The first vertex ID of each pair (will be identical to first if specified).
        ddf['second']: dask_cudf.Series
            The second vertex ID of each pair (will be identical to second if
            specified).
        ddf['overlap_coeff']: dask_cudf.Series
            The computed overlap coefficient between the first and the second
            vertex ID.
    """

    if input_graph.is_directed():
        raise ValueError("input graph must be undirected")

    # Initialize dask client
    client = default_client()

    if vertices is not None:
        if isinstance(vertices, int):
            vertices = [vertices]

        if isinstance(vertices, list):
            vertices = cudf.Series(
                vertices,
                dtype=input_graph.edgelist.edgelist_df[
                    input_graph.renumber_map.renumbered_src_col_name
                ].dtype,
            )

        if not isinstance(vertices, (dask_cudf.Series)):
            vertices = dask_cudf.from_cudf(vertices, npartitions=get_n_workers())

        if input_graph.renumbered:
            vertices = input_graph.lookup_internal_vertex_id(vertices)

        n_workers = get_n_workers()
        vertices = vertices.repartition(npartitions=n_workers)
        vertices = persist_dask_df_equal_parts_per_worker(vertices, client)
        vertices = get_persisted_df_worker_map(vertices, client)

    do_expensive_check = False

    result = [
        client.submit(
            _call_plc_all_pairs_overlap,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            vertices[w][0] if vertices is not None else None,
            use_weight,
            topk,
            do_expensive_check,
            workers=[w],
            allow_other_workers=False,
        )
        for w in Comms.get_workers()
    ]

    wait(result)

    cudf_result = [client.submit(convert_to_cudf, cp_arrays) for cp_arrays in result]

    wait(cudf_result)

    ddf = dask_cudf.from_delayed(cudf_result).persist()
    wait(ddf)

    # Wait until the inactive futures are released
    wait([(r.release(), c_r.release()) for r, c_r in zip(result, cudf_result)])

    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "first")
        ddf = input_graph.unrenumber(ddf, "second")

    return ddf
