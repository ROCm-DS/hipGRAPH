# Copyright (c) 2019-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import cudf
import cupy
import dask_cudf
import hipgraph.dask.comms.comms as Comms
from dask.distributed import default_client, wait
from pylibhipgraph import ResourceHandle
from pylibhipgraph import sssp as pylibhipgraph_sssp


def _call_plc_sssp(
    sID, mg_graph_x, source, cutoff, compute_predecessors, do_expensive_check
):
    vertices, distances, predecessors = pylibhipgraph_sssp(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        source=source,
        cutoff=cutoff,
        compute_predecessors=compute_predecessors,
        do_expensive_check=do_expensive_check,
    )
    return cudf.DataFrame(
        {
            "distance": cudf.Series(distances),
            "vertex": cudf.Series(vertices),
            "predecessor": cudf.Series(predecessors),
        }
    )


def sssp(input_graph, source, cutoff=None, check_source=True):
    """
    Compute the distance and predecessors for shortest paths from the specified
    source to all the vertices in the input_graph. The distances column will
    store the distance from the source to each vertex. The predecessors column
    will store each vertex's predecessor in the shortest path. Vertices that
    are unreachable will have a distance of infinity denoted by the maximum
    value of the data type and the predecessor set as -1. The source vertex's
    predecessor is also set to -1.  The input graph must contain edge list as
    dask-cudf dataframe with one partition per GPU.

    Parameters
    ----------
    input_graph : hipgraph.Graph
        hipGRAPH graph descriptor, should contain the connectivity information
        as dask cudf edge list dataframe.

    source : Integer
        Specify source vertex

    cutoff : double, optional (default = None)
        Maximum edge weight sum considered by the algorithm

    check_source : bool, optional (default=True)
        If True, performs more extensive tests on the start vertices
        to ensure validitity, at the expense of increased run time.

    Returns
    -------
    df : dask_cudf.DataFrame
        df['vertex'] gives the vertex id

        df['distance'] gives the path distance from the
        starting vertex

        df['predecessor'] gives the vertex id it was
        reached from in the traversal

    Examples
    --------
    >>> import hipgraph.dask as dcg
    >>> import dask_cudf
    >>> # ... Init a DASK Cluster
    >>> #    see https://docs.rapids.ai/api/hipgraph/stable/dask-hipgraph.html
    >>> # Download dataset from https://github.com/rapidsai/hipgraph/datasets/..
    >>> chunksize = dcg.get_chunksize(datasets_path / "karate.csv")
    >>> ddf = dask_cudf.read_csv(datasets_path / "karate.csv",
    ...                          blocksize=chunksize, delimiter=" ",
    ...                          names=["src", "dst", "value"],
    ...                          dtype=["int32", "int32", "float32"])
    >>> dg = hipgraph.Graph(directed=True)
    >>> dg.from_dask_cudf_edgelist(ddf, source='src', destination='dst',
    ...                            edge_attr='value')
    >>> df = dcg.sssp(dg, 0)

    """

    # FIXME: Implement a better way to check if the graph is weighted similar
    # to 'simpleGraph'
    if not input_graph.weighted:
        err_msg = (
            "'SSSP' requires the input graph to be weighted."
            "'BFS' should be used instead of 'SSSP' for unweighted graphs."
        )
        raise ValueError(err_msg)

    client = default_client()

    def check_valid_vertex(G, source):
        is_valid_vertex = G.has_node(source)
        if not is_valid_vertex:
            raise ValueError("Invalid source vertex")

    if check_source:
        check_valid_vertex(input_graph, source)

    if cutoff is None:
        cutoff = cupy.inf

    if input_graph.renumbered:
        source = (
            input_graph.lookup_internal_vertex_id(cudf.Series([source]))
            .fillna(-1)
            .compute()
        )
        source = source.iloc[0]

    do_expensive_check = False
    compute_predecessors = True
    result = [
        client.submit(
            _call_plc_sssp,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            source,
            cutoff,
            compute_predecessors,
            do_expensive_check,
            workers=[w],
            allow_other_workers=False,
        )
        for w in Comms.get_workers()
    ]

    wait(result)
    ddf = dask_cudf.from_delayed(result).persist()
    wait(ddf)

    # Wait until the inactive futures are released
    wait([r.release() for r in result])

    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "vertex")
        ddf = input_graph.unrenumber(ddf, "predecessor")
        ddf["predecessor"] = ddf["predecessor"].fillna(-1)

    return ddf
