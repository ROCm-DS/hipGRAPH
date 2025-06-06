# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import cudf
import dask_cudf
import hipgraph.dask.comms.comms as Comms
from dask.distributed import default_client, wait
from pylibhipgraph import ResourceHandle
from pylibhipgraph import core_number as pylibhipgraph_core_number


def convert_to_cudf(cp_arrays):
    """
    Creates a cudf DataFrame from cupy arrays from pylibhipgraph wrapper
    """
    cupy_vertices, cupy_core_number = cp_arrays
    df = cudf.DataFrame()
    df["vertex"] = cupy_vertices
    df["core_number"] = cupy_core_number

    return df


def _call_plc_core_number(sID, mg_graph_x, dt_x, do_expensive_check):
    return pylibhipgraph_core_number(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        degree_type=dt_x,
        do_expensive_check=do_expensive_check,
    )


def core_number(input_graph, degree_type="bidirectional"):
    """
    Compute the core numbers for the nodes of the graph G. A k-core of a graph
    is a maximal subgraph that contains nodes of degree k or more.
    A node has a core number of k if it belongs a k-core but not to k+1-core.
    This call does not support a graph with self-loops and parallel
    edges.

    Parameters
    ----------
    input_graph : hipgraph.graph
        The current implementation only supports undirected graphs.  The graph
        can contain edge weights, but they don't participate in the calculation
        of the core numbers.

    degree_type: str, (default="bidirectional")
        This option is currently ignored.  This option may eventually determine
        if the core number computation should be based on input, output, or
        both directed edges, with valid values being "incoming", "outgoing",
        and "bidirectional" respectively.

    Returns
    -------
    result : dask_cudf.DataFrame
        GPU distributed data frame containing 2 dask_cudf.Series

        ddf['vertex']: dask_cudf.Series
            Contains the core number vertices
        ddf['core_number']: dask_cudf.Series
            Contains the core number of vertices
    """

    if input_graph.is_directed():
        raise ValueError("input graph must be undirected")

    # degree_type is currently ignored until libhipgraph supports directed
    # graphs for core_number. Once supporteed, degree_type should be checked
    # like so:
    # if degree_type not in ["incoming", "outgoing", "bidirectional"]:
    #     raise ValueError(
    #         f"'degree_type' must be either incoming, "
    #         f"outgoing or bidirectional, got: {degree_type}"
    #     )

    # Initialize dask client
    client = default_client()

    do_expensive_check = False

    result = [
        client.submit(
            _call_plc_core_number,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            degree_type,
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
        ddf = input_graph.unrenumber(ddf, "vertex")

    return ddf
