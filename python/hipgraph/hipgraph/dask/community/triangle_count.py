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
from hipgraph.dask.common.part_utils import (
    get_persisted_df_worker_map,
    persist_dask_df_equal_parts_per_worker,
)
from pylibhipgraph import ResourceHandle
from pylibhipgraph import triangle_count as pylibhipgraph_triangle_count


def _call_triangle_count(
    sID,
    mg_graph_x,
    start_list,
    do_expensive_check,
):
    return pylibhipgraph_triangle_count(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        start_list=start_list,
        do_expensive_check=do_expensive_check,
    )


def convert_to_cudf(cp_arrays):
    """
    Creates a cudf DataFrame from cupy arrays from pylibhipgraph wrapper
    """
    cupy_vertices, cupy_counts = cp_arrays
    df = cudf.DataFrame()
    df["vertex"] = cupy_vertices
    df["counts"] = cupy_counts

    return df


def triangle_count(input_graph, start_list=None):
    """
    Computes the number of triangles (cycles of length three) and the number
    per vertex in the input graph.

    Parameters
    ----------
    input_graph : hipgraph.graph
        hipGRAPH graph descriptor, should contain the connectivity information,
        (edge weights are not used in this algorithm).
        The current implementation only supports undirected graphs.

    start_list : list or cudf.Series
        list of vertices for triangle count. if None the entire set of vertices
        in the graph is processed


    Returns
    -------
    result : dask_cudf.DataFrame
        GPU distributed data frame containing 2 dask_cudf.Series

    ddf['vertex']: dask_cudf.Series
            Contains the triangle counting vertices
    ddf['counts']: dask_cudf.Series
        Contains the triangle counting counts
    """
    if input_graph.is_directed():
        raise ValueError("input graph must be undirected")
    # Initialize dask client.
    client = default_client()

    if start_list is not None:
        if isinstance(start_list, int):
            start_list = [start_list]
        if isinstance(start_list, list):
            start_list = cudf.Series(start_list)
        if not isinstance(start_list, cudf.Series):
            raise TypeError(
                f"'start_list' must be either a list or a cudf.Series,"
                f"got: {start_list.dtype}"
            )

        # start_list uses "external" vertex IDs, but since the graph has been
        # renumbered, the start vertex IDs must also be renumbered.
        if input_graph.renumbered:
            start_list = input_graph.lookup_internal_vertex_id(start_list).compute()

        # Ensure correct dtype.
        start_list.astype(
            input_graph.edgelist.edgelist_df[
                input_graph.renumber_map.renumbered_src_col_name
            ].dtype
        )

        n_workers = get_n_workers()
        start_list = dask_cudf.from_cudf(start_list, npartitions=get_n_workers())

        start_list = start_list.repartition(npartitions=n_workers)
        start_list = persist_dask_df_equal_parts_per_worker(start_list, client)
        start_list = get_persisted_df_worker_map(start_list, client)
    do_expensive_check = False

    result = [
        client.submit(
            _call_triangle_count,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            start_list[w][0] if start_list is not None else None,
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
