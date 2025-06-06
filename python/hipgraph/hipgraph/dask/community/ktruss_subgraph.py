# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

from typing import Tuple

import cudf
import cupy as cp
import dask_cudf
import hipgraph.dask.comms.comms as Comms
from dask.distributed import default_client, wait
from pylibhipgraph import ResourceHandle
from pylibhipgraph import k_truss_subgraph as pylibhipgraph_k_truss_subgraph


def _call_k_truss_subgraph(
    sID: bytes,
    mg_graph_x,
    k: int,
    do_expensive_check: bool,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:

    return pylibhipgraph_k_truss_subgraph(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        k=k,
        do_expensive_check=do_expensive_check,
    )


def convert_to_cudf(cp_arrays: cp.ndarray) -> cudf.DataFrame:
    cp_src, cp_dst, cp_weight, _ = cp_arrays

    df = cudf.DataFrame()
    if cp_src is not None:
        df["src"] = cp_src
        df["dst"] = cp_dst
    if cp_weight is not None:
        df["weight"] = cp_weight

    return df


def ktruss_subgraph(input_graph, k: int) -> dask_cudf.DataFrame:
    """
    Returns the K-Truss subgraph of a graph for a specific k.

    The k-truss of a graph is a subgraph where each edge is incident to at
    least (k−2) triangles. K-trusses are used for finding tighlty knit groups
    of vertices in a graph. A k-truss is a relaxation of a k-clique in the graph.
    Finding cliques is computationally demanding and finding the maximal
    k-clique is known to be NP-Hard.

    Parameters
    ----------
    input_graph : hipgraph.Graph
        Graph or matrix object, which should contain the connectivity
        information. Edge weights, if present, should be single or double
        precision floating point values

    k : int
        The desired k to be used for extracting the k-truss subgraph.


    Returns
    -------
    k_truss_edge_lists : dask_cudf.DataFrame
        Distributed GPU data frame containing all source identifiers,
        destination identifiers, and edge weights belonging to the truss.
    """
    if input_graph.is_directed():
        raise ValueError("input graph must be undirected")
    # Initialize dask client
    client = default_client()

    do_expensive_check = False

    result = [
        client.submit(
            _call_k_truss_subgraph,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            k,
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
        ddf = input_graph.unrenumber(ddf, "src")
        ddf = input_graph.unrenumber(ddf, "dst")

    return ddf
