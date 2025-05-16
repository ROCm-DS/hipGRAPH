# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import warnings

import cudf
import dask_cudf
import hipgraph.dask.comms.comms as Comms
from dask.distributed import default_client, wait
from pylibhipgraph import ResourceHandle
from pylibhipgraph import hits as pylibhipgraph_hits


def _call_plc_hits(
    sID,
    mg_graph_x,
    tol,
    max_iter,
    initial_hubs_guess_vertices,
    initial_hubs_guess_values,
    normalized,
    do_expensive_check,
):

    return pylibhipgraph_hits(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        tol=tol,
        max_iter=max_iter,
        initial_hubs_guess_vertices=initial_hubs_guess_vertices,
        initial_hubs_guess_values=initial_hubs_guess_values,
        normalized=normalized,
        do_expensive_check=do_expensive_check,
    )


def convert_to_cudf(cp_arrays):
    """
    create a cudf DataFrame from cupy arrays
    """
    cupy_vertices, cupy_hubs, cupy_authorities = cp_arrays
    df = cudf.DataFrame()
    df["vertex"] = cupy_vertices
    df["hubs"] = cupy_hubs
    df["authorities"] = cupy_authorities
    return df


def hits(input_graph, tol=1.0e-5, max_iter=100, nstart=None, normalized=True):
    """
    Compute HITS hubs and authorities values for each vertex

    The HITS algorithm computes two numbers for a node.  Authorities
    estimates the node value based on the incoming links.  Hubs estimates
    the node value based on outgoing links.

    Both hipGRAPH and networkx implementation use a 1-norm.

    Parameters
    ----------

    input_graph : hipgraph.Graph
        hipGRAPH graph descriptor, should contain the connectivity information
        as an edge list (edge weights are not used for this algorithm).
        The adjacency list will be computed if not already present.

    tol : float, optional (default=1.0e-5)
        Set the tolerance of the approximation, this parameter should be a
        small magnitude value.

    max_iter : int, optional (default=100)
        The maximum number of iterations before an answer is returned.

    nstart : cudf.Dataframe, optional (default=None)
        The initial hubs guess vertices along with their initial hubs guess
        value

        nstart['vertex'] : cudf.Series
            Initial hubs guess vertices
        nstart['values'] : cudf.Series
            Initial hubs guess values

    normalized : bool, optional (default=True)
        A flag to normalize the results

    Returns
    -------
    HubsAndAuthorities : dask_cudf.DataFrame
        GPU distributed data frame containing three dask_cudf.Series of
        size V: the vertex identifiers and the corresponding hubs and
        authorities values.

        df['vertex'] : dask_cudf.Series
            Contains the vertex identifiers
        df['hubs'] : dask_cudf.Series
            Contains the hubs score
        df['authorities'] : dask_cudf.Series
            Contains the authorities score

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
    >>> hits = dcg.hits(dg, max_iter = 50)

    """

    client = default_client()

    if input_graph.store_transposed is False:
        warning_msg = (
            "HITS expects the 'store_transposed' flag "
            "to be set to 'True' for optimal performance during "
            "the graph creation"
        )
        warnings.warn(warning_msg, UserWarning)

    do_expensive_check = False
    initial_hubs_guess_vertices = None
    initial_hubs_guess_values = None

    if nstart is not None:
        initial_hubs_guess_vertices = nstart["vertex"]
        initial_hubs_guess_values = nstart["values"]

    cupy_result = [
        client.submit(
            _call_plc_hits,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            tol,
            max_iter,
            initial_hubs_guess_vertices,
            initial_hubs_guess_values,
            normalized,
            do_expensive_check,
            workers=[w],
            allow_other_workers=False,
        )
        for w in Comms.get_workers()
    ]

    wait(cupy_result)

    cudf_result = [
        client.submit(
            convert_to_cudf, cp_arrays, workers=client.who_has(cp_arrays)[cp_arrays.key]
        )
        for cp_arrays in cupy_result
    ]

    wait(cudf_result)

    ddf = dask_cudf.from_delayed(cudf_result).persist()

    wait(ddf)

    # Wait until the inactive futures are released
    wait([(r.release(), c_r.release()) for r, c_r in zip(cupy_result, cudf_result)])

    if input_graph.renumbered:
        return input_graph.unrenumber(ddf, "vertex")

    return ddf
