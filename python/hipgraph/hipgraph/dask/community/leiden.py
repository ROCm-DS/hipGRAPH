# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import cudf
import cupy as cp
import dask
import dask_cudf
import hipgraph.dask.comms.comms as Comms
import numpy
from dask import delayed
from dask.distributed import default_client, wait
from pylibhipgraph import ResourceHandle
from pylibhipgraph import leiden as pylibhipgraph_leiden

if TYPE_CHECKING:
    from hipgraph import Graph


def convert_to_cudf(result: cp.ndarray) -> Tuple[cudf.DataFrame, float]:
    """
    Creates a cudf DataFrame from cupy arrays from pylibhipgraph wrapper
    """
    cupy_vertex, cupy_partition, modularity = result
    df = cudf.DataFrame()
    df["vertex"] = cupy_vertex
    df["partition"] = cupy_partition

    return df, modularity


def _call_plc_leiden(
    sID: bytes,
    mg_graph_x,
    max_iter: int,
    resolution: int,
    random_state: int,
    theta: int,
    do_expensive_check: bool,
) -> Tuple[cp.ndarray, cp.ndarray, float]:
    return pylibhipgraph_leiden(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        random_state=random_state,
        graph=mg_graph_x,
        max_level=max_iter,
        resolution=resolution,
        theta=theta,
        do_expensive_check=do_expensive_check,
    )


def leiden(
    input_graph: Graph,
    max_iter: int = 100,
    resolution: int = 1.0,
    random_state: int = None,
    theta: int = 1.0,
) -> Tuple[dask_cudf.DataFrame, float]:
    """
    Compute the modularity optimizing partition of the input graph using the
    Leiden method

    Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden:
    guaranteeing well-connected communities. Scientific reports, 9(1), 5233.
    doi: 10.1038/s41598-019-41695-z

    Parameters
    ----------
    G : hipgraph.Graph
        The graph descriptor should contain the connectivity information
        and weights. The adjacency list will be computed if not already
        present.
        The current implementation only supports undirected graphs.

    max_iter : integer, optional (default=100)
        This controls the maximum number of levels/iterations of the Leiden
        algorithm. When specified the algorithm will terminate after no more
        than the specified number of iterations. No error occurs when the
        algorithm terminates early in this manner.

    resolution: float, optional (default=1.0)
        Called gamma in the modularity formula, this changes the size
        of the communities.  Higher resolutions lead to more smaller
        communities, lower resolutions lead to fewer larger communities.
        Defaults to 1.

    random_state: int, optional(default=None)
        Random state to use when generating samples.  Optional argument,
        defaults to a hash of process id, time, and hostname.

    theta: float, optional (default=1.0)
        Called theta in the Leiden algorithm, this is used to scale
        modularity gain in Leiden refinement phase, to compute
        the probability of joining a random leiden community.

    Returns
    -------
    parts : dask_cudf.DataFrame
        GPU data frame of size V containing two columns the vertex id and the
        partition id it is assigned to.

        ddf['vertex'] : cudf.Series
            Contains the vertex identifiers
        ddf['partition'] : cudf.Series
            Contains the partition assigned to the vertices

    modularity_score : float
        a floating point number containing the global modularity score of the
        partitioning.

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
    >>> dg = hipgraph.Graph()
    >>> dg.from_dask_cudf_edgelist(ddf, source='src', destination='dst')
    >>> parts, modularity_score = dcg.leiden(dg)

    """

    if input_graph.is_directed():
        raise ValueError("input graph must be undirected")

    # Return a client if one has started
    client = default_client()

    do_expensive_check = False

    result = [
        client.submit(
            _call_plc_leiden,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            max_iter,
            resolution,
            random_state,
            theta,
            do_expensive_check,
            workers=[w],
            allow_other_workers=False,
        )
        for w in Comms.get_workers()
    ]

    wait(result)

    part_mod_score = [client.submit(convert_to_cudf, r) for r in result]
    wait(part_mod_score)

    vertex_dtype = input_graph.edgelist.edgelist_df.dtypes.iloc[0]
    empty_df = cudf.DataFrame(
        {
            "vertex": numpy.empty(shape=0, dtype=vertex_dtype),
            "partition": numpy.empty(shape=0, dtype="int32"),
        }
    )

    part_mod_score = [delayed(lambda x: x, nout=2)(r) for r in part_mod_score]

    ddf = dask_cudf.from_delayed(
        [r[0] for r in part_mod_score], meta=empty_df, verify_meta=False
    ).persist()

    mod_score = dask.array.from_delayed(
        part_mod_score[0][1], shape=(1,), dtype=float
    ).compute()

    wait(ddf)
    wait(mod_score)

    wait([r.release() for r in part_mod_score])

    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "vertex")

    return ddf, mod_score
