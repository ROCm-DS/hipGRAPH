# Copyright (c) 2020-2022, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import cudf
from hipgraph.linear_assignment import lap_wrapper


def hungarian(G, workers, epsilon=None):
    """
    Execute the Hungarian algorithm against a symmetric, weighted,
    bipartite graph.

    As a bipartite graph, the vertex set of the graph can be partitioned
    into two disjoint sets such that all edges connect a vertex from
    one set to a vertex of the other set.  The workers variable identifies
    one of the sets of vertices, the other set is all vertices not in
    the workers set (V - workers).

    The edge weights reflect the cost of assigning a particular job to a
    worker.

    The Hungarian algorithm identifies the lowest cost matching of vertices
    such that all workers that can be assigned work are assigned exactly
    on job.

    Parameters
    ----------
    G : hipgraph.Graph
        hipGRAPH graph descriptor, should contain the connectivity information
        as an an edge list.  Edge weights are required. If an edge list is
        not provided then it will be computed.

    workers : cudf.Series or cudf.DataFrame
        A series or column that identifies the vertex ids of the vertices
        in the workers set.  In case of multi-column vertices, it should be a
        cudf.DataFrame. All vertices in G that are not in the workers
        set are implicitly assigned to the jobs set.

    epsilon : float/double (matching weight in graph), optional (default=None)
        Used for determining when value is close enough to zero to consider 0.
        Defaults (if not specified) to 1e-6 in the C++ code.  Unused for
        integer weight types.

    Returns
    -------
    cost : matches costs.dtype
        The cost of the overall assignment

    df : cudf.DataFrame
      df['vertex'][i] gives the vertex id of the i'th vertex.  Only vertices
                      in the workers list are defined in this column.
      df['assignment'][i] gives the vertex id of the "job" assigned to the
                          corresponding vertex.

    Examples
    --------
    >>> workers, G, costs = hipgraph.utils.create_random_bipartite(5, 5,
    ...                                                           100, float)
    >>> cost, df = hipgraph.hungarian(G, workers)

    """

    if G.renumbered:
        if isinstance(workers, cudf.DataFrame):
            local_workers = G.lookup_internal_vertex_id(workers, workers.columns)
        else:
            local_workers = G.lookup_internal_vertex_id(workers)
    else:
        local_workers = workers

    cost, df = lap_wrapper.sparse_hungarian(G, local_workers, epsilon)

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    return cost, df


def dense_hungarian(costs, num_rows, num_columns, epsilon=None):
    """
    Execute the Hungarian algorithm against a dense bipartite
    graph representation.

    *NOTE*: This API is unstable and subject to change

    The Hungarian algorithm identifies the lowest cost matching of vertices
    such that all workers that can be assigned work are assigned exactly
    on job.

    Parameters
    ----------
    costs : cudf.Series
        A dense representation (row major order) of the bipartite
        graph.  Each row represents a worker, each column represents
        a task, cost[i][j] represents the cost of worker i performing
        task j.

    num_rows : int
        Number of rows in the matrix

    num_columns : int
        Number of columns in the matrix

    epsilon : float or double (matching weight type in graph)
        Used for determining when value is close enough to zero to consider 0.
        Defaults (if not specified) to 1e-6 in the C++ code.  Unused for
        integer weight types.

    Returns
    -------
    cost : matches costs.dtype
        The cost of the overall assignment

    assignment : cudf.Series
        assignment[i] gives the vertex id of the task assigned to the
                    worker i

    Examples
    --------
    >>> workers, G, costs = hipgraph.utils.create_random_bipartite(5, 5,
    ...                                                           100, float)
    >>> costs_flattened = cudf.Series(costs.flatten())
    >>> cost, assignment = hipgraph.dense_hungarian(costs_flattened, 5, 5)

    """

    return lap_wrapper.dense_hungarian(costs, num_rows, num_columns, epsilon)
