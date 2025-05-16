# Copyright (c) 2019-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import cudf
from hipgraph.utilities import df_score_to_dictionary, ensure_hipgraph_obj_for_nx
from pylibhipgraph import ResourceHandle
from pylibhipgraph import core_number as pylibhipgraph_core_number


def core_number(G, degree_type="bidirectional"):
    """
    Compute the core numbers for the nodes of the graph G. A k-core of a graph
    is a maximal subgraph that contains nodes of degree k or more.  A node has
    a core number of k if it belongs to a k-core but not to k+1-core.  This
    call does not support a graph with self-loops and parallel edges.

    Parameters
    ----------
    G : hipGRAPH.Graph or networkx.Graph
        The current implementation only supports undirected graphs.  The graph
        can contain edge weights, but they don't participate in the calculation
        of the core numbers.

        .. deprecated:: 24.12
           Accepting a ``networkx.Graph`` is deprecated and will be removed in a
           future version.  For ``networkx.Graph`` use networkx directly with
           the ``nx-hipgraph`` backend. See:  https://rapids.ai/nx-hipgraph/

    degree_type: str, (default="bidirectional")
        This option is currently ignored.  This option may eventually determine
        if the core number computation should be based on input, output, or
        both directed edges, with valid values being "incoming", "outgoing",
        and "bidirectional" respectively.

    Returns
    -------
    df : cudf.DataFrame or python dictionary (in NetworkX input)
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding core number values.

        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df['core_number'] : cudf.Series
            Contains the core number of vertices

    Examples
    --------
    >>> from hipgraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> df = hipgraph.core_number(G)
    >>> df.head()
       vertex  core_number
    0      33            4
    1       0            4
    2      32            4
    3       2            4
    4       1            4
    """

    G, isNx = ensure_hipgraph_obj_for_nx(G)

    if G.is_directed():
        raise ValueError("input graph must be undirected")

    # degree_type is currently ignored until libhipgraph supports directed
    # graphs for core_number. Once supporteed, degree_type should be checked
    # like so:
    # if degree_type not in ["incoming", "outgoing", "bidirectional"]:
    #     raise ValueError(
    #         f"'degree_type' must be either incoming, "
    #         f"outgoing or bidirectional, got: {degree_type}"
    #     )

    vertex, core_number = pylibhipgraph_core_number(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        degree_type=degree_type,
        do_expensive_check=False,
    )

    df = cudf.DataFrame()
    df["vertex"] = vertex
    df["core_number"] = core_number

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    if isNx is True:
        df = df_score_to_dictionary(df, "core_number")

    return df
