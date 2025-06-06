# Copyright (c) 2019-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import warnings
from typing import Union

import cudf
import numpy as np
from hipgraph.utilities import (
    df_edge_score_to_dictionary,
    df_score_to_dictionary,
    ensure_hipgraph_obj_for_nx,
)
from pylibhipgraph import ResourceHandle
from pylibhipgraph import betweenness_centrality as pylibhipgraph_betweenness_centrality
from pylibhipgraph import (
    edge_betweenness_centrality as pylibhipgraph_edge_betweenness_centrality,
)


def betweenness_centrality(
    G,
    k: Union[int, list, cudf.Series, cudf.DataFrame] = None,
    normalized: bool = True,
    weight: cudf.DataFrame = None,
    endpoints: bool = False,
    seed: int = None,
    random_state: int = None,
    result_dtype: Union[np.float32, np.float64] = np.float64,
) -> Union[cudf.DataFrame, dict]:
    """
    Compute the betweenness centrality for all vertices of the graph G.
    Betweenness centrality is a measure of the number of shortest paths that
    pass through a vertex.  A vertex with a high betweenness centrality score
    has more paths passing through it and is therefore believed to be more
    important.

    To improve performance. rather than doing an all-pair shortest path,
    a sample of k starting vertices can be used.

    hipGRAPH does not currently support 'weight' parameters.

    Parameters
    ----------
    G : hipGRAPH.Graph or networkx.Graph
        The graph can be either directed (Graph(directed=True)) or undirected.
        The current implementation uses a parallel variation of the Brandes
        Algorithm (2001) to compute exact or approximate betweenness.
        If weights are provided in the edgelist, they will not be used.

        .. deprecated:: 24.12
           Accepting a ``networkx.Graph`` is deprecated and will be removed in a
           future version.  For ``networkx.Graph`` use networkx directly with
           the ``nx-hipgraph`` backend. See:  https://rapids.ai/nx-hipgraph/

    k : int, list or cudf object or None, optional (default=None)
        If k is not None, use k node samples to estimate betweenness. Higher
        values give better approximation.  If k is either a list, a cudf DataFrame,
        or a dask_cudf DataFrame, then its contents are assumed to be vertex
        identifiers to be used for estimation. If k is None (the default), all the
        vertices are used to estimate betweenness. Vertices obtained through
        sampling or defined as a list will be used as sources for traversals inside
        the algorithm.

    normalized : bool, optional (default=True)
        If true, the betweenness values are normalized by
        __2 / ((n - 1) * (n - 2))__ for undirected Graphs, and
        __1 / ((n - 1) * (n - 2))__ for directed Graphs
        where n is the number of nodes in G.
        Normalization will ensure that values are in [0, 1],
        this normalization scales for the highest possible value where one
        node is crossed by every single shortest path.

    weight : cudf.DataFrame, optional (default=None)
        Specifies the weights to be used for each edge.
        Should contain a mapping between
        edges and weights.

        (Not Supported): if weights are provided at the Graph creation,
        they will not be used.

    endpoints : bool, optional (default=False)
        If true, include the endpoints in the shortest path counts.

    seed : int, optional (default=None)
        if k is specified and k is an integer, use seed to initialize
        the random number generator.
        Using None defaults to a hash of process id, time, and hostname
        If k is either None or list: seed parameter is ignored.

        This parameter is here for backwards-compatibility and identical
        to 'random_state'.

    random_state : int, optional (default=None)
        if k is specified and k is an integer, use random_state to initialize
        the random number generator.
        Using None defaults to a hash of process id, time, and hostname
        If k is either None or list: random_state parameter is ignored.

    result_dtype : np.float32 or np.float64, optional, default=np.float64
        Indicate the data type of the betweenness centrality scores.

    Returns
    -------
    df : cudf.DataFrame or Dictionary if using NetworkX
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding betweenness centrality values.
        Please note that the resulting the 'vertex' column might not be
        in ascending order.  The Dictionary contains the same two columns

        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df['betweenness_centrality'] : cudf.Series
            Contains the betweenness centrality of vertices

    Examples
    --------
    >>> from hipgraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> bc = hipgraph.betweenness_centrality(G)

    """

    if seed is not None:
        warning_msg = (
            "This parameter is deprecated and will be remove "
            "in the next release. Use 'random_state' instead."
        )
        warnings.warn(warning_msg, UserWarning)

    G, isNx = ensure_hipgraph_obj_for_nx(G)

    if weight is not None:
        raise NotImplementedError(
            "weighted implementation of betweenness "
            "centrality not currently supported"
        )

    if G.store_transposed is True:
        warning_msg = (
            "Betweenness centrality expects the 'store_transposed' flag "
            "to be set to 'False' for optimal performance during "
            "the graph creation"
        )
        warnings.warn(warning_msg, UserWarning)

    # FIXME: Should we now remove this paramter?
    if result_dtype not in [np.float32, np.float64]:
        raise TypeError("result type can only be np.float32 or np.float64")
    else:
        warning_msg = (
            "This parameter is deprecated and will be remove " "in the next release."
        )
        warnings.warn(warning_msg, PendingDeprecationWarning)

    if not isinstance(k, (cudf.DataFrame, cudf.Series)):
        if isinstance(k, list):
            vertex_dtype = G.edgelist.edgelist_df.dtypes.iloc[0]
            k = cudf.Series(k, dtype=vertex_dtype)

    if isinstance(k, (cudf.DataFrame, cudf.Series)):
        if G.renumbered:
            k = G.lookup_internal_vertex_id(k)

    vertices, values = pylibhipgraph_betweenness_centrality(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        k=k,
        random_state=random_state,
        normalized=normalized,
        include_endpoints=endpoints,
        do_expensive_check=False,
    )

    vertices = cudf.Series(vertices)
    values = cudf.Series(values)

    df = cudf.DataFrame()
    df["vertex"] = vertices
    df["betweenness_centrality"] = values

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    if df["betweenness_centrality"].dtype != result_dtype:
        df["betweenness_centrality"] = df["betweenness_centrality"].astype(result_dtype)

    if isNx is True:
        dict = df_score_to_dictionary(df, "betweenness_centrality")
        return dict
    else:
        return df


def edge_betweenness_centrality(
    G,
    k: Union[int, list, cudf.Series, cudf.DataFrame] = None,
    normalized: bool = True,
    weight: cudf.DataFrame = None,
    seed: int = None,
    result_dtype: Union[np.float32, np.float64] = np.float64,
) -> Union[cudf.DataFrame, dict]:
    """
    Compute the edge betweenness centrality for all edges of the graph G.
    Betweenness centrality is a measure of the number of shortest paths
    that pass over an edge.  An edge with a high betweenness centrality
    score has more paths passing over it and is therefore believed to be
    more important.

    To improve performance, rather than doing an all-pair shortest path,
    a sample of k starting vertices can be used.

    hipGRAPH does not currently support the 'weight' parameter.

    Parameters
    ----------
    G : hipGRAPH.Graph or networkx.Graph
        The graph can be either directed (Graph(directed=True)) or undirected.
        The current implementation uses BFS traversals. Use weight parameter
        if weights need to be considered (currently not supported).

        .. deprecated:: 24.12
           Accepting a ``networkx.Graph`` is deprecated and will be removed in a
           future version.  For ``networkx.Graph`` use networkx directly with
           the ``nx-hipgraph`` backend. See:  https://rapids.ai/nx-hipgraph/

    k : int or list or None, optional (default=None)
        If k is not None, use k node samples to estimate betweenness. Higher
        values give better approximation.  If k is either a list, a cudf DataFrame,
        or a dask_cudf DataFrame, then its contents are assumed to be vertex
        identifiers to be used for estimation. If k is None (the default), all the
        vertices are used to estimate betweenness. Vertices obtained through
        sampling or defined as a list will be used as sources for traversals inside
        the algorithm.

    normalized : bool, optional (default=True)
        If true, the betweenness values are normalized by
        __2 / (n * (n - 1))__ for undirected Graphs, and
        __1 / (n * (n - 1))__ for directed Graphs
        where n is the number of nodes in G.
        Normalization will ensure that values are in [0, 1],
        this normalization scales for the highest possible value where one
        edge is crossed by every single shortest path.

    weight : cudf.DataFrame, optional (default=None)
        Specifies the weights to be used for each edge.
        Should contain a mapping between
        edges and weights.
        (Not Supported)

    seed : optional (default=None)
        if k is specified and k is an integer, use seed to initialize the
        random number generator.
        Using None as seed relies on random.seed() behavior: using current
        system time
        If k is either None or list: seed parameter is ignored

    result_dtype : np.float32 or np.float64, optional (default=np.float64)
        Indicate the data type of the betweenness centrality scores
        Using double automatically switch implementation to "default"

    Returns
    -------
    df : cudf.DataFrame or Dictionary if using NetworkX
        GPU data frame containing three cudf.Series of size E: the vertex
        identifiers of the sources, the vertex identifies of the destinations
        and the corresponding betweenness centrality values.
        Please note that the resulting the 'src', 'dst' column might not be
        in ascending order.

        df['src'] : cudf.Series
            Contains the vertex identifiers of the source of each edge

        df['dst'] : cudf.Series
            Contains the vertex identifiers of the destination of each edge

        df['betweenness_centrality'] : cudf.Series
            Contains the betweenness centrality of edges

        df["edge_id"] : cudf.Series
            Contains the edge ids of edges if present.


    Examples
    --------
    >>> from hipgraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> bc = hipgraph.edge_betweenness_centrality(G)

    """
    if weight is not None:
        raise NotImplementedError(
            "weighted implementation of edge betweenness "
            "centrality not currently supported"
        )
    if result_dtype not in [np.float32, np.float64]:
        raise TypeError("result type can only be np.float32 or np.float64")

    G, isNx = ensure_hipgraph_obj_for_nx(G)

    if not isinstance(k, (cudf.DataFrame, cudf.Series)):
        if isinstance(k, list):
            vertex_dtype = G.edgelist.edgelist_df.dtypes.iloc[0]
            k = cudf.Series(k, dtype=vertex_dtype)

    if isinstance(k, (cudf.DataFrame, cudf.Series)):
        if G.renumbered:
            k = G.lookup_internal_vertex_id(k)

    # FIXME: src, dst and edge_ids need to be of the same type which should not
    # be the case

    (
        src_vertices,
        dst_vertices,
        values,
        edge_ids,
    ) = pylibhipgraph_edge_betweenness_centrality(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        k=k,
        random_state=seed,
        normalized=normalized,
        do_expensive_check=False,
    )

    df = cudf.DataFrame()
    df["src"] = src_vertices
    df["dst"] = dst_vertices
    df["betweenness_centrality"] = values
    if edge_ids is not None:
        df["edge_id"] = edge_ids

    if G.renumbered:
        df = G.unrenumber(df, "src")
        df = G.unrenumber(df, "dst")

    if df["betweenness_centrality"].dtype != result_dtype:
        df["betweenness_centrality"] = df["betweenness_centrality"].astype(result_dtype)

    if G.is_directed() is False:
        # select the lower triangle of the df based on src/dst vertex value
        lower_triangle = df["src"] >= df["dst"]
        # swap the src and dst vertices for the lower triangle only. Because
        # this is a symmeterized graph, this operation results in a df with
        # multiple src/dst entries.
        df["src"][lower_triangle], df["dst"][lower_triangle] = (
            df["dst"][lower_triangle],
            df["src"][lower_triangle],
        )
        # overwrite the df with the sum of the values for all alike src/dst
        # vertex pairs, resulting in half the edges of the original df from the
        # symmeterized graph.
        df = df.groupby(by=["src", "dst"]).sum().reset_index()

    if isNx is True:
        return df_edge_score_to_dictionary(df, "betweenness_centrality")
    else:
        return df
