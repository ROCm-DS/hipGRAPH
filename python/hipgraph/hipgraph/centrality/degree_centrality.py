# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

from hipgraph.utilities import df_score_to_dictionary, ensure_hipgraph_obj_for_nx


def degree_centrality(G, normalized=True):
    """
    Computes the degree centrality of each vertex of the input graph.

    Parameters
    ----------
    G : hipGRAPH.Graph or networkx.Graph
        hipGRAPH graph descriptor with connectivity information. The graph can
        contain either directed or undirected edges.

        .. deprecated:: 24.12
           Accepting a ``networkx.Graph`` is deprecated and will be removed in a
           future version.  For ``networkx.Graph`` use networkx directly with
           the ``nx-hipgraph`` backend. See:  https://rapids.ai/nx-hipgraph/

    normalized : bool, optional, default=True
        If True normalize the resulting degree centrality values

    Returns
    -------
    df : cudf.DataFrame or Dictionary if using NetworkX
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding degree centrality values.

        df['vertex'] : cudf.Series
            Contains the vertex identifiers

        df['degree_centrality'] : cudf.Series
            Contains the degree centrality of vertices

    Examples
    --------
    >>> from hipgraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> dc = hipgraph.degree_centrality(G)

    """
    G, isNx = ensure_hipgraph_obj_for_nx(G)

    df = G.degree()
    df.rename(columns={"degree": "degree_centrality"}, inplace=True)

    if normalized:
        df["degree_centrality"] /= G.number_of_nodes() - 1

    if isNx is True:
        dict = df_score_to_dictionary(df, "degree_centrality")
        return dict
    else:
        return df
