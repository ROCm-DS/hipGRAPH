# Copyright (c) 2019-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

from hipgraph.structure.graph_classes import Graph
from hipgraph.tree import minimum_spanning_tree_wrapper
from hipgraph.utilities import ensure_hipgraph_obj_for_nx, hipgraph_to_nx


def _minimum_spanning_tree_subgraph(G):
    mst_subgraph = Graph()
    if G.is_directed():
        raise ValueError("input graph must be undirected")
    mst_df = minimum_spanning_tree_wrapper.minimum_spanning_tree(G)
    if G.renumbered:
        mst_df = G.unrenumber(mst_df, "src")
        mst_df = G.unrenumber(mst_df, "dst")

    mst_subgraph.from_cudf_edgelist(
        mst_df, source="src", destination="dst", edge_attr="weight"
    )
    return mst_subgraph


def _maximum_spanning_tree_subgraph(G):
    mst_subgraph = Graph()
    if G.is_directed():
        raise ValueError("input graph must be undirected")

    if not G.adjlist:
        G.view_adj_list()

    if G.adjlist.weights is not None:
        G.adjlist.weights = G.adjlist.weights.mul(-1)

    mst_df = minimum_spanning_tree_wrapper.minimum_spanning_tree(G)

    # revert to original weights
    if G.adjlist.weights is not None:
        G.adjlist.weights = G.adjlist.weights.mul(-1)
        mst_df["weight"] = mst_df["weight"].mul(-1)

    if G.renumbered:
        mst_df = G.unrenumber(mst_df, "src")
        mst_df = G.unrenumber(mst_df, "dst")

    mst_subgraph.from_cudf_edgelist(
        mst_df, source="src", destination="dst", edge_attr="weight"
    )
    return mst_subgraph


def minimum_spanning_tree(G, weight=None, algorithm="boruvka", ignore_nan=False):
    """
    Returns a minimum spanning tree (MST) or forest (MSF) on an undirected
    graph

    Parameters
    ----------
    G : hipGRAPH.Graph or networkx.Graph
        hipGRAPH graph descriptor with connectivity information.

        .. deprecated:: 24.12
           Accepting a ``networkx.Graph`` is deprecated and will be removed in a
           future version.  For ``networkx.Graph`` use networkx directly with
           the ``nx-hipgraph`` backend. See:  https://rapids.ai/nx-hipgraph/

    weight : string
        default to the weights in the graph, if the graph edges do not have a
        weight attribute a default weight of 1 will be used.

    algorithm : string
        Default to 'boruvka'. The parallel algorithm to use when finding a
        minimum spanning tree.

    ignore_nan : bool
        Default to False

    Returns
    -------
    G_mst : hipGRAPH.Graph or networkx.Graph
        A graph descriptor with a minimum spanning tree or forest.
        The networkx graph will not have all attributes copied over

    Examples
    --------
    >>> from hipgraph.datasets import netscience
    >>> G = netscience.get_graph(download=True)
    >>> G_mst = hipgraph.minimum_spanning_tree(G)

    """
    G, isNx = ensure_hipgraph_obj_for_nx(G)

    if isNx is True:
        mst = _minimum_spanning_tree_subgraph(G)
        return hipgraph_to_nx(mst)
    else:
        return _minimum_spanning_tree_subgraph(G)


def maximum_spanning_tree(G, weight=None, algorithm="boruvka", ignore_nan=False):
    """
    Returns a maximum spanning tree (MST) or forest (MSF) on an undirected
    graph. Also computes the adjacency list if G does not have one.

    Parameters
    ----------
    G : hipGRAPH.Graph or networkx.Graph
        hipGRAPH graph descriptor with connectivity information.

        .. deprecated:: 24.12
           Accepting a ``networkx.Graph`` is deprecated and will be removed in a
           future version.  For ``networkx.Graph`` use networkx directly with
           the ``nx-hipgraph`` backend. See:  https://rapids.ai/nx-hipgraph/

    weight : string
        default to the weights in the graph, if the graph edges do not have a
        weight attribute a default weight of 1 will be used.

    algorithm : string
        Default to 'boruvka'. The parallel algorithm to use when finding a
        maximum spanning tree.

    ignore_nan : bool
        Default to False

    Returns
    -------
    G_mst : hipGRAPH.Graph or networkx.Graph
        A graph descriptor with a maximum spanning tree or forest.
        The networkx graph will not have all attributes copied over

    Examples
    --------
    >>> from hipgraph.datasets import netscience
    >>> G = netscience.get_graph(download=True)
    >>> G_mst = hipgraph.maximum_spanning_tree(G)

    """
    G, isNx = ensure_hipgraph_obj_for_nx(G)

    if isNx is True:
        mst = _maximum_spanning_tree_subgraph(G)
        return hipgraph_to_nx(mst)
    else:
        return _maximum_spanning_tree_subgraph(G)
