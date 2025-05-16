# Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import operator as op

import cupy as cp
import networkx as nx
import nx_hipgraph as nxcg

from ..utils import index_dtype

# 3.2.1 fixed some issues in generators that occur in 3.2 and earlier
_IS_NX32_OR_LESS = (nxver := nx.__version__)[:3] <= "3.2" and (
    len(nxver) <= 3 or nxver[3] != "." and not nxver[3].isdigit()
)


def _ensure_int(n):
    """Ensure n is integral."""
    return op.index(n)


def _ensure_nonnegative_int(n):
    """Ensure n is a nonnegative integer."""
    n = op.index(n)
    if n < 0:
        raise nx.NetworkXError(f"Negative number of nodes not valid: {n}")
    return n


def _complete_graph_indices(n):
    all_indices = cp.indices((n, n), dtype=index_dtype)
    src_indices = all_indices[0].ravel()
    dst_indices = all_indices[1].ravel()
    del all_indices
    mask = src_indices != dst_indices
    return (src_indices[mask], dst_indices[mask])


def _common_small_graph(n, nodes, create_using, *, allow_directed=True):
    """Create a "common graph" for small n.

    n == 0: empty graph
    n == 1: empty graph
    n == 2: complete graph
    n > 2: undefined
    """
    graph_class, inplace = _create_using_class(create_using)
    if not allow_directed and graph_class.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")
    if n < 2:
        G = graph_class.from_coo(
            n, cp.empty(0, index_dtype), cp.empty(0, index_dtype), id_to_key=nodes
        )
    else:
        G = graph_class.from_coo(
            n,
            cp.arange(2, dtype=index_dtype),
            cp.array([1, 0], index_dtype),
            id_to_key=nodes,
        )
    if inplace:
        return create_using._become(G)
    return G


def _create_using_class(create_using, *, default=nxcg.Graph):
    """Handle ``create_using`` argument and return a Graph type from nx_hipgraph."""
    inplace = False
    if create_using is None:
        G = default()
    elif isinstance(create_using, type):
        G = create_using()
    elif not hasattr(create_using, "is_directed") or not hasattr(
        create_using, "is_multigraph"
    ):
        raise TypeError("create_using is not a valid graph type or instance")
    elif not isinstance(create_using, nxcg.Graph):
        raise NotImplementedError(
            f"create_using with object of type {type(create_using)} is not supported "
            "by the hipgraph backend; only nx_hipgraph.Graph objects are allowed."
        )
    else:
        inplace = True
        G = create_using
        G.clear()
    if not isinstance(G, nxcg.Graph):
        if G.is_multigraph():
            if G.is_directed():
                graph_class = nxcg.MultiDiGraph
            else:
                graph_class = nxcg.MultiGraph
        elif G.is_directed():
            graph_class = nxcg.DiGraph
        else:
            graph_class = nxcg.Graph
        if G.__class__ not in {nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph}:
            raise NotImplementedError(
                f"create_using with type {type(G)} is not supported by the hipgraph "
                "backend; only standard networkx or nx_hipgraph Graph objects are "
                "allowed (but not customized subclasses derived from them)."
            )
    else:
        graph_class = G.__class__
    return graph_class, inplace


def _number_and_nodes(n_and_nodes):
    n, nodes = n_and_nodes
    try:
        n = op.index(n)
    except TypeError:
        n = len(nodes)
    if n < 0:
        raise nx.NetworkXError(f"Negative number of nodes not valid: {n}")
    if not isinstance(nodes, list):
        nodes = list(nodes)
    if not nodes:
        return (n, None)
    if nodes[0] == 0 and nodes[n - 1] == n - 1:
        try:
            if nodes == list(range(n)):
                return (n, None)
        except Exception:
            pass
    return (n, nodes)
