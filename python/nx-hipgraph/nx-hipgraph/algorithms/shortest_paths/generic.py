# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import networkx as nx
import numpy as np
import nx_hipgraph as nxcg
from nx_hipgraph.convert import _to_graph
from nx_hipgraph.utils import _dtype_param, _get_float_dtype, networkx_algorithm

from .unweighted import _bfs
from .weighted import _sssp

__all__ = [
    "shortest_path",
    "shortest_path_length",
    "has_path",
]


@networkx_algorithm(version_added="24.04", _plc="bfs")
def has_path(G, source, target):
    # TODO PERF: make faster in core
    try:
        nxcg.bidirectional_shortest_path(G, source, target)
    except nx.NetworkXNoPath:
        return False
    return True


@networkx_algorithm(
    extra_params=_dtype_param, version_added="24.04", _plc={"bfs", "sssp"}
)
def shortest_path(
    G, source=None, target=None, weight=None, method="dijkstra", *, dtype=None
):
    """Negative weights are not yet supported, and method is ununsed."""
    if method not in {"dijkstra", "bellman-ford"}:
        raise ValueError(f"method not supported: {method}")
    if weight is None:
        method = "unweighted"
    if source is None:
        if target is None:
            # All pairs
            if method == "unweighted":
                paths = nxcg.all_pairs_shortest_path(G)
            else:
                # method == "dijkstra":
                # method == 'bellman-ford':
                paths = nxcg.all_pairs_bellman_ford_path(G, weight=weight, dtype=dtype)
            if nx.__version__[:3] <= "3.4":
                paths = dict(paths)
        # To target
        elif method == "unweighted":
            paths = nxcg.single_target_shortest_path(G, target)
        else:
            # method == "dijkstra":
            # method == 'bellman-ford':
            # XXX: it seems weird that `reverse_path=True` is necessary here
            G = _to_graph(G, weight, 1, np.float32)
            dtype = _get_float_dtype(dtype, graph=G, weight=weight)
            paths = _sssp(
                G, target, weight, return_type="path", dtype=dtype, reverse_path=True
            )
    elif target is None:
        # From source
        if method == "unweighted":
            paths = nxcg.single_source_shortest_path(G, source)
        else:
            # method == "dijkstra":
            # method == 'bellman-ford':
            paths = nxcg.single_source_bellman_ford_path(
                G, source, weight=weight, dtype=dtype
            )
    # From source to target
    elif method == "unweighted":
        paths = nxcg.bidirectional_shortest_path(G, source, target)
    else:
        # method == "dijkstra":
        # method == 'bellman-ford':
        paths = nxcg.bellman_ford_path(G, source, target, weight, dtype=dtype)
    return paths


@shortest_path._can_run
def _(G, source=None, target=None, weight=None, method="dijkstra", *, dtype=None):
    return (
        weight is None
        or not callable(weight)
        and not nx.is_negatively_weighted(G, weight=weight)
    )


@networkx_algorithm(
    extra_params=_dtype_param, version_added="24.04", _plc={"bfs", "sssp"}
)
def shortest_path_length(
    G, source=None, target=None, weight=None, method="dijkstra", *, dtype=None
):
    """Negative weights are not yet supported, and method is ununsed."""
    if method not in {"dijkstra", "bellman-ford"}:
        raise ValueError(f"method not supported: {method}")
    if weight is None:
        method = "unweighted"
    if source is None:
        if target is None:
            # All pairs
            if method == "unweighted":
                lengths = nxcg.all_pairs_shortest_path_length(G)
            else:
                # method == "dijkstra":
                # method == 'bellman-ford':
                lengths = nxcg.all_pairs_bellman_ford_path_length(
                    G, weight=weight, dtype=dtype
                )
        # To target
        elif method == "unweighted":
            lengths = nxcg.single_target_shortest_path_length(G, target)
            if nx.__version__[:3] <= "3.4":
                lengths = dict(lengths)
        else:
            # method == "dijkstra":
            # method == 'bellman-ford':
            lengths = nxcg.single_source_bellman_ford_path_length(
                G, target, weight=weight, dtype=dtype
            )
    elif target is None:
        # From source
        if method == "unweighted":
            lengths = nxcg.single_source_shortest_path_length(G, source)
        else:
            # method == "dijkstra":
            # method == 'bellman-ford':
            lengths = dict(
                nxcg.single_source_bellman_ford_path_length(
                    G, source, weight=weight, dtype=dtype
                )
            )
    # From source to target
    elif method == "unweighted":
        G = _to_graph(G)
        lengths = _bfs(G, source, None, "Source", return_type="length", target=target)
    else:
        # method == "dijkstra":
        # method == 'bellman-ford':
        lengths = nxcg.bellman_ford_path_length(G, source, target, weight, dtype=dtype)
    return lengths


@shortest_path_length._can_run
def _(G, source=None, target=None, weight=None, method="dijkstra", *, dtype=None):
    return (
        weight is None
        or not callable(weight)
        and not nx.is_negatively_weighted(G, weight=weight)
    )
