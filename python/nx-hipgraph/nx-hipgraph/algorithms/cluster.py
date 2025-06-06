# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import cupy as cp
import pylibhipgraph as plc
from nx_hipgraph.convert import _to_undirected_graph
from nx_hipgraph.utils import networkx_algorithm, not_implemented_for

__all__ = [
    "triangles",
    "average_clustering",
    "clustering",
    "transitivity",
]


def _triangles(G, nodes, symmetrize=None):
    if nodes is not None:
        if is_single_node := (nodes in G):
            nodes = [nodes if G.key_to_id is None else G.key_to_id[nodes]]
        else:
            nodes = list(nodes)
        nodes = G._list_to_nodearray(nodes)
    else:
        is_single_node = False
    if len(G) == 0:
        return None, None, is_single_node
    node_ids, triangles = plc.triangle_count(
        resource_handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(symmetrize=symmetrize),
        start_list=nodes,
        do_expensive_check=False,
    )
    return node_ids, triangles, is_single_node


@not_implemented_for("directed")
@networkx_algorithm(version_added="24.02", _plc="triangle_count")
def triangles(G, nodes=None):
    G = _to_undirected_graph(G)
    node_ids, triangles, is_single_node = _triangles(G, nodes)
    if len(G) == 0:
        return {}
    if is_single_node:
        return int(triangles[0])
    return G._nodearrays_to_dict(node_ids, triangles)


@triangles._should_run
def _(G, nodes=None):
    if nodes is None or nodes not in G:
        return True
    return "Fast algorithm when computing for a single node; not worth converting."


@not_implemented_for("directed")
@networkx_algorithm(is_incomplete=True, version_added="24.02", _plc="triangle_count")
def clustering(G, nodes=None, weight=None):
    """Directed graphs and `weight` parameter are not yet supported."""
    if weight is not None:
        raise NotImplementedError(
            "Weighted implementation of clustering not currently supported"
        )
    G = _to_undirected_graph(G)
    node_ids, triangles, is_single_node = _triangles(G, nodes)
    if len(G) == 0:
        return {}
    if is_single_node:
        numer = int(triangles[0])
        if numer == 0:
            return 0
        degree = int((G.src_indices == nodes).sum())
        return 2 * numer / (degree * (degree - 1))
    degrees = G._degrees_array(ignore_selfloops=True)[node_ids]
    denom = degrees * (degrees - 1)
    results = 2 * triangles / denom
    results = cp.where(denom, results, 0)  # 0 where we divided by 0
    return G._nodearrays_to_dict(node_ids, results)


@clustering._can_run
def _(G, nodes=None, weight=None):
    return weight is None and not G.is_directed()


@clustering._should_run
def _(G, nodes=None, weight=None):
    if nodes is None or nodes not in G:
        return True
    return "Fast algorithm when computing for a single node; not worth converting."


@not_implemented_for("directed")
@networkx_algorithm(is_incomplete=True, version_added="24.02", _plc="triangle_count")
def average_clustering(G, nodes=None, weight=None, count_zeros=True):
    """Directed graphs and `weight` parameter are not yet supported."""
    if weight is not None:
        raise NotImplementedError(
            "Weighted implementation of average_clustering not currently supported"
        )
    G = _to_undirected_graph(G)
    node_ids, triangles, is_single_node = _triangles(G, nodes)
    if len(G) == 0:
        raise ZeroDivisionError
    degrees = G._degrees_array(ignore_selfloops=True)[node_ids]
    if not count_zeros:
        mask = triangles != 0
        triangles = triangles[mask]
        if triangles.size == 0:
            raise ZeroDivisionError
        degrees = degrees[mask]
    denom = degrees * (degrees - 1)
    results = 2 * triangles / denom
    if count_zeros:
        results = cp.where(denom, results, 0)  # 0 where we divided by 0
    return float(results.mean())


@average_clustering._can_run
def _(G, nodes=None, weight=None, count_zeros=True):
    return weight is None and not G.is_directed()


@not_implemented_for("directed")
@networkx_algorithm(is_incomplete=True, version_added="24.02", _plc="triangle_count")
def transitivity(G):
    """Directed graphs are not yet supported."""
    G = _to_undirected_graph(G)
    if len(G) == 0:
        return 0
    node_ids, triangles = plc.triangle_count(
        resource_handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(),
        start_list=None,
        do_expensive_check=False,
    )
    numer = int(triangles.sum())
    if numer == 0:
        return 0
    degrees = G._degrees_array(ignore_selfloops=True)[node_ids]
    denom = int((degrees * (degrees - 1)).sum())
    return 2 * numer / denom


@transitivity._can_run
def _(G):
    # Is transitivity supposed to work on directed graphs?
    return not G.is_directed()
