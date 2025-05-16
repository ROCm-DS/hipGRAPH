# Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import pylibhipgraph as plc
from nx_hipgraph.convert import _to_graph
from nx_hipgraph.utils import _seed_to_int, networkx_algorithm

__all__ = ["betweenness_centrality", "edge_betweenness_centrality"]


@networkx_algorithm(
    is_incomplete=True,  # weight not supported
    is_different=True,  # RNG with seed is different
    version_added="23.10",
    _plc="betweenness_centrality",
)
def betweenness_centrality(
    G, k=None, normalized=True, weight=None, endpoints=False, seed=None
):
    """`weight` parameter is not yet supported, and RNG with seed may be different."""
    if weight is not None:
        raise NotImplementedError(
            "Weighted implementation of betweenness centrality not currently supported"
        )
    seed = _seed_to_int(seed)
    G = _to_graph(G, weight)
    node_ids, values = plc.betweenness_centrality(
        resource_handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(),
        k=k,
        random_state=seed,
        normalized=normalized,
        include_endpoints=endpoints,
        do_expensive_check=False,
    )
    return G._nodearrays_to_dict(node_ids, values)


@betweenness_centrality._can_run
def _(G, k=None, normalized=True, weight=None, endpoints=False, seed=None):
    return weight is None


@networkx_algorithm(
    is_incomplete=True,  # weight not supported
    is_different=True,  # RNG with seed is different
    version_added="23.10",
    _plc="edge_betweenness_centrality",
)
def edge_betweenness_centrality(G, k=None, normalized=True, weight=None, seed=None):
    """`weight` parameter is not yet supported, and RNG with seed may be different."""
    if weight is not None:
        raise NotImplementedError(
            "Weighted implementation of betweenness centrality not currently supported"
        )
    seed = _seed_to_int(seed)
    G = _to_graph(G, weight)
    src_ids, dst_ids, values, _edge_ids = plc.edge_betweenness_centrality(
        resource_handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(),
        k=k,
        random_state=seed,
        normalized=normalized,
        do_expensive_check=False,
    )
    if not G.is_directed():
        mask = src_ids <= dst_ids
        src_ids = src_ids[mask]
        dst_ids = dst_ids[mask]
        values = 2 * values[mask]
    return G._edgearrays_to_dict(src_ids, dst_ids, values)


@edge_betweenness_centrality._can_run
def _(G, k=None, normalized=True, weight=None, seed=None):
    return weight is None
