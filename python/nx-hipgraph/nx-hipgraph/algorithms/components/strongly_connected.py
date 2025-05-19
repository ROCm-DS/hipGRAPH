# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import cupy as cp
import networkx as nx
import pylibhipgraph as plc
from nx_hipgraph.convert import _to_directed_graph
from nx_hipgraph.utils import _groupby, index_dtype, not_implemented_for

__all__ = [
    "number_strongly_connected_components",
    "strongly_connected_components",
    "is_strongly_connected",
]


def _strongly_connected_components(G):
    # TODO: create utility function to convert just the indices to CSR
    # TODO: this uses a legacy PLC function (strongly_connected_components)
    N = len(G)
    indices = cp.lexsort(cp.vstack((G.dst_indices, G.src_indices)))
    dst_indices = G.dst_indices[indices]
    offsets = cp.searchsorted(
        G.src_indices, cp.arange(N + 1, dtype=index_dtype), sorter=indices
    ).astype(index_dtype)
    labels = cp.zeros(N, dtype=index_dtype)
    plc.strongly_connected_components(
        offsets=offsets,
        indices=dst_indices,
        weights=None,
        num_verts=N,
        num_edges=dst_indices.size,
        labels=labels,
    )
    return labels


# The networkx_algorithm decorator is (temporarily) removed to disable
# dispatching for this function. The current hipgraph
# strongly_connected_components is a legacy implementation with known issues,
# and in most cases should not be used until the hipgraph team can provide an
# update.
#
# Users can still call this via the nx_hipgraph module directly:
# >>> import nx_hipgraph as nxcg
# >>> nxcg.strongly_connected_components(...)


@not_implemented_for("undirected")
# @networkx_algorithm(version_added="24.02", _plc="strongly_connected_components")
def strongly_connected_components(G):
    G = _to_directed_graph(G)
    if G.src_indices.size == 0:
        return [{key} for key in G._nodeiter_to_iter(range(len(G)))]
    labels = _strongly_connected_components(G)
    groups = _groupby(labels, cp.arange(len(G), dtype=index_dtype))
    return (G._nodearray_to_set(connected_ids) for connected_ids in groups.values())


@not_implemented_for("undirected")
# @networkx_algorithm(version_added="24.02", _plc="strongly_connected_components")
def number_strongly_connected_components(G):
    G = _to_directed_graph(G)
    if G.src_indices.size == 0:
        return len(G)
    labels = _strongly_connected_components(G)
    return cp.unique(labels).size


@not_implemented_for("undirected")
# @networkx_algorithm(version_added="24.02", _plc="strongly_connected_components")
def is_strongly_connected(G):
    G = _to_directed_graph(G)
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept(
            "Connectivity is undefined for the null graph."
        )
    if G.src_indices.size == 0:
        return len(G) == 1
    labels = _strongly_connected_components(G)
    return bool((labels == labels[0]).all())
