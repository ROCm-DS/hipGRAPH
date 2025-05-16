# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import cupy as cp
import networkx as nx
import numpy as np
import nx_hipgraph as nxcg
from nx_hipgraph.convert import _to_graph
from nx_hipgraph.utils import index_dtype, networkx_algorithm

__all__ = ["complement", "reverse"]


@networkx_algorithm(version_added="24.02")
def complement(G):
    G = _to_graph(G)
    N = G._N
    # Upcast to int64 so indices don't overflow.
    edges_a_b = N * G.src_indices.astype(np.int64) + G.dst_indices
    # Now compute flattened indices for all edges except self-loops
    # Alt (slower):
    # edges_full = np.arange(N * N)
    # edges_full = edges_full[(edges_full % (N + 1)).astype(bool)]
    edges_full = cp.arange(1, N * (N - 1) + 1) + cp.repeat(cp.arange(N - 1), N)
    edges_comp = cp.setdiff1d(
        edges_full,
        edges_a_b,
        assume_unique=not G.is_multigraph(),
    )
    src_indices, dst_indices = cp.divmod(edges_comp, N)
    return G.__class__.from_coo(
        N,
        src_indices.astype(index_dtype),
        dst_indices.astype(index_dtype),
        key_to_id=G.key_to_id,
    )


@networkx_algorithm(version_added="24.02")
def reverse(G, copy=True):
    if not G.is_directed():
        raise nx.NetworkXError("Cannot reverse an undirected graph.")
    if isinstance(G, nx.Graph):
        G = nxcg.from_networkx(G, preserve_all_attrs=True)
    return G.reverse(copy=copy)
