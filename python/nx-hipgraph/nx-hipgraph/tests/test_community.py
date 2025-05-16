# Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import networkx as nx
import nx_hipgraph as nxcg
import pytest


def test_louvain_isolated_nodes():
    is_nx_30_or_31 = hasattr(nx.classes, "backends")

    def check(left, right):
        assert len(left) == len(right)
        assert set(map(frozenset, left)) == set(map(frozenset, right))

    # Empty graph (no nodes)
    G = nx.Graph()
    if is_nx_30_or_31:
        with pytest.raises(ZeroDivisionError):
            nx.community.louvain_communities(G)
    else:
        nx_result = nx.community.louvain_communities(G)
        cg_result = nxcg.community.louvain_communities(G)
        check(nx_result, cg_result)
    # Graph with no edges
    G.add_nodes_from(range(5))
    if is_nx_30_or_31:
        with pytest.raises(ZeroDivisionError):
            nx.community.louvain_communities(G)
    else:
        nx_result = nx.community.louvain_communities(G)
        cg_result = nxcg.community.louvain_communities(G)
        check(nx_result, cg_result)
    # Graph with isolated nodes
    G.add_edge(1, 2)
    nx_result = nx.community.louvain_communities(G)
    cg_result = nxcg.community.louvain_communities(G)
    check(nx_result, cg_result)
    # Another one
    G.add_edge(4, 4)
    nx_result = nx.community.louvain_communities(G)
    cg_result = nxcg.community.louvain_communities(G)
    check(nx_result, cg_result)
