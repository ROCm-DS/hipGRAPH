# Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import networkx as nx
import nx_hipgraph as nxcg


def assert_graphs_equal(Gnx, Gcg):
    assert isinstance(Gnx, nx.Graph)
    assert isinstance(Gcg, nxcg.Graph)
    assert Gnx.number_of_nodes() == Gcg.number_of_nodes()
    assert Gnx.number_of_edges() == Gcg.number_of_edges()
    assert Gnx.is_directed() == Gcg.is_directed()
    assert Gnx.is_multigraph() == Gcg.is_multigraph()
    G = nxcg.to_networkx(Gcg)
    rv = nx.utils.graphs_equal(G, Gnx)
    if not rv:
        print("GRAPHS ARE NOT EQUAL!")
        assert sorted(G) == sorted(Gnx)
        assert sorted(G._adj) == sorted(Gnx._adj)
        assert sorted(G._node) == sorted(Gnx._node)
        for k in sorted(G._adj):
            print(k, sorted(G._adj[k]), sorted(Gnx._adj[k]))
        print(nx.to_scipy_sparse_array(G).todense())
        print(nx.to_scipy_sparse_array(Gnx).todense())
        print(G.graph)
        print(Gnx.graph)
    assert rv
