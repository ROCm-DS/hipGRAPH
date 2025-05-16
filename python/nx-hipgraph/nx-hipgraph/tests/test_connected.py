# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import networkx as nx
import nx_hipgraph as nxcg


def test_connected_isolated_nodes():
    G = nx.complete_graph(4)
    G.add_node(max(G) + 1)
    assert nx.is_connected(G) is False
    assert nxcg.is_connected(G) is False
    assert nx.number_connected_components(G) == 2
    assert nxcg.number_connected_components(G) == 2
    assert sorted(nx.connected_components(G)) == [{0, 1, 2, 3}, {4}]
    assert sorted(nxcg.connected_components(G)) == [{0, 1, 2, 3}, {4}]
    assert nx.node_connected_component(G, 0) == {0, 1, 2, 3}
    assert nxcg.node_connected_component(G, 0) == {0, 1, 2, 3}
    assert nx.node_connected_component(G, 4) == {4}
    assert nxcg.node_connected_component(G, 4) == {4}
