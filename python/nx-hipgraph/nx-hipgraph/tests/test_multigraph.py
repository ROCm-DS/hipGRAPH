# Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import networkx as nx
import nx_hipgraph as nxcg
import pytest


@pytest.mark.parametrize("test_nxhipgraph", [False, True])
def test_get_edge_data(test_nxhipgraph):
    G = nx.MultiGraph()
    G.add_edge(0, 1, 0, x=10)
    G.add_edge(0, 1, 1, y=20)
    G.add_edge(0, 2, "a", x=100)
    G.add_edge(0, 2, "b", y=200)
    G.add_edge(0, 3)
    G.add_edge(0, 3)
    if test_nxhipgraph:
        G = nxcg.MultiGraph(G)
    default = object()
    assert G.get_edge_data(0, 0, default=default) is default
    assert G.get_edge_data("a", "b", default=default) is default
    assert G.get_edge_data(0, 1, 2, default=default) is default
    assert G.get_edge_data(-1, 1, default=default) is default
    assert G.get_edge_data(0, 1, 0, default=default) == {"x": 10}
    assert G.get_edge_data(0, 1, 1, default=default) == {"y": 20}
    assert G.get_edge_data(0, 1, default=default) == {0: {"x": 10}, 1: {"y": 20}}
    assert G.get_edge_data(0, 2, "a", default=default) == {"x": 100}
    assert G.get_edge_data(0, 2, "b", default=default) == {"y": 200}
    assert G.get_edge_data(0, 2, default=default) == {"a": {"x": 100}, "b": {"y": 200}}
    assert G.get_edge_data(0, 3, 0, default=default) == {}
    assert G.get_edge_data(0, 3, 1, default=default) == {}
    assert G.get_edge_data(0, 3, 2, default=default) is default
    assert G.get_edge_data(0, 3, default=default) == {0: {}, 1: {}}
    assert G.has_edge(0, 1)
    assert G.has_edge(0, 1, 0)
    assert G.has_edge(0, 1, 1)
    assert not G.has_edge(0, 1, 2)
    assert not G.has_edge(0, 1, "a")
    assert not G.has_edge(0, -1)
    assert G.has_edge(0, 2)
    assert G.has_edge(0, 2, "a")
    assert G.has_edge(0, 2, "b")
    assert not G.has_edge(0, 2, "c")
    assert not G.has_edge(0, 2, 0)
    assert G.has_edge(0, 3)
    assert not G.has_edge(0, 0)
    assert not G.has_edge(0, 0, 0)

    G = nx.MultiGraph()
    G.add_edge(0, 1)
    if test_nxhipgraph:
        G = nxcg.MultiGraph(G)
    assert G.get_edge_data(0, 1, default=default) == {0: {}}
    assert G.get_edge_data(0, 1, 0, default=default) == {}
    assert G.get_edge_data(0, 1, 1, default=default) is default
    assert G.get_edge_data(0, 1, "b", default=default) is default
    assert G.get_edge_data(-1, 2, default=default) is default
    assert G.has_edge(0, 1)
    assert G.has_edge(0, 1, 0)
    assert not G.has_edge(0, 1, 1)
    assert not G.has_edge(0, 1, "a")
