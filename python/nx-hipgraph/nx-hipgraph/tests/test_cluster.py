# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import networkx as nx
import pytest
from packaging.version import parse

nxver = parse(nx.__version__)

if nxver.major == 3 and nxver.minor < 2:
    pytest.skip("Need NetworkX >=3.2 to test clustering", allow_module_level=True)


def test_selfloops():
    G = nx.complete_graph(5)
    H = nx.complete_graph(5)
    H.add_edge(0, 0)
    H.add_edge(1, 1)
    H.add_edge(2, 2)
    # triangles
    expected = nx.triangles(G)
    assert expected == nx.triangles(H)
    assert expected == nx.triangles(G, backend="hipgraph")
    assert expected == nx.triangles(H, backend="hipgraph")
    # average_clustering
    expected = nx.average_clustering(G)
    assert expected == nx.average_clustering(H)
    assert expected == nx.average_clustering(G, backend="hipgraph")
    assert expected == nx.average_clustering(H, backend="hipgraph")
    # clustering
    expected = nx.clustering(G)
    assert expected == nx.clustering(H)
    assert expected == nx.clustering(G, backend="hipgraph")
    assert expected == nx.clustering(H, backend="hipgraph")
    # transitivity
    expected = nx.transitivity(G)
    assert expected == nx.transitivity(H)
    assert expected == nx.transitivity(G, backend="hipgraph")
    assert expected == nx.transitivity(H, backend="hipgraph")
