# Copyright (c) 2022-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import hipgraph.experimental.compat.nx as nx
import pytest


@pytest.mark.sg
def test_connectivity():
    # Tests a run of a native nx algorithm that hasnt been overridden.
    expected = [{1, 2, 3, 4, 5}, {8, 9, 7}]
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])
    G.add_edges_from([(7, 8), (8, 9), (7, 9)])
    assert list(nx.connected_components(G)) == expected


@pytest.mark.sg
def test_pagerank_result_type():
    G = nx.DiGraph()
    [G.add_node(k) for k in ["A", "B", "C", "D", "E", "F", "G"]]
    G.add_edges_from(
        [
            ("G", "A"),
            ("A", "G"),
            ("B", "A"),
            ("C", "A"),
            ("A", "C"),
            ("A", "D"),
            ("E", "A"),
            ("F", "A"),
            ("D", "B"),
            ("D", "F"),
        ]
    )
    ppr1 = nx.pagerank(G)
    # This just tests that the right type is returned.
    assert isinstance(ppr1, dict)
