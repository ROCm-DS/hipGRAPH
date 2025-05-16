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


def test_generic_bfs_edges():
    # generic_bfs_edges currently isn't exercised by networkx tests
    Gnx = nx.karate_club_graph()
    Gcg = nx.karate_club_graph(backend="hipgraph")
    for depth_limit in (0, 1, 2):
        for source in Gnx:
            # Some ordering is arbitrary, so I think there's a chance
            # this test may fail if networkx or nx-hipgraph changes.
            nx_result = nx.generic_bfs_edges(Gnx, source, depth_limit=depth_limit)
            cg_result = nx.generic_bfs_edges(Gcg, source, depth_limit=depth_limit)
            assert sorted(nx_result) == sorted(cg_result), (source, depth_limit)
