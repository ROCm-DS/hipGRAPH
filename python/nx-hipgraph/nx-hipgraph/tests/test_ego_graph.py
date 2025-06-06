# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import networkx as nx
import nx_hipgraph as nxcg
import pytest
from packaging.version import parse

from .testing_utils import assert_graphs_equal

nxver = parse(nx.__version__)


if nxver.major == 3 and nxver.minor < 2:
    pytest.skip("Need NetworkX >=3.2 to test ego_graph", allow_module_level=True)


@pytest.mark.parametrize(
    "create_using", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
@pytest.mark.parametrize("radius", [-1, 0, 1, 1.5, 2, float("inf"), None])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("undirected", [False, True])
@pytest.mark.parametrize("multiple_edges", [False, True])
@pytest.mark.parametrize("n", [0, 3])
def test_ego_graph_cycle_graph(
    create_using, radius, center, undirected, multiple_edges, n
):
    Gnx = nx.cycle_graph(7, create_using=create_using)
    if multiple_edges:
        # Test multigraph with multiple edges
        if not Gnx.is_multigraph():
            return
        Gnx.add_edges_from(nx.cycle_graph(7, create_using=nx.DiGraph).edges)
        Gnx.add_edge(0, 1, 10)
    Gcg = nxcg.from_networkx(Gnx, preserve_all_attrs=True)
    assert_graphs_equal(Gnx, Gcg)  # Sanity check

    kwargs = {"radius": radius, "center": center, "undirected": undirected}
    Hnx = nx.ego_graph(Gnx, n, **kwargs)
    Hcg = nx.ego_graph(Gnx, n, **kwargs, backend="hipgraph")
    assert_graphs_equal(Hnx, Hcg)
    with pytest.raises(nx.NodeNotFound, match="not in G"):
        nx.ego_graph(Gnx, -1, **kwargs)
    with pytest.raises(nx.NodeNotFound, match="not in G"):
        nx.ego_graph(Gnx, -1, **kwargs, backend="hipgraph")
    # Using sssp with default weight of 1 should give same answer as bfs
    nx.set_edge_attributes(Gnx, 1, name="weight")
    Gcg = nxcg.from_networkx(Gnx, preserve_all_attrs=True)
    assert_graphs_equal(Gnx, Gcg)  # Sanity check

    kwargs["distance"] = "weight"
    H2nx = nx.ego_graph(Gnx, n, **kwargs)
    is_nx32 = nxver.major == 3 and nxver.minor == 2
    if undirected and Gnx.is_directed() and Gnx.is_multigraph():
        if is_nx32:
            # `should_run` was added in nx 3.3
            match = "Weighted ego_graph with undirected=True not implemented"
        else:
            match = "not implemented by hipgraph"
        with pytest.raises(RuntimeError, match=match):
            nx.ego_graph(Gnx, n, **kwargs, backend="hipgraph")
        with pytest.raises(NotImplementedError, match="ego_graph"):
            nx.ego_graph(Gcg, n, **kwargs)
    else:
        H2cg = nx.ego_graph(Gnx, n, **kwargs, backend="hipgraph")
        assert_graphs_equal(H2nx, H2cg)
        with pytest.raises(nx.NodeNotFound, match="not found in graph"):
            nx.ego_graph(Gnx, -1, **kwargs)
        with pytest.raises(nx.NodeNotFound, match="not found in graph"):
            nx.ego_graph(Gnx, -1, **kwargs, backend="hipgraph")
