# Copyright (c) 2020-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc

import hipgraph
import networkx as nx
import numpy as np
import pytest
from hipgraph.testing import DEFAULT_DATASETS, utils


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DEFAULT_DATASETS)
def test_multigraph(graph_file):
    # FIXME: Migrate to new test fixtures for Graph setup once available
    G = graph_file.get_graph(create_using=hipgraph.MultiGraph(directed=True))
    dataset_path = graph_file.get_path()
    nxM = utils.read_csv_for_nx(dataset_path, read_weights_in_sp=True)
    Gnx = nx.from_pandas_edgelist(
        nxM,
        source="0",
        target="1",
        edge_attr="weight",
        create_using=nx.MultiDiGraph(),
    )

    assert G.number_of_edges() == Gnx.number_of_edges()
    assert G.number_of_nodes() == Gnx.number_of_nodes()
    cuedges = hipgraph.to_pandas_edgelist(G)
    cuedges.rename(
        columns={"src": "source", "dst": "target", "wgt": "weight"}, inplace=True
    )
    cuedges["weight"] = cuedges["weight"].round(decimals=3)
    nxedges = nx.to_pandas_edgelist(Gnx).astype(
        dtype={"source": "int32", "target": "int32", "weight": "float32"}
    )
    cuedges = cuedges.sort_values(by=["source", "target"]).reset_index(drop=True)
    nxedges = nxedges.sort_values(by=["source", "target"]).reset_index(drop=True)
    nxedges["weight"] = nxedges["weight"].round(decimals=3)
    assert nxedges.equals(cuedges[["source", "target", "weight"]])


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DEFAULT_DATASETS)
def test_Graph_from_MultiGraph(graph_file):
    # FIXME: Migrate to new test fixtures for Graph setup once available
    GM = graph_file.get_graph(create_using=hipgraph.MultiGraph())
    dataset_path = graph_file.get_path()
    nxM = utils.read_csv_for_nx(dataset_path, read_weights_in_sp=True)
    GnxM = nx.from_pandas_edgelist(
        nxM,
        source="0",
        target="1",
        edge_attr="weight",
        create_using=nx.MultiGraph(),
    )

    G = hipgraph.Graph(GM)
    Gnx = nx.Graph(GnxM)
    assert Gnx.number_of_edges() == G.number_of_edges(directed_edges=True)
    GdM = graph_file.get_graph(create_using=hipgraph.MultiGraph(directed=True))
    GnxdM = nx.from_pandas_edgelist(
        nxM,
        source="0",
        target="1",
        edge_attr="weight",
        create_using=nx.MultiGraph(),
    )
    Gd = hipgraph.Graph(GdM, directed=True)
    Gnxd = nx.DiGraph(GnxdM)
    assert Gnxd.number_of_edges() == Gd.number_of_edges()


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DEFAULT_DATASETS)
def test_multigraph_sssp(graph_file):
    # FIXME: Migrate to new test fixtures for Graph setup once available
    G = graph_file.get_graph(create_using=hipgraph.MultiGraph(directed=True))
    cu_paths = hipgraph.sssp(G, 0)
    max_val = np.finfo(cu_paths["distance"].dtype).max
    cu_paths = cu_paths[cu_paths["distance"] != max_val]
    dataset_path = graph_file.get_path()
    nxM = utils.read_csv_for_nx(dataset_path, read_weights_in_sp=True)
    Gnx = nx.from_pandas_edgelist(
        nxM,
        source="0",
        target="1",
        edge_attr="weight",
        create_using=nx.MultiDiGraph(),
    )
    nx_paths = nx.single_source_dijkstra_path_length(Gnx, 0)

    cu_dist = cu_paths.sort_values(by="vertex")["distance"].to_numpy()
    nx_dist = [i[1] for i in sorted(nx_paths.items())]

    assert (cu_dist == nx_dist).all()
