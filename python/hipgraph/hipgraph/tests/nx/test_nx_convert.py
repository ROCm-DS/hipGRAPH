# Copyright (c) 2019-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import cudf
import hipgraph
import networkx as nx
import pandas as pd
import pytest
from hipgraph.testing import DEFAULT_DATASETS, utils


def _compare_graphs(nxG, cuG, has_wt=True):
    assert nxG.number_of_nodes() == cuG.number_of_nodes()
    assert nxG.number_of_edges() == cuG.number_of_edges()

    cu_df = cuG.view_edge_list().to_pandas()
    cu_df = cu_df.rename(columns={"0": "src", "1": "dst"})
    if has_wt is True:
        cu_df = cu_df.drop(columns=["weight"])

    out_of_order = cu_df[cu_df["src"] > cu_df["dst"]]
    if len(out_of_order) > 0:
        out_of_order = out_of_order.rename(columns={"src": "dst", "dst": "src"})
        right_order = cu_df[cu_df["src"] < cu_df["dst"]]
        cu_df = pd.concat([out_of_order, right_order])
        del out_of_order
        del right_order
    cu_df = cu_df.sort_values(by=["src", "dst"]).reset_index(drop=True)

    nx_df = nx.to_pandas_edgelist(nxG)
    if has_wt is True:
        nx_df = nx_df.drop(columns=["weight"])
    nx_df = nx_df.rename(columns={"source": "src", "target": "dst"})
    nx_df = nx_df.astype("int32")

    out_of_order = nx_df[nx_df["src"] > nx_df["dst"]]
    if len(out_of_order) > 0:
        out_of_order = out_of_order.rename(columns={"src": "dst", "dst": "src"})
        right_order = nx_df[nx_df["src"] < nx_df["dst"]]

        nx_df = pd.concat([out_of_order, right_order])
        del out_of_order
        del right_order

    nx_df = nx_df.sort_values(by=["src", "dst"]).reset_index(drop=True)

    assert cu_df.to_dict() == nx_df.to_dict()


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DEFAULT_DATASETS)
def test_networkx_compatibility(graph_file):
    # test to make sure hipGRAPH and Nx build similar Graphs
    # Read in the graph
    dataset_path = graph_file.get_path()
    M = utils.read_csv_for_nx(dataset_path, read_weights_in_sp=True)

    # create a NetworkX DiGraph
    nxG = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.DiGraph()
    )

    # create a hipGRAPH Directed Graph
    gdf = cudf.from_pandas(M)
    cuG = hipgraph.from_cudf_edgelist(
        gdf,
        source="0",
        destination="1",
        edge_attr="weight",
        create_using=hipgraph.Graph(directed=True),
    )

    _compare_graphs(nxG, cuG)


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DEFAULT_DATASETS)
def test_nx_convert_undirected(graph_file):
    # read data and create a Nx Graph
    dataset_path = graph_file.get_path()
    nx_df = utils.read_csv_for_nx(dataset_path)
    nxG = nx.from_pandas_edgelist(nx_df, "0", "1", create_using=nx.Graph)
    assert nx.is_directed(nxG) is False
    assert nx.is_weighted(nxG) is False

    cuG = hipgraph.utilities.convert_from_nx(nxG)
    assert cuG.is_directed() is False
    assert cuG.is_weighted() is False

    _compare_graphs(nxG, cuG, has_wt=False)


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DEFAULT_DATASETS)
def test_nx_convert_directed(graph_file):
    # read data and create a Nx DiGraph
    dataset_path = graph_file.get_path()
    nx_df = utils.read_csv_for_nx(dataset_path)
    nxG = nx.from_pandas_edgelist(nx_df, "0", "1", create_using=nx.DiGraph)
    assert nxG.is_directed() is True

    cuG = hipgraph.utilities.convert_from_nx(nxG)
    assert cuG.is_directed() is True
    assert cuG.is_weighted() is False

    _compare_graphs(nxG, cuG, has_wt=False)


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DEFAULT_DATASETS)
def test_nx_convert_weighted(graph_file):
    # read data and create a Nx DiGraph
    dataset_path = graph_file.get_path()
    nx_df = utils.read_csv_for_nx(dataset_path, read_weights_in_sp=True)
    nxG = nx.from_pandas_edgelist(nx_df, "0", "1", "weight", create_using=nx.DiGraph)
    assert nx.is_directed(nxG) is True
    assert nx.is_weighted(nxG) is True

    cuG = hipgraph.utilities.convert_from_nx(nxG, weight="weight")
    assert hipgraph.is_directed(cuG) is True
    assert hipgraph.is_weighted(cuG) is True

    _compare_graphs(nxG, cuG, has_wt=True)


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DEFAULT_DATASETS)
def test_nx_convert_multicol(graph_file):
    # read data and create a Nx Graph
    dataset_path = graph_file.get_path()
    nx_df = utils.read_csv_for_nx(dataset_path)

    G = nx.DiGraph()

    for row in nx_df.iterrows():
        G.add_edge(row[1]["0"], row[1]["1"], count=[row[1]["0"], row[1]["1"]])

    nxG = nx.from_pandas_edgelist(nx_df, "0", "1")

    cuG = hipgraph.utilities.convert_from_nx(nxG)

    assert nxG.number_of_nodes() == cuG.number_of_nodes()
    assert nxG.number_of_edges() == cuG.number_of_edges()
