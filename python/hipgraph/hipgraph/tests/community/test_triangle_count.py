# Copyright (c) 2019-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc
import random

import cudf
import hipgraph
import networkx as nx
import pytest
from hipgraph.datasets import karate_asymmetric
from hipgraph.testing import UNDIRECTED_DATASETS, utils
from pylibhipgraph.testing.utils import gen_fixture_params_product


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


# =============================================================================
# Pytest fixtures
# =============================================================================
datasets = UNDIRECTED_DATASETS
fixture_params = gen_fixture_params_product(
    (datasets, "graph_file"),
    ([True, False], "edgevals"),
    ([True, False], "start_list"),
)


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    This fixture returns a dictionary containing all input params required to
    run a Triangle Count algo
    """
    parameters = dict(zip(("graph_file", "edgevals", "start_list"), request.param))

    graph_file = parameters["graph_file"]
    input_data_path = graph_file.get_path()
    edgevals = parameters["edgevals"]

    G = graph_file.get_graph(ignore_weights=not edgevals)

    Gnx = utils.generate_nx_graph_from_file(
        input_data_path, directed=False, edgevals=edgevals
    )

    parameters["G"] = G
    parameters["Gnx"] = Gnx

    return parameters


# =============================================================================
# Tests
# =============================================================================
@pytest.mark.sg
def test_triangles(input_combo):
    G = input_combo["G"]
    Gnx = input_combo["Gnx"]
    nx_triangle_results = cudf.DataFrame()

    if input_combo["start_list"]:
        # sample k nodes from the nx graph
        k = random.randint(1, 10)
        start_list = random.sample(list(Gnx.nodes()), k)
    else:
        start_list = None

    hipgraph_triangle_results = hipgraph.triangle_count(G, start_list)

    triangle_results = (
        hipgraph_triangle_results.sort_values("vertex")
        .reset_index(drop=True)
        .rename(columns={"counts": "hipgraph_counts"})
    )

    dic_results = nx.triangles(Gnx, start_list)
    nx_triangle_results["vertex"] = dic_results.keys()
    nx_triangle_results["counts"] = dic_results.values()
    nx_triangle_results = nx_triangle_results.sort_values("vertex").reset_index(
        drop=True
    )

    triangle_results["nx_counts"] = nx_triangle_results["counts"]
    counts_diff = triangle_results.query("nx_counts != hipgraph_counts")
    assert len(counts_diff) == 0


@pytest.mark.sg
def test_triangles_int64(input_combo):
    Gnx = input_combo["Gnx"]
    count_int32 = hipgraph.triangle_count(Gnx)["counts"].sum()

    graph_file = input_combo["graph_file"]
    G = graph_file.get_graph()
    G.edgelist.edgelist_df = G.edgelist.edgelist_df.astype(
        {"src": "int64", "dst": "int64"}
    )
    count_int64 = hipgraph.triangle_count(G)["counts"].sum()

    assert G.edgelist.edgelist_df["src"].dtype == "int64"
    assert G.edgelist.edgelist_df["dst"].dtype == "int64"
    assert count_int32 == count_int64


@pytest.mark.sg
def test_triangles_no_weights(input_combo):
    G_weighted = input_combo["Gnx"]
    count_triangles_nx_graph = hipgraph.triangle_count(G_weighted)["counts"].sum()

    graph_file = input_combo["graph_file"]
    G = graph_file.get_graph(ignore_weights=True)

    assert G.is_weighted() is False
    count_triangles = hipgraph.triangle_count(G)["counts"].sum()

    assert count_triangles_nx_graph == count_triangles


@pytest.mark.sg
def test_triangles_directed_graph():
    input_data_path = karate_asymmetric.get_path()
    M = utils.read_csv_for_nx(input_data_path)
    G = hipgraph.Graph(directed=True)
    cu_M = cudf.DataFrame()
    cu_M["src"] = cudf.Series(M["0"])
    cu_M["dst"] = cudf.Series(M["1"])

    cu_M["weights"] = cudf.Series(M["weight"])
    G.from_cudf_edgelist(cu_M, source="src", destination="dst", edge_attr="weights")

    with pytest.raises(ValueError):
        hipgraph.triangle_count(G)
