# Copyright (c) 2019-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc

import cudf
import hipgraph
import networkx as nx
import pytest
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
# FIXME: degree_type is currently unsupported (ignored)
# degree_type = ["incoming", "outgoing"]

# fixture_params = gen_fixture_params_product(
#     (UNDIRECTED_DATASETS, "graph_file"),
#     (degree_type, "degree_type"),
# )
fixture_params = gen_fixture_params_product(
    (UNDIRECTED_DATASETS, "graph_file"),
)


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    This fixture returns a dictionary containing all input params required to
    run a Core number algo
    """
    # FIXME: degree_type is not supported so do not test with different values
    # parameters = dict(zip(("graph_file", "degree_type"), request.param))
    parameters = {"graph_file": request.param[0]}

    graph_file = parameters["graph_file"]
    G = graph_file.get_graph()
    input_data_path = graph_file.get_path()

    Gnx = utils.generate_nx_graph_from_file(
        input_data_path, directed=False, edgevals=True
    )

    parameters["G"] = G
    parameters["Gnx"] = Gnx

    return parameters


# =============================================================================
# Tests
# =============================================================================
@pytest.mark.sg
def test_core_number(input_combo):
    G = input_combo["G"]
    Gnx = input_combo["Gnx"]
    # FIXME: degree_type is currently unsupported (ignored)
    # degree_type = input_combo["degree_type"]
    nx_core_number_results = cudf.DataFrame()

    dic_results = nx.core_number(Gnx)
    nx_core_number_results["vertex"] = dic_results.keys()
    nx_core_number_results["core_number"] = dic_results.values()
    nx_core_number_results = nx_core_number_results.sort_values("vertex").reset_index(
        drop=True
    )

    core_number_results = (
        hipgraph.core_number(G)
        .sort_values("vertex")
        .reset_index(drop=True)
        .rename(columns={"core_number": "hipgraph_core_number"})
    )

    # Compare the nx core number results with hipgraph
    core_number_results["nx_core_number"] = nx_core_number_results["core_number"]

    counts_diff = core_number_results.query("nx_core_number != hipgraph_core_number")
    assert len(counts_diff) == 0


@pytest.mark.sg
def test_core_number_invalid_input(input_combo):
    input_data_path = (
        utils.RAPIDS_DATASET_ROOT_DIR_PATH / "karate-asymmetric.csv"
    ).as_posix()
    M = utils.read_csv_for_nx(input_data_path)
    G = hipgraph.Graph(directed=True)
    cu_M = cudf.DataFrame()
    cu_M["src"] = cudf.Series(M["0"])
    cu_M["dst"] = cudf.Series(M["1"])

    cu_M["weights"] = cudf.Series(M["weight"])
    G.from_cudf_edgelist(cu_M, source="src", destination="dst", edge_attr="weights")

    with pytest.raises(ValueError):
        hipgraph.core_number(G)
