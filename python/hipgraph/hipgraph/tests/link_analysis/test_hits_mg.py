# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc

import dask_cudf
import hipgraph
import hipgraph.dask as dcg
import pytest
from hipgraph.testing import utils
from pylibhipgraph.testing.utils import gen_fixture_params_product

# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


IS_DIRECTED = [True, False]


# =============================================================================
# Pytest fixtures
# =============================================================================

datasets = utils.DATASETS_UNDIRECTED + [
    utils.RAPIDS_DATASET_ROOT_DIR_PATH / "email-Eu-core.csv"
]

fixture_params = gen_fixture_params_product(
    (datasets, "graph_file"),
    ([50], "max_iter"),
    ([1.0e-4], "tol"),  # FIXME: Temporarily lower tolerance
    (IS_DIRECTED, "directed"),
)


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    parameters = dict(zip(("graph_file", "max_iter", "tol", "directed"), request.param))

    return parameters


@pytest.fixture(scope="module")
def input_expected_output(input_combo):
    """
    This fixture returns the inputs and expected results from the HITS algo.
    (based on hipGRAPH HITS) which can be used for validation.
    """

    input_data_path = input_combo["graph_file"]
    directed = input_combo["directed"]
    G = utils.generate_hipgraph_graph_from_file(input_data_path, directed=directed)
    sg_hipgraph_hits = hipgraph.hits(G, input_combo["max_iter"], input_combo["tol"])
    # Save the results back to the input_combo dictionary to prevent redundant
    # hipGRAPH runs. Other tests using the input_combo fixture will look for
    # them, and if not present they will have to re-run the same hipGRAPH call.
    sg_hipgraph_hits = sg_hipgraph_hits.sort_values("vertex").reset_index(drop=True)

    input_combo["sg_hipgraph_results"] = sg_hipgraph_hits
    chunksize = dcg.get_chunksize(input_data_path)
    ddf = dask_cudf.read_csv(
        input_data_path,
        blocksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    dg = hipgraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(
        ddf,
        source="src",
        destination="dst",
        edge_attr="value",
        renumber=True,
        store_transposed=True,
    )

    input_combo["MGGraph"] = dg

    return input_combo


# =============================================================================
# Tests
# =============================================================================


# @pytest.mark.skipif(
#    is_single_gpu(), reason="skipping MG testing on Single GPU system"
# )
@pytest.mark.mg
def test_dask_mg_hits(dask_client, benchmark, input_expected_output):

    dg = input_expected_output["MGGraph"]

    result_hits = benchmark(
        dcg.hits, dg, input_expected_output["tol"], input_expected_output["max_iter"]
    )

    result_hits = (
        result_hits.compute()
        .sort_values("vertex")
        .reset_index(drop=True)
        .rename(
            columns={
                "hubs": "mg_hipgraph_hubs",
                "authorities": "mg_hipgraph_authorities",
            }
        )
    )

    expected_output = (
        input_expected_output["sg_hipgraph_results"]
        .sort_values("vertex")
        .reset_index(drop=True)
    )

    # Update the dask hipgraph HITS results with sg hipgraph results for easy
    # comparison using cuDF DataFrame methods.
    result_hits["sg_hipgraph_hubs"] = expected_output["hubs"]
    result_hits["sg_hipgraph_authorities"] = expected_output["authorities"]

    hubs_diffs1 = result_hits.query("mg_hipgraph_hubs - sg_hipgraph_hubs > 0.00001")
    hubs_diffs2 = result_hits.query("mg_hipgraph_hubs - sg_hipgraph_hubs < -0.00001")
    authorities_diffs1 = result_hits.query(
        "mg_hipgraph_authorities - sg_hipgraph_authorities > 0.0001"
    )
    authorities_diffs2 = result_hits.query(
        "mg_hipgraph_authorities - sg_hipgraph_authorities < -0.0001"
    )

    assert len(hubs_diffs1) == 0
    assert len(hubs_diffs2) == 0
    assert len(authorities_diffs1) == 0
    assert len(authorities_diffs2) == 0


@pytest.mark.mg
def test_dask_mg_hits_transposed_false(dask_client):
    input_data_path = (utils.RAPIDS_DATASET_ROOT_DIR_PATH / "karate.csv").as_posix()

    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(
        input_data_path,
        blocksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    dg = hipgraph.Graph(directed=True)
    dg.from_dask_cudf_edgelist(ddf, "src", "dst", store_transposed=False)

    warning_msg = (
        "HITS expects the 'store_transposed' "
        "flag to be set to 'True' for optimal performance during "
        "the graph creation"
    )

    with pytest.warns(UserWarning, match=warning_msg):
        dcg.hits(dg)
