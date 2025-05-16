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
from cudf.testing.testing import assert_frame_equal, assert_series_equal
from hipgraph.dask.common.mg_utils import is_single_gpu
from hipgraph.testing import utils
from pylibhipgraph.testing import gen_fixture_params_product

# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


IS_DIRECTED = [True, False]
SEEDS = [0, 5, 13, [0, 2]]
RADIUS = [1, 2, 3]


# =============================================================================
# Pytest fixtures
# =============================================================================

datasets = utils.DATASETS_UNDIRECTED + [
    utils.RAPIDS_DATASET_ROOT_DIR_PATH / "email-Eu-core.csv"
]

fixture_params = gen_fixture_params_product(
    (datasets, "graph_file"),
    (IS_DIRECTED, "directed"),
    (SEEDS, "seeds"),
    (RADIUS, "radius"),
)


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    parameters = dict(zip(("graph_file", "directed", "seeds", "radius"), request.param))

    return parameters


@pytest.fixture(scope="module")
def input_expected_output(input_combo):
    """
    This fixture returns the inputs and expected results from the egonet algo.
    (based on hipGRAPH batched_ego_graphs) which can be used for validation.
    """

    input_data_path = input_combo["graph_file"]
    directed = input_combo["directed"]
    seeds = input_combo["seeds"]
    radius = input_combo["radius"]
    G = utils.generate_hipgraph_graph_from_file(
        input_data_path, directed=directed, edgevals=True
    )

    sg_hipgraph_ego_graphs = hipgraph.batched_ego_graphs(G, seeds=seeds, radius=radius)

    # Save the results back to the input_combo dictionary to prevent redundant
    # hipGRAPH runs. Other tests using the input_combo fixture will look for
    # them, and if not present they will have to re-run the same hipGRAPH call.

    input_combo["sg_hipgraph_results"] = sg_hipgraph_ego_graphs
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


@pytest.mark.mg
@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
def test_dask_mg_ego_graphs(dask_client, benchmark, input_expected_output):

    dg = input_expected_output["MGGraph"]

    result_ego_graph = benchmark(
        dcg.ego_graph,
        dg,
        input_expected_output["seeds"],
        input_expected_output["radius"],
    )

    mg_df, mg_offsets = result_ego_graph

    mg_df = mg_df.compute()
    mg_offsets = mg_offsets.compute().reset_index(drop=True)

    sg_df, sg_offsets = input_expected_output["sg_hipgraph_results"]

    assert_series_equal(sg_offsets, mg_offsets, check_dtype=False)
    # slice array from offsets, sort the df by src dst and compare
    for i in range(len(sg_offsets) - 1):
        start = sg_offsets[i]
        end = sg_offsets[i + 1]
        mg_df_part = mg_df[start:end].sort_values(["src", "dst"]).reset_index(drop=True)
        sg_df_part = sg_df[start:end].sort_values(["src", "dst"]).reset_index(drop=True)

        assert_frame_equal(mg_df_part, sg_df_part, check_dtype=False, check_like=True)
