# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc
import random

import dask_cudf
import hipgraph
import hipgraph.dask as dcg
import pytest
from hipgraph.testing import utils
from pylibhipgraph.testing import gen_fixture_params_product

# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


IS_DIRECTED = [False]
HAS_VERTEX_PAIR = [False, True]
HAS_VERTICES = [False, True]
HAS_TOPK = [False, True]
IS_WEIGHTED = [False, True]


# =============================================================================
# Pytest fixtures
# =============================================================================

datasets = utils.DATASETS_UNDIRECTED + [
    utils.RAPIDS_DATASET_ROOT_DIR_PATH / "email-Eu-core.csv"
]

fixture_params = gen_fixture_params_product(
    (datasets, "graph_file"),
    (IS_DIRECTED, "directed"),
    (HAS_VERTEX_PAIR, "has_vertex_pair"),
    (HAS_VERTICES, "has_vertices"),
    (HAS_TOPK, "has_topk"),
    (IS_WEIGHTED, "is_weighted"),
)


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    parameters = dict(
        zip(
            (
                "graph_file",
                "directed",
                "has_vertex_pair",
                "has_vertices",
                "has_topk",
                "is_weighted",
            ),
            request.param,
        )
    )

    return parameters


@pytest.fixture(scope="module")
def input_expected_output(input_combo):
    """
    This fixture returns the inputs and expected results from the Jaccard algo.
    (based on hipGRAPH Jaccard) which can be used for validation.
    """

    input_data_path = input_combo["graph_file"]
    directed = input_combo["directed"]
    has_vertex_pair = input_combo["has_vertex_pair"]
    is_weighted = input_combo["is_weighted"]
    G = utils.generate_hipgraph_graph_from_file(
        input_data_path, directed=directed, edgevals=is_weighted
    )
    if has_vertex_pair:
        # Sample random vertices from the graph and compute the two_hop_neighbors
        # with those seeds
        k = random.randint(1, 10)
        seeds = random.sample(range(G.number_of_vertices()), k)

        vertex_pair = G.get_two_hop_neighbors(start_vertices=seeds)
    else:
        vertex_pair = None

    input_combo["vertex_pair"] = vertex_pair
    sg_hipgraph_jaccard = hipgraph.jaccard(
        G, input_combo["vertex_pair"], use_weight=is_weighted
    )
    # Save the results back to the input_combo dictionary to prevent redundant
    # hipGRAPH runs. Other tests using the input_combo fixture will look for
    # them, and if not present they will have to re-run the same hipGRAPH call.

    input_combo["sg_hipgraph_results"] = sg_hipgraph_jaccard
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
        edge_attr="value" if is_weighted else None,
        renumber=True,
        store_transposed=True,
    )

    input_combo["MGGraph"] = dg

    return input_combo


@pytest.fixture(scope="module")
def input_expected_output_all_pairs(input_combo):
    """
    This fixture returns the inputs and expected results from the Jaccard algo.
    (based on hipGRAPH Jaccard) which can be used for validation.
    """

    input_data_path = input_combo["graph_file"]
    directed = input_combo["directed"]
    has_vertices = input_combo["has_vertices"]
    has_topk = input_combo["has_topk"]
    is_weighted = input_combo["is_weighted"]
    G = utils.generate_hipgraph_graph_from_file(
        input_data_path, directed=directed, edgevals=is_weighted
    )

    if has_topk:
        topk = 5
    else:
        topk = None

    if has_vertices:
        # Sample random vertices from the graph and compute the two_hop_neighbors
        # with those seeds
        k = random.randint(1, 10)
        vertices = random.sample(range(G.number_of_vertices()), k)

    else:
        vertices = None
        # If no start_vertices are passed, all_pairs similarity runs OOM
        topk = 10

    input_combo["vertices"] = vertices
    input_combo["topk"] = topk
    print("vertices ", vertices)
    sg_hipgraph_all_pairs_jaccard = hipgraph.all_pairs_jaccard(
        G,
        vertices=input_combo["vertices"],
        topk=input_combo["topk"],
        use_weight=is_weighted,
    )
    # Save the results back to the input_combo dictionary to prevent redundant
    # hipGRAPH runs. Other tests using the input_combo fixture will look for
    # them, and if not present they will have to re-run the same hipGRAPH call.

    input_combo["sg_hipgraph_results"] = sg_hipgraph_all_pairs_jaccard
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
        edge_attr="value" if is_weighted else None,
        renumber=True,
        store_transposed=True,
    )

    input_combo["MGGraph"] = dg

    return input_combo


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.mg
def test_dask_mg_jaccard(dask_client, benchmark, input_expected_output):

    dg = input_expected_output["MGGraph"]
    use_weight = input_expected_output["is_weighted"]

    result_jaccard = benchmark(
        dcg.jaccard, dg, input_expected_output["vertex_pair"], use_weight=use_weight
    )

    result_jaccard = (
        result_jaccard.compute()
        .sort_values(["first", "second"])
        .reset_index(drop=True)
        .rename(columns={"jaccard_coeff": "mg_hipgraph_jaccard_coeff"})
    )

    expected_output = (
        input_expected_output["sg_hipgraph_results"]
        .sort_values(["first", "second"])
        .reset_index(drop=True)
    )

    # Update the dask hipgraph Jaccard results with sg hipgraph results for easy
    # comparison using cuDF DataFrame methods.
    result_jaccard["sg_hipgraph_jaccard_coeff"] = expected_output["jaccard_coeff"]

    jaccard_coeff_diffs1 = result_jaccard.query(
        "mg_hipgraph_jaccard_coeff - sg_hipgraph_jaccard_coeff > 0.00001"
    )
    jaccard_coeff_diffs2 = result_jaccard.query(
        "mg_hipgraph_jaccard_coeff - sg_hipgraph_jaccard_coeff < -0.00001"
    )

    assert len(jaccard_coeff_diffs1) == 0
    assert len(jaccard_coeff_diffs2) == 0


@pytest.mark.mg
def test_dask_mg_all_pairs_jaccard(
    dask_client, benchmark, input_expected_output_all_pairs
):

    dg = input_expected_output_all_pairs["MGGraph"]

    use_weight = input_expected_output_all_pairs["is_weighted"]

    result_jaccard = benchmark(
        dcg.all_pairs_jaccard,
        dg,
        vertices=input_expected_output_all_pairs["vertices"],
        topk=input_expected_output_all_pairs["topk"],
        use_weight=use_weight,
    )

    result_jaccard = (
        result_jaccard.compute()
        .sort_values(["first", "second"])
        .reset_index(drop=True)
        .rename(columns={"jaccard_coeff": "mg_hipgraph_jaccard_coeff"})
    )

    expected_output = (
        input_expected_output_all_pairs["sg_hipgraph_results"]
        .sort_values(["first", "second"])
        .reset_index(drop=True)
    )

    # Update the dask hipgraph Jaccard results with sg hipgraph results for easy
    # comparison using cuDF DataFrame methods.
    result_jaccard["sg_hipgraph_jaccard_coeff"] = expected_output["jaccard_coeff"]

    jaccard_coeff_diffs1 = result_jaccard.query(
        "mg_hipgraph_jaccard_coeff - sg_hipgraph_jaccard_coeff > 0.00001"
    )
    jaccard_coeff_diffs2 = result_jaccard.query(
        "mg_hipgraph_jaccard_coeff - sg_hipgraph_jaccard_coeff < -0.00001"
    )

    assert len(jaccard_coeff_diffs1) == 0
    assert len(jaccard_coeff_diffs2) == 0
