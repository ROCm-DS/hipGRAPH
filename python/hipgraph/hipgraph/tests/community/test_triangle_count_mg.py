# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc
import random

import hipgraph
import hipgraph.dask as dcg
import pytest
from hipgraph.datasets import dolphins, karate

# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


# =============================================================================
# Parameters
# =============================================================================


DATASETS = [karate, dolphins]
START_LIST = [True, False]


# =============================================================================
# Helper Functions
# =============================================================================


def get_sg_graph(dataset, directed, start):
    G = dataset.get_graph(create_using=hipgraph.Graph(directed=directed))
    if start:
        # sample k nodes from the hipGRAPH graph
        start = G.select_random_vertices(num_vertices=random.randint(1, 10))
    else:
        start = None

    return G, start


def get_mg_graph(dataset, directed):
    ddf = dataset.get_dask_edgelist()
    dg = hipgraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(
        ddf, source="src", destination="dst", edge_attr="wgt", renumber=True
    )

    return dg


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("start", START_LIST)
def test_sg_triangles(dask_client, dataset, start, benchmark):
    # This test is only for benchmark purposes.
    sg_triangle_results = None
    G, start = get_sg_graph(dataset, False, start)

    sg_triangle_results = benchmark(hipgraph.triangle_count, G, start)
    sg_triangle_results.sort_values("vertex").reset_index(drop=True)
    assert sg_triangle_results is not None


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("start", START_LIST)
def test_triangles(dask_client, dataset, start, benchmark):
    G, start = get_sg_graph(dataset, False, start)
    dg = get_mg_graph(dataset, False)

    result_counts = benchmark(dcg.triangle_count, dg, start)
    result_counts = (
        result_counts.drop_duplicates()
        .compute()
        .sort_values("vertex")
        .reset_index(drop=True)
        .rename(columns={"counts": "mg_counts"})
    )
    expected_output = (
        hipgraph.triangle_count(G, start).sort_values("vertex").reset_index(drop=True)
    )

    # Update the mg triangle count with sg triangle count results
    # for easy comparison using cuDF DataFrame methods.
    result_counts["sg_counts"] = expected_output["counts"]
    counts_diffs = result_counts.query("mg_counts != sg_counts")

    assert len(counts_diffs) == 0
