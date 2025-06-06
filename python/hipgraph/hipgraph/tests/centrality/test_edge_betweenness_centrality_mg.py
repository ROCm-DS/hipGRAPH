# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc

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
IS_DIRECTED = [True, False]
IS_WEIGHTED = [True, False]
INCLUDE_EDGE_IDS = [True, False]
IS_NORMALIZED = [True, False]
SUBSET_SIZES = [4, None]


# =============================================================================
# Helper functions
# =============================================================================


def get_sg_graph(dataset, directed, edge_ids):
    df = dataset.get_edgelist()
    if edge_ids:
        if not directed:
            # Edge ids not supported for undirected graph
            return None

        dtype = df.dtypes.iloc[0]
        edge_id = "edge_id"
        df[edge_id] = df.index
        df = df.astype(dtype)

    else:
        edge_id = None

    G = hipgraph.Graph(directed=directed)
    G.from_cudf_edgelist(
        df, source="src", destination="dst", weight="wgt", edge_id=edge_id
    )

    return G


def get_mg_graph(dataset, directed, edge_ids, weight):
    ddf = dataset.get_dask_edgelist()

    if weight:
        weight = ddf
    else:
        weight = None

    if edge_ids:
        dtype = ddf.dtypes.iloc[0]
        edge_id = "edge_id"
        ddf = ddf.assign(idx=1)
        ddf["edge_id"] = ddf.idx.cumsum().astype(dtype) - 1
    else:
        edge_id = None

    dg = hipgraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(
        ddf,
        source="src",
        destination="dst",
        weight="wgt",
        edge_id=edge_id,
        renumber=True,
    )

    return dg, weight


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("directed", IS_DIRECTED)
@pytest.mark.parametrize("weighted", IS_WEIGHTED)
@pytest.mark.parametrize("edge_ids", INCLUDE_EDGE_IDS)
@pytest.mark.parametrize("normalized", IS_NORMALIZED)
@pytest.mark.parametrize("subset_size", SUBSET_SIZES)
def test_dask_mg_edge_betweenness_centrality(
    dask_client,
    dataset,
    directed,
    weighted,
    edge_ids,
    normalized,
    subset_size,
    benchmark,
):
    g = get_sg_graph(dataset, directed, edge_ids)

    if g is None:
        pytest.skip("Edge_ids not supported for undirected graph")

    dg, weight = get_mg_graph(dataset, directed, edge_ids, weighted)
    subset_seed = 42

    k = subset_size
    if isinstance(k, int):
        k = g.select_random_vertices(subset_seed, k)

    sg_hipgraph_edge_bc = (
        hipgraph.edge_betweenness_centrality(g, k, normalized)
        .sort_values(["src", "dst"])
        .reset_index(drop=True)
    )

    if weight is not None:
        with pytest.raises(NotImplementedError):
            result_edge_bc = benchmark(
                dcg.edge_betweenness_centrality, dg, k, normalized, weight=weight
            )

    else:
        result_edge_bc = benchmark(
            dcg.edge_betweenness_centrality, dg, k, normalized, weight=weight
        )
        result_edge_bc = (
            result_edge_bc.compute()
            .sort_values(["src", "dst"])
            .reset_index(drop=True)
            .rename(columns={"betweenness_centrality": "mg_betweenness_centrality"})
        )

        if len(result_edge_bc.columns) > 3:
            result_edge_bc = result_edge_bc.rename(columns={"edge_id": "mg_edge_id"})

        expected_output = sg_hipgraph_edge_bc.reset_index(drop=True)
        result_edge_bc["betweenness_centrality"] = expected_output[
            "betweenness_centrality"
        ]
        if len(expected_output.columns) > 3:
            result_edge_bc["edge_id"] = expected_output["edge_id"]
            edge_id_diff = result_edge_bc.query("mg_edge_id != edge_id")
            assert len(edge_id_diff) == 0

        edge_bc_diffs1 = result_edge_bc.query(
            "mg_betweenness_centrality - betweenness_centrality > 0.01"
        )
        edge_bc_diffs2 = result_edge_bc.query(
            "betweenness_centrality - mg_betweenness_centrality < -0.01"
        )

        assert len(edge_bc_diffs1) == 0
        assert len(edge_bc_diffs2) == 0
