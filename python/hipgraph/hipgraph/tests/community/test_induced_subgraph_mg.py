# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc

import hipgraph
import hipgraph.dask as dcg
import pytest
from cudf.testing.testing import assert_frame_equal
from hipgraph.dask.common.mg_utils import is_single_gpu
from hipgraph.datasets import dolphins, email_Eu_core, karate

# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


# =============================================================================
# Parameters
# =============================================================================


DATASETS = [karate, dolphins, email_Eu_core]
IS_DIRECTED = [True, False]
NUM_VERTICES = [2, 5, 10, 20]
OFFSETS = [None]


# =============================================================================
# Helper functions
# =============================================================================


def get_sg_graph(dataset, directed):
    G = dataset.get_graph(create_using=hipgraph.Graph(directed=directed))

    return G


def get_mg_graph(dataset, directed):
    ddf = dataset.get_dask_edgelist()
    dg = hipgraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(
        ddf,
        source="src",
        destination="dst",
        edge_attr="wgt",
        renumber=True,
        store_transposed=True,
    )

    return dg


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.mg
@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("is_directed", IS_DIRECTED)
@pytest.mark.parametrize("num_vertices", NUM_VERTICES)
@pytest.mark.parametrize("offsets", OFFSETS)
def test_mg_induced_subgraph(
    dask_client, benchmark, dataset, is_directed, num_vertices, offsets
):
    # Create SG and MG Graphs
    g = get_sg_graph(dataset, is_directed)
    dg = get_mg_graph(dataset, is_directed)

    # Sample N random vertices to create the induced subgraph
    vertices = g.select_random_vertices(num_vertices=num_vertices)
    # print randomly sample n seeds from the graph
    print("\nvertices: \n", vertices)

    sg_induced_subgraph, _ = hipgraph.induced_subgraph(g, vertices=vertices)
    result_induced_subgraph = benchmark(
        dcg.induced_subgraph,
        dg,
        vertices,
        offsets,
    )

    # FIXME: This parameter is not yet tested
    # mg_offsets = mg_offsets.compute().reset_index(drop=True)
    mg_df, _ = result_induced_subgraph

    if mg_df is not None and sg_induced_subgraph is not None:
        # FIXME: 'edges()' or 'view_edgelist()' takes half the edges out if
        # 'directed=False'.
        sg_result = sg_induced_subgraph.input_df

        sg_df = sg_result.sort_values(["src", "dst"]).reset_index(drop=True)
        mg_df = mg_df.compute().sort_values(["src", "dst"]).reset_index(drop=True)

        assert_frame_equal(sg_df, mg_df, check_dtype=False, check_like=True)

    else:
        # There is no edges between the vertices provided
        # FIXME: Once k-hop neighbors is implemented, find one hop neighbors
        # of all the vertices and ensure that there is None
        assert sg_induced_subgraph is None
        assert mg_df is None
