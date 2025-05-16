# Copyright (c) 2018-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc

import hipgraph
import pytest
from cudf.testing import assert_series_equal
from hipgraph.dask.common.mg_utils import is_single_gpu
from hipgraph.datasets import email_Eu_core, karate_asymmetric, polbooks

# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


# =============================================================================
# Parameters
# =============================================================================


DATASETS = [karate_asymmetric, polbooks, email_Eu_core]
IS_DIRECTED = [True, False]


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
@pytest.mark.parametrize("directed", IS_DIRECTED)
def test_dask_mg_degree(dask_client, dataset, directed):
    dg = get_mg_graph(dataset, directed)
    dg.compute_renumber_edge_list()

    g = get_sg_graph(dataset, directed)

    merge_df_in_degree = (
        dg.in_degree()
        .merge(g.in_degree(), on="vertex", suffixes=["_dg", "_g"])
        .compute()
    )

    merge_df_out_degree = (
        dg.out_degree()
        .merge(g.out_degree(), on="vertex", suffixes=["_dg", "_g"])
        .compute()
    )

    merge_df_degree = (
        dg.degree().merge(g.degree(), on="vertex", suffixes=["_dg", "_g"]).compute()
    )

    assert_series_equal(
        merge_df_in_degree["degree_dg"],
        merge_df_in_degree["degree_g"],
        check_names=False,
        check_dtype=False,
    )

    assert_series_equal(
        merge_df_out_degree["degree_dg"],
        merge_df_out_degree["degree_g"],
        check_names=False,
        check_dtype=False,
    )

    assert_series_equal(
        merge_df_degree["degree_dg"],
        merge_df_degree["degree_g"],
        check_names=False,
        check_dtype=False,
    )
