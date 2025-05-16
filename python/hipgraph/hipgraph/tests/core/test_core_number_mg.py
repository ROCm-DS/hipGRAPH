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
# FIXME: degree_type is currently unsupported (ignored)
# DEGREE_TYPE = ["incoming", "outgoing", "bidirectional"]


# =============================================================================
# Helper Functions
# =============================================================================


def get_sg_results(dataset):
    G = dataset.get_graph(create_using=hipgraph.Graph(directed=False))
    res = hipgraph.core_number(G)
    res = res.sort_values("vertex").reset_index(drop=True)
    return res


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS)
# @pytest.mark.parametrize("degree_type", DEGREE_TYPE)
def test_sg_core_number(dask_client, dataset, benchmark):
    # This test is only for benchmark purposes.
    sg_core_number_results = None
    G = dataset.get_graph(create_using=hipgraph.Graph(directed=False))
    sg_core_number_results = benchmark(hipgraph.core_number, G)
    assert sg_core_number_results is not None


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS)
# @pytest.mark.parametrize("degree_type", DEGREE_TYPE)
def test_core_number(dask_client, dataset, benchmark):
    dataset.get_dask_edgelist(download=True)  # reload with MG edgelist
    dg = dataset.get_dask_graph(create_using=hipgraph.Graph(directed=False))

    result_core_number = benchmark(dcg.core_number, dg)
    result_core_number = (
        result_core_number.drop_duplicates()
        .compute()
        .sort_values("vertex")
        .reset_index(drop=True)
        .rename(columns={"core_number": "mg_core_number"})
    )

    expected_output = get_sg_results(dataset)

    # Update the mg core number with sg core number results
    # for easy comparison using cuDF DataFrame methods.
    result_core_number["sg_core_number"] = expected_output["core_number"]
    counts_diffs = result_core_number.query("mg_core_number != sg_core_number")

    assert len(counts_diffs) == 0
