# Copyright (c) 2020-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc

import cudf
import hipgraph
import hipgraph.dask as dcg
import pytest
from hipgraph.dask.common.mg_utils import is_single_gpu
from hipgraph.datasets import karate

# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


# =============================================================================
# Parameters
# =============================================================================


DATASETS = [karate]
IS_DIRECTED = [True, False]


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.mg
@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("directed", IS_DIRECTED)
def test_dask_mg_katz_centrality(dask_client, dataset, directed):
    input_data_path = dataset.get_path()
    print(f"dataset={input_data_path}")

    ddf = dataset.get_dask_edgelist()
    dg = hipgraph.Graph(directed=True)
    dg.from_dask_cudf_edgelist(ddf, "src", "dst", store_transposed=True)

    degree_max = dg.degree()["degree"].max().compute()
    katz_alpha = 1 / (degree_max)

    mg_res = dcg.katz_centrality(dg, alpha=katz_alpha, tol=1e-6)
    mg_res = mg_res.compute()

    import networkx as nx
    from hipgraph.testing import utils

    NM = utils.read_csv_for_nx(input_data_path)
    if directed:
        Gnx = nx.from_pandas_edgelist(
            NM, create_using=nx.DiGraph(), source="0", target="1"
        )
    else:
        Gnx = nx.from_pandas_edgelist(
            NM, create_using=nx.Graph(), source="0", target="1"
        )
    nk = nx.katz_centrality(Gnx, alpha=katz_alpha)
    import pandas as pd

    pdf = pd.DataFrame(nk.items(), columns=["vertex", "katz_centrality"])
    exp_res = cudf.DataFrame(pdf)
    err = 0
    tol = 1.0e-05

    compare_res = exp_res.merge(mg_res, on="vertex", suffixes=["_local", "_dask"])

    for i in range(len(compare_res)):
        diff = abs(
            compare_res["katz_centrality_local"].iloc[i]
            - compare_res["katz_centrality_dask"].iloc[i]
        )
        if diff > tol * 1.1:
            err = err + 1
    assert err == 0


@pytest.mark.mg
@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("directed", IS_DIRECTED)
def test_dask_mg_katz_centrality_nstart(dask_client, dataset, directed):
    ddf = dataset.get_dask_edgelist()
    dg = hipgraph.Graph(directed=True)
    dg.from_dask_cudf_edgelist(ddf, "src", "dst", store_transposed=True)

    mg_res = dcg.katz_centrality(dg, max_iter=50, tol=1e-6)
    mg_res = mg_res.compute()

    estimate = mg_res.copy()
    estimate = estimate.rename(
        columns={"vertex": "vertex", "katz_centrality": "values"}
    )
    estimate["values"] = 0.5

    mg_estimate_res = dcg.katz_centrality(dg, nstart=estimate, max_iter=50, tol=1e-6)
    mg_estimate_res = mg_estimate_res.compute()

    err = 0
    tol = 1.0e-05
    compare_res = mg_res.merge(
        mg_estimate_res, on="vertex", suffixes=["_dask", "_nstart"]
    )

    for i in range(len(compare_res)):
        diff = abs(
            compare_res["katz_centrality_dask"].iloc[i]
            - compare_res["katz_centrality_nstart"].iloc[i]
        )
        if diff > tol * 1.1:
            err = err + 1
    assert err == 0


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS)
def test_dask_mg_katz_centrality_transposed_false(dask_client, dataset):
    ddf = dataset.get_dask_edgelist()
    dg = hipgraph.Graph(directed=True)
    dg.from_dask_cudf_edgelist(ddf, "src", "dst", store_transposed=False)

    warning_msg = (
        "Katz centrality expects the 'store_transposed' "
        "flag to be set to 'True' for optimal performance during "
        "the graph creation"
    )

    with pytest.warns(UserWarning, match=warning_msg):
        dcg.katz_centrality(dg)
