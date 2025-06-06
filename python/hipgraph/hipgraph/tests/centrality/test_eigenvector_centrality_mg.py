# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
from hipgraph.datasets import dolphins, karate_disjoint, netscience

# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


# =============================================================================
# Parameters
# =============================================================================


DATASETS = [karate_disjoint, dolphins, netscience]
IS_DIRECTED = [True, False]


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.mg
@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("directed", IS_DIRECTED)
def test_dask_mg_eigenvector_centrality(dask_client, dataset, directed):
    input_data_path = dataset.get_path()
    print(f"dataset={input_data_path}")
    ddf = dataset.get_dask_edgelist()
    dg = hipgraph.Graph(directed=True)
    dg.from_dask_cudf_edgelist(ddf, "src", "dst", store_transposed=True)
    mg_res = dcg.eigenvector_centrality(dg, tol=1e-6)
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
    # FIXME: Compare against hipgraph instead of nx
    nk = nx.eigenvector_centrality(Gnx)
    import pandas as pd

    pdf = pd.DataFrame(nk.items(), columns=["vertex", "eigenvector_centrality"])
    exp_res = cudf.DataFrame(pdf)
    err = 0
    tol = 1.0e-05
    compare_res = exp_res.merge(mg_res, on="vertex", suffixes=["_local", "_dask"])
    for i in range(len(compare_res)):
        diff = abs(
            compare_res["eigenvector_centrality_local"].iloc[i]
            - compare_res["eigenvector_centrality_dask"].iloc[i]
        )
        if diff > tol * 1.1:
            err = err + 1
    assert err == 0


@pytest.mark.mg
def test_dask_mg_eigenvector_centrality_transposed_false(dask_client):
    dataset = DATASETS[0]

    ddf = dataset.get_dask_edgelist()
    dg = hipgraph.Graph(directed=True)
    dg.from_dask_cudf_edgelist(ddf, "src", "dst", store_transposed=False)

    warning_msg = (
        "Eigenvector centrality expects the 'store_transposed' "
        "flag to be set to 'True' for optimal performance during "
        "the graph creation"
    )

    with pytest.warns(UserWarning, match=warning_msg):
        dcg.eigenvector_centrality(dg)
