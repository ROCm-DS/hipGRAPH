# Copyright (c) 2020-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc

import cudf
import dask_cudf
import hipgraph
import hipgraph.dask as dcg
import numpy as np
import pytest
from hipgraph.dask.common.mg_utils import is_single_gpu
from hipgraph.testing import utils
from hipgraph.testing.utils import RAPIDS_DATASET_ROOT_DIR_PATH


# The function selects personalization_perc% of accessible vertices in graph M
# and randomly assigns them personalization values
def personalize(vertices, personalization_perc):
    personalization = None
    if personalization_perc != 0:
        personalization = {}
        nnz_vtx = vertices.values_host
        personalization_count = int((nnz_vtx.size * personalization_perc) / 100.0)
        nnz_vtx = np.random.choice(
            nnz_vtx, min(nnz_vtx.size, personalization_count), replace=False
        )
        nnz_val = np.random.random(nnz_vtx.size)
        nnz_val = nnz_val / sum(nnz_val)
        for vtx, val in zip(nnz_vtx, nnz_val):
            personalization[vtx] = val

        k = np.fromiter(personalization.keys(), dtype="int32")
        v = np.fromiter(personalization.values(), dtype="float32")
        cu_personalization = cudf.DataFrame({"vertex": k, "values": v})

    return cu_personalization, personalization


def create_distributed_karate_graph(store_transposed=True):
    input_data_path = (RAPIDS_DATASET_ROOT_DIR_PATH / "karate.csv").as_posix()

    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(
        input_data_path,
        blocksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    dg = hipgraph.Graph(directed=True)
    dg.from_dask_cudf_edgelist(ddf, "src", "dst", store_transposed=store_transposed)

    return dg


# =============================================================================
# Parameters
# =============================================================================
PERSONALIZATION_PERC = [0, 10, 50]
IS_DIRECTED = [True, False]
HAS_GUESS = [0, 1]
HAS_PRECOMPUTED = [0, 1]


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


@pytest.mark.mg
@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize("personalization_perc", PERSONALIZATION_PERC)
@pytest.mark.parametrize("directed", IS_DIRECTED)
@pytest.mark.parametrize("has_precomputed_vertex_out_weight", HAS_PRECOMPUTED)
@pytest.mark.parametrize("has_guess", HAS_GUESS)
def test_dask_mg_pagerank(
    dask_client,
    personalization_perc,
    directed,
    has_precomputed_vertex_out_weight,
    has_guess,
):

    input_data_path = (RAPIDS_DATASET_ROOT_DIR_PATH / "karate.csv").as_posix()
    print(f"dataset={input_data_path}")
    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(
        input_data_path,
        blocksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    df = cudf.read_csv(
        input_data_path,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    g = hipgraph.Graph(directed=directed)
    g.from_cudf_edgelist(df, "src", "dst", "value")

    dg = hipgraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(ddf, "src", "dst", "value", store_transposed=True)

    personalization = None
    pre_vtx_o_wgt = None
    nstart = None
    max_iter = 100
    has_precomputed_vertex_out_weight
    if personalization_perc != 0:
        personalization, p = personalize(g.nodes(), personalization_perc)
    if has_precomputed_vertex_out_weight == 1:
        df = df[["src", "value"]]
        pre_vtx_o_wgt = (
            df.groupby(["src"], as_index=False)
            .sum()
            .rename(columns={"src": "vertex", "value": "sums"})
        )

    if has_guess == 1:
        nstart = hipgraph.pagerank(g, personalization=personalization, tol=1e-6).rename(
            columns={"pagerank": "values"}
        )
        max_iter = 20

    expected_pr = hipgraph.pagerank(
        g,
        personalization=personalization,
        precomputed_vertex_out_weight=pre_vtx_o_wgt,
        max_iter=max_iter,
        tol=1e-6,
        nstart=nstart,
    )
    result_pr = dcg.pagerank(
        dg,
        personalization=personalization,
        precomputed_vertex_out_weight=pre_vtx_o_wgt,
        max_iter=max_iter,
        tol=1e-6,
        nstart=nstart,
    )
    result_pr = result_pr.compute()

    err = 0
    tol = 1.0e-05

    assert len(expected_pr) == len(result_pr)

    compare_pr = expected_pr.merge(result_pr, on="vertex", suffixes=["_local", "_dask"])

    for i in range(len(compare_pr)):
        diff = abs(
            compare_pr["pagerank_local"].iloc[i] - compare_pr["pagerank_dask"].iloc[i]
        )
        if diff > tol * 1.1:
            err = err + 1
    assert err == 0


@pytest.mark.mg
def test_pagerank_invalid_personalization_dtype(dask_client):
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
    dg.from_dask_cudf_edgelist(
        ddf,
        source="src",
        destination="dst",
        edge_attr="value",
        renumber=True,
        store_transposed=True,
    )

    personalization_vec = cudf.DataFrame()
    personalization_vec["vertex"] = [17, 26]
    personalization_vec["values"] = [0.5, 0.75]
    warning_msg = (
        "PageRank requires 'personalization' values to match the "
        "graph's 'edge_attr' type. edge_attr type is: "
        "float32 and got 'personalization' values "
        "of type: float64."
    )

    with pytest.warns(UserWarning, match=warning_msg):
        dcg.pagerank(dg, personalization=personalization_vec)


@pytest.mark.mg
def test_dask_mg_pagerank_transposed_false(dask_client):
    dg = create_distributed_karate_graph(store_transposed=False)

    warning_msg = (
        "Pagerank expects the 'store_transposed' "
        "flag to be set to 'True' for optimal performance during "
        "the graph creation"
    )

    with pytest.warns(UserWarning, match=warning_msg):
        dcg.pagerank(dg)


@pytest.mark.mg
def test_pagerank_non_convergence(dask_client):
    dg = create_distributed_karate_graph()

    # Not enough allowed iterations, should not converge
    with pytest.raises(hipgraph.exceptions.FailedToConvergeError):
        ddf = dcg.pagerank(dg, max_iter=1, fail_on_nonconvergence=True)

    # Not enough allowed iterations, should not converge but do not consider
    # that an error
    (ddf, converged) = dcg.pagerank(dg, max_iter=1, fail_on_nonconvergence=False)
    assert type(ddf) is dask_cudf.DataFrame
    assert type(converged) is bool
    assert converged is False

    # The default max_iter value should allow convergence for this graph
    (ddf, converged) = dcg.pagerank(dg, fail_on_nonconvergence=False)
    assert type(ddf) is dask_cudf.DataFrame
    assert type(converged) is bool
    assert converged is True

    # Test personalized pagerank the same way
    personalization = cudf.DataFrame()
    personalization["vertex"] = [17, 26]
    personalization["values"] = [0.5, 0.75]

    with pytest.raises(hipgraph.exceptions.FailedToConvergeError):
        df = dcg.pagerank(
            dg, max_iter=1, personalization=personalization, fail_on_nonconvergence=True
        )

    (df, converged) = dcg.pagerank(
        dg, max_iter=1, personalization=personalization, fail_on_nonconvergence=False
    )
    assert type(df) is dask_cudf.DataFrame
    assert type(converged) is bool
    assert converged is False

    (df, converged) = dcg.pagerank(
        dg, personalization=personalization, fail_on_nonconvergence=False
    )
    assert type(df) is dask_cudf.DataFrame
    assert type(converged) is bool
    assert converged is True
