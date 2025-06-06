# Copyright (c) 2020-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc
import random

import cudf
import dask_cudf
import hipgraph
import hipgraph.dask as dcg
import pytest
from hipgraph.testing.utils import RAPIDS_DATASET_ROOT_DIR_PATH

# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


IS_DIRECTED = [True, False]


# @pytest.mark.skipif(
#    is_single_gpu(), reason="skipping MG testing on Single GPU system"
# )
@pytest.mark.mg
@pytest.mark.parametrize("directed", IS_DIRECTED)
def test_dask_mg_bfs(dask_client, directed):

    input_data_path = (RAPIDS_DATASET_ROOT_DIR_PATH / "netscience.csv").as_posix()

    print(f"dataset={input_data_path}")
    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(
        input_data_path,
        blocksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    def modify_dataset(df):
        temp_df = cudf.DataFrame()
        temp_df["src"] = df["src"] + 1000
        temp_df["dst"] = df["dst"] + 1000
        temp_df["value"] = df["value"]
        return cudf.concat([df, temp_df])

    meta = ddf._meta
    ddf = ddf.map_partitions(
        modify_dataset, meta=meta, token="custom-" + str(random.random())
    )

    df = cudf.read_csv(
        input_data_path,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    df = modify_dataset(df)

    g = hipgraph.Graph(directed=directed)
    g.from_cudf_edgelist(df, "src", "dst")

    dg = hipgraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(ddf, "src", "dst")

    expected_dist = hipgraph.bfs(g, [0, 1000])
    result_dist = dcg.bfs(dg, [0, 1000])
    result_dist = result_dist.compute()

    compare_dist = expected_dist.merge(
        result_dist, on="vertex", suffixes=["_local", "_dask"]
    )

    err = 0

    for i in range(len(compare_dist)):
        if (
            compare_dist["distance_local"].iloc[i]
            != compare_dist["distance_dask"].iloc[i]
        ):
            err = err + 1
    assert err == 0


# @pytest.mark.skipif(
#    is_single_gpu(), reason="skipping MG testing on Single GPU system"
# )
@pytest.mark.mg
@pytest.mark.parametrize("directed", IS_DIRECTED)
def test_dask_mg_bfs_invalid_start(dask_client, directed):
    source_vertex = 10
    input_data_path = (RAPIDS_DATASET_ROOT_DIR_PATH / "netscience.csv").as_posix()

    print(f"dataset={input_data_path}")
    chunksize = dcg.get_chunksize(input_data_path)

    el = dask_cudf.read_csv(
        input_data_path,
        blocksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )
    newval = max(el.src.max().compute(), el.dst.max().compute()) + 1
    el.src = el.src.replace(source_vertex, newval)
    el.dst = el.dst.replace(source_vertex, newval)

    G = hipgraph.Graph(directed=directed)
    G.from_dask_cudf_edgelist(el, "src", "dst")

    with pytest.raises(ValueError):
        dcg.bfs(G, source_vertex).compute()

    # invalid dtype (the default cudf.Series() dtype is int64)
    source_vertex = cudf.Series([0, 1])
    warning_msg = "The 'start' values dtype must match " "the graph's vertices dtype."
    with pytest.warns(UserWarning, match=warning_msg):
        dcg.bfs(G, source_vertex).compute()


# @pytest.mark.skipif(
#     is_single_gpu(), reason="skipping MG testing on Single GPU system"
# )
@pytest.mark.mg
@pytest.mark.parametrize("directed", IS_DIRECTED)
def test_dask_mg_bfs_multi_column_depthlimit(dask_client, directed):
    gc.collect()

    input_data_path = (RAPIDS_DATASET_ROOT_DIR_PATH / "netscience.csv").as_posix()
    print(f"dataset={input_data_path}")
    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(
        input_data_path,
        blocksize=chunksize,
        delimiter=" ",
        names=["src_a", "dst_a", "value"],
        dtype=["int32", "int32", "float32"],
    )
    ddf["src_b"] = ddf["src_a"] + 1000
    ddf["dst_b"] = ddf["dst_a"] + 1000

    df = cudf.read_csv(
        input_data_path,
        delimiter=" ",
        names=["src_a", "dst_a", "value"],
        dtype=["int32", "int32", "float32"],
    )
    df["src_b"] = df["src_a"] + 1000
    df["dst_b"] = df["dst_a"] + 1000

    g = hipgraph.Graph(directed=directed)
    g.from_cudf_edgelist(df, ["src_a", "src_b"], ["dst_a", "dst_b"])

    dg = hipgraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(ddf, ["src_a", "src_b"], ["dst_a", "dst_b"])

    start = cudf.DataFrame()
    start["a"] = [0]
    start["b"] = [1000]

    depth_limit = 18
    expected_dist = hipgraph.bfs(g, start, depth_limit=depth_limit)
    result_dist = dcg.bfs(dg, start, depth_limit=depth_limit)
    result_dist = result_dist.compute()

    compare_dist = expected_dist.merge(
        result_dist, on=["0_vertex", "1_vertex"], suffixes=["_local", "_dask"]
    )

    err = 0
    for i in range(len(compare_dist)):
        if (
            compare_dist["distance_local"].iloc[i] <= depth_limit
            and compare_dist["distance_dask"].iloc[i] <= depth_limit
            and compare_dist["distance_local"].iloc[i]
            != compare_dist["distance_dask"].iloc[i]
        ):
            err = err + 1
    assert err == 0
