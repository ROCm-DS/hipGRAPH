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
def test_dask_mg_sssp(dask_client, directed):

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

    df = cudf.read_csv(
        input_data_path,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    g = hipgraph.Graph(directed=directed)
    g.from_cudf_edgelist(df, "src", "dst", "value", renumber=True)

    dg = hipgraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(ddf, "src", "dst", "value")

    expected_dist = hipgraph.sssp(g, 0)
    print(expected_dist)
    result_dist = dcg.sssp(dg, 0)
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
