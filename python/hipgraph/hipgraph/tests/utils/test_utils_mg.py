# Copyright (c) 2018-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc
import os
import time

import dask_cudf
import hipgraph
import hipgraph.dask as dcg
import numpy as np
import pytest
from dask.distributed import default_client, futures_of, wait
from hipgraph.dask.common.mg_utils import is_single_gpu
from hipgraph.dask.common.part_utils import concat_within_workers
from hipgraph.dask.common.read_utils import get_n_workers
from hipgraph.testing import utils
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
def test_from_edgelist(dask_client, directed):
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

    dg1 = hipgraph.from_edgelist(
        ddf,
        source="src",
        destination="dst",
        edge_attr="value",
        create_using=hipgraph.Graph(directed=directed),
    )

    dg2 = hipgraph.Graph(directed=directed)
    dg2.from_dask_cudf_edgelist(ddf, source="src", destination="dst", edge_attr="value")

    assert dg1.EdgeList == dg2.EdgeList


@pytest.mark.mg
@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.skip(reason="MG not supported on CI")
def test_parquet_concat_within_workers(dask_client):
    if not os.path.exists("test_files_parquet"):
        print("Generate data... ")
        os.mkdir("test_files_parquet")
    for x in range(10):
        if not os.path.exists("test_files_parquet/df" + str(x)):
            df = utils.random_edgelist(
                e=100, ef=16, dtypes={"src": np.int32, "dst": np.int32}, seed=x
            )
            df.to_parquet("test_files_parquet/df" + str(x), index=False)

    n_gpu = get_n_workers()

    print("Read_parquet... ")
    t1 = time.time()
    ddf = dask_cudf.read_parquet("test_files_parquet/*", dtype=["int32", "int32"])
    ddf = ddf.persist()
    futures_of(ddf)
    wait(ddf)
    t1 = time.time() - t1
    print("*** Read Time: ", t1, "s")
    print(ddf)

    assert ddf.npartitions > n_gpu

    print("Drop_duplicates... ")
    t2 = time.time()
    ddf.drop_duplicates(inplace=True)
    ddf = ddf.persist()
    futures_of(ddf)
    wait(ddf)
    t2 = time.time() - t2
    print("*** Drop duplicate time: ", t2, "s")
    assert t2 < t1

    print("Repartition... ")
    t3 = time.time()
    # Notice that ideally we would use :
    # ddf = ddf.repartition(npartitions=n_gpu)
    # However this is slower than reading and requires more memory
    # Using custom concat instead
    client = default_client()
    ddf = concat_within_workers(client, ddf)
    ddf = ddf.persist()
    futures_of(ddf)
    wait(ddf)
    t3 = time.time() - t3
    print("*** repartition Time: ", t3, "s")
    print(ddf)

    assert t3 < t1
