# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc

import dask_cudf
import numpy as np
import pytest
from cudf.testing.testing import assert_frame_equal
from hipgraph.datasets import dolphins, karate, karate_disjoint
from hipgraph.structure.replicate_edgelist import replicate_edgelist

# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


# =============================================================================
# Parameters
# =============================================================================


edgeWeightCol = "weights"
edgeIdCol = "edge_id"
edgeTypeCol = "edge_type"
srcCol = "src"
dstCol = "dst"

DATASETS = [karate, dolphins, karate_disjoint]
IS_DISTRIBUTED = [True, False]
USE_WEIGHTS = [True, False]
USE_EDGE_IDS = [True, False]
USE_EDGE_TYPE_IDS = [True, False]


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("distributed", IS_DISTRIBUTED)
@pytest.mark.parametrize("use_weights", USE_WEIGHTS)
@pytest.mark.parametrize("use_edge_ids", USE_EDGE_IDS)
@pytest.mark.parametrize("use_edge_type_ids", USE_EDGE_TYPE_IDS)
def test_mg_replicate_edgelist(
    dask_client, dataset, distributed, use_weights, use_edge_ids, use_edge_type_ids
):
    dataset.unload()
    df = dataset.get_edgelist()

    columns = [srcCol, dstCol]
    weight = None
    edge_id = None
    edge_type = None

    if use_weights:
        df = df.rename(columns={"wgt": edgeWeightCol})
        columns.append(edgeWeightCol)
        weight = edgeWeightCol
    if use_edge_ids:
        df = df.reset_index().rename(columns={"index": edgeIdCol})
        df[edgeIdCol] = df[edgeIdCol].astype(df[srcCol].dtype)
        columns.append(edgeIdCol)
        edge_id = edgeIdCol
    if use_edge_type_ids:
        df[edgeTypeCol] = np.random.randint(0, 10, size=len(df))
        df[edgeTypeCol] = df[edgeTypeCol].astype(df[srcCol].dtype)
        columns.append(edgeTypeCol)
        edge_type = edgeTypeCol

    if distributed:
        # Distribute the edges across all ranks
        num_workers = len(dask_client.scheduler_info()["workers"])
        df = dask_cudf.from_cudf(df, npartitions=num_workers)
    ddf = replicate_edgelist(
        df[columns], weight=weight, edge_id=edge_id, edge_type=edge_type
    )

    if distributed:
        df = df.compute()

    for i in range(ddf.npartitions):
        result_df = (
            ddf.get_partition(i)
            .compute()
            .sort_values([srcCol, dstCol])
            .reset_index(drop=True)
        )
        expected_df = df[columns].sort_values([srcCol, dstCol]).reset_index(drop=True)

        assert_frame_equal(expected_df, result_df, check_dtype=False, check_like=True)
