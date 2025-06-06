# Copyright (c) 2020-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc

import hipgraph
import hipgraph.dask as dcg
import pytest
from hipgraph.datasets import netscience

# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


# =============================================================================
# Parameters
# =============================================================================


DATASETS = [netscience]
# Directed graph is not currently supported
IS_DIRECTED = [False, True]


# =============================================================================
# Helper
# =============================================================================


def get_mg_graph(dataset, directed):
    """Returns an MG graph"""
    ddf = dataset.get_dask_edgelist()

    dg = hipgraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(ddf, "src", "dst", "wgt")

    return dg


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("directed", IS_DIRECTED)
def test_dask_mg_wcc(dask_client, dataset, directed):
    input_data_path = dataset.get_path()
    print(f"dataset={input_data_path}")

    g = dataset.get_graph(create_using=hipgraph.Graph(directed=directed))
    dg = get_mg_graph(dataset, directed)

    # breakpoint()
    if not directed:
        expected_dist = hipgraph.weakly_connected_components(g)
        result_dist = dcg.weakly_connected_components(dg)

        result_dist = result_dist.compute()
        compare_dist = expected_dist.merge(
            result_dist, on="vertex", suffixes=["_local", "_dask"]
        )

        unique_local_labels = compare_dist["labels_local"].unique()

        for label in unique_local_labels.values.tolist():
            dask_labels_df = compare_dist[compare_dist["labels_local"] == label]
            dask_labels = dask_labels_df["labels_dask"]
            assert (dask_labels.iloc[0] == dask_labels).all()
    else:
        with pytest.raises(ValueError):
            hipgraph.weakly_connected_components(g)
