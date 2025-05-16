# Copyright (c) 2020-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import hipgraph
import hipgraph.dask as dcg
import pytest
from hipgraph.datasets import dolphins, karate, karate_asymmetric

# =============================================================================
# Parameters
# =============================================================================


DATASETS = [karate, dolphins]
DATASETS_ASYMMETRIC = [karate_asymmetric]


# =============================================================================
# Helper Functions
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
# FIXME: Implement more robust tests


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS_ASYMMETRIC)
def test_mg_leiden_with_edgevals_directed_graph(dask_client, dataset):
    dg = get_mg_graph(dataset, directed=True)
    # Directed graphs are not supported by Leiden and a ValueError should be
    # raised
    with pytest.raises(ValueError):
        parts, mod = dcg.leiden(dg)


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS)
def test_mg_leiden_with_edgevals_undirected_graph(dask_client, dataset):
    dg = get_mg_graph(dataset, directed=False)
    parts, mod = dcg.leiden(dg)

    # FIXME: either call Nx with the same dataset and compare results, or
    # hardcode golden results to compare to.
    print()
    print(parts.compute())
    print(mod)
    print()
