# Copyright (c) 2020-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc
from timeit import default_timer as timer

import cudf
import hipgraph
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment


def create_random_bipartite(v1, v2, size, dtype):

    #
    #   Create a full bipartite graph
    #
    df1 = cudf.DataFrame()
    df1["src"] = cudf.Series(range(0, v1, 1))
    df1["key"] = 1

    df2 = cudf.DataFrame()
    df2["dst"] = cudf.Series(range(v1, v1 + v2, 1))
    df2["key"] = 1

    edges = df1.merge(df2, on="key")[["src", "dst"]]
    edges = edges.sort_values(["src", "dst"]).reset_index()

    # Generate edge weights
    a = np.random.randint(1, high=size, size=(v1, v2)).astype(dtype)
    edges["weight"] = a.flatten()

    g = hipgraph.Graph()
    g.from_cudf_edgelist(
        edges, source="src", destination="dst", edge_attr="weight", renumber=False
    )

    return df1["src"], g, a


SPARSE_SIZES = [[5, 5, 100], [500, 500, 10000]]
DENSE_SIZES = [[5, 100], [500, 10000]]


def setup_function():
    gc.collect()


@pytest.mark.sg
@pytest.mark.parametrize("v1_size, v2_size, weight_limit", SPARSE_SIZES)
def test_hungarian(v1_size, v2_size, weight_limit):
    v1, g, m = create_random_bipartite(v1_size, v2_size, weight_limit, np.float64)

    start = timer()
    hipgraph_cost, matching = hipgraph.hungarian(g, v1)
    end = timer()

    print("hipgraph time: ", (end - start))

    start = timer()
    np_matching = linear_sum_assignment(m)
    end = timer()

    print("scipy time: ", (end - start))

    scipy_cost = m[np_matching[0], np_matching[1]].sum()

    assert scipy_cost == hipgraph_cost


@pytest.mark.sg
@pytest.mark.parametrize("n, weight_limit", DENSE_SIZES)
def test_dense_hungarian(n, weight_limit):
    C = np.random.uniform(0, weight_limit, size=(n, n)).round().astype(np.float32)

    C_series = cudf.Series(C.flatten())

    start = timer()
    hipgraph_cost, matching = hipgraph.dense_hungarian(C_series, n, n)
    end = timer()

    print("hipgraph time: ", (end - start))

    start = timer()
    np_matching = linear_sum_assignment(C)
    end = timer()

    print("scipy time: ", (end - start))

    scipy_cost = C[np_matching[0], np_matching[1]].sum()

    assert scipy_cost == hipgraph_cost
