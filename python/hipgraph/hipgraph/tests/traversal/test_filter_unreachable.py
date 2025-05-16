# Copyright (c) 2019-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc
import time

import hipgraph
import networkx as nx
import numpy as np
import pytest
from hipgraph.testing import DEFAULT_DATASETS


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


print("Networkx version : {} ".format(nx.__version__))

SOURCES = [1]


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DEFAULT_DATASETS)
@pytest.mark.parametrize("source", SOURCES)
def test_filter_unreachable(graph_file, source):
    G = graph_file.get_graph(create_using=hipgraph.Graph(directed=True))
    cu_M = G.view_edge_list()

    print("sources size = " + str(len(cu_M)))
    print("destinations size = " + str(len(cu_M)))

    print("hipgraph Solving... ")
    t1 = time.time()

    df = hipgraph.sssp(G, source)

    t2 = time.time() - t1
    print("Time : " + str(t2))

    reachable_df = hipgraph.filter_unreachable(df)

    if np.issubdtype(df["distance"].dtype, np.integer):
        inf = np.iinfo(reachable_df["distance"].dtype).max
        assert len(reachable_df.query("distance == @inf")) == 0
    elif np.issubdtype(df["distance"].dtype, np.inexact):
        inf = np.finfo(reachable_df["distance"].dtype).max  # noqa: F841
        assert len(reachable_df.query("distance == @inf")) == 0

    assert len(reachable_df) != 0
