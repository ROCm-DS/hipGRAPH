# Copyright (c) 2020-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc

import hipgraph
import networkx as nx
import numpy as np
import pytest
from hipgraph.datasets import karate_asymmetric, polbooks
from hipgraph.testing import utils

# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


# These ground truth files have been created by running the networkx ktruss
# function on reference graphs. Currently networkx ktruss has an error such
# that nx.k_truss(G,k-2) gives the expected result for running ktruss with
# parameter k. This fix (https://github.com/networkx/networkx/pull/3713) is
# currently in networkx master and will hopefully will make it to a release
# soon.
def ktruss_ground_truth(graph_file):
    G = nx.read_edgelist(str(graph_file), nodetype=int, data=(("weight", float),))
    df = nx.to_pandas_edgelist(G)
    return df


def compare_k_truss(k_truss_hipgraph, k, ground_truth_file):
    k_truss_nx = ktruss_ground_truth(ground_truth_file)

    edgelist_df = k_truss_hipgraph.view_edge_list()
    src = edgelist_df["src"]
    dst = edgelist_df["dst"]
    wgt = edgelist_df["weight"]
    assert len(edgelist_df) == len(k_truss_nx)
    for i in range(len(src)):
        has_edge = (
            (k_truss_nx["source"] == src[i])
            & (k_truss_nx["target"] == dst[i])
            & np.isclose(k_truss_nx["weight"], wgt[i])
        ).any()
        has_opp_edge = (
            (k_truss_nx["source"] == dst[i])
            & (k_truss_nx["target"] == src[i])
            & np.isclose(k_truss_nx["weight"], wgt[i])
        ).any()
        assert has_edge or has_opp_edge
    return True


@pytest.mark.sg
@pytest.mark.parametrize("_, nx_ground_truth", utils.DATASETS_KTRUSS)
def test_ktruss_subgraph_Graph(_, nx_ground_truth):

    k = 5
    G = polbooks.get_graph(download=True, create_using=hipgraph.Graph(directed=False))
    k_subgraph = hipgraph.ktruss_subgraph(G, k, use_weights=False)

    compare_k_truss(k_subgraph, k, nx_ground_truth)


@pytest.mark.sg
def test_ktruss_subgraph_Graph_nx():
    k = 5
    dataset_path = polbooks.get_path()
    M = utils.read_csv_for_nx(dataset_path, read_weights_in_sp=True)
    G = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.Graph()
    )
    k_subgraph = hipgraph.k_truss(G, k)
    k_truss_nx = nx.k_truss(G, k)

    assert nx.is_isomorphic(k_subgraph, k_truss_nx)


@pytest.mark.sg
def test_ktruss_subgraph_directed_Graph():
    k = 5
    edgevals = True
    G = karate_asymmetric.get_graph(
        download=True,
        create_using=hipgraph.Graph(directed=True),
        ignore_weights=not edgevals,
    )
    with pytest.raises(ValueError):
        hipgraph.k_truss(G, k)
