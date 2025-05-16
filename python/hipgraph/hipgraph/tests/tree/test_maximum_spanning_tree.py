# Copyright (c) 2019-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc
import time

import cudf
import hipgraph
import networkx as nx
import numpy as np
import pytest
import rmm
from hipgraph.datasets import netscience
from hipgraph.testing import utils

print("Networkx version : {} ".format(nx.__version__))

UNDIRECTED_WEIGHTED_DATASET = [netscience]

# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


def _get_param_args(param_name, param_values):
    """
    Returns a tuple of (<param_name>, <pytest.param list>) which can be applied
    as the args to pytest.mark.parametrize(). The pytest.param list also
    contains param id string formed from the param name and values.
    """
    return (param_name, [pytest.param(v, id=f"{param_name}={v}") for v in param_values])


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", UNDIRECTED_WEIGHTED_DATASET)
def test_maximum_spanning_tree_nx(graph_file):
    # hipgraph
    G = graph_file.get_graph()
    # read_weights_in_sp=False => value column dtype is float64
    G.edgelist.edgelist_df["weights"] = G.edgelist.edgelist_df["weights"].astype(
        "float64"
    )

    # Just for getting relevant timing
    G.view_adj_list()
    t1 = time.time()
    hipgraph_mst = hipgraph.maximum_spanning_tree(G)
    t2 = time.time() - t1
    print("hipGRAPH time : " + str(t2))

    # Nx
    dataset_path = graph_file.get_path()
    df = utils.read_csv_for_nx(dataset_path, read_weights_in_sp=True)
    Gnx = nx.from_pandas_edgelist(
        df, create_using=nx.Graph(), source="0", target="1", edge_attr="weight"
    )
    t1 = time.time()
    mst_nx = nx.maximum_spanning_tree(Gnx)
    t2 = time.time() - t1
    print("Nx Time : " + str(t2))

    utils.compare_mst(hipgraph_mst, mst_nx)


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", UNDIRECTED_WEIGHTED_DATASET)
@pytest.mark.parametrize(*_get_param_args("use_adjlist", [True, False]))
def test_maximum_spanning_tree_graph_repr_compat(graph_file, use_adjlist):
    G = graph_file.get_graph()
    # read_weights_in_sp=False => value column dtype is float64
    G.edgelist.edgelist_df["weights"] = G.edgelist.edgelist_df["weights"].astype(
        "float64"
    )
    if use_adjlist:
        G.view_adj_list()
    hipgraph.maximum_spanning_tree(G)


DATASETS_SIZES = [
    100000,
    1000000,
    10000000,
    100000000,
]


@pytest.mark.sg
@pytest.mark.skip(reason="Skipping large tests")
@pytest.mark.parametrize("graph_size", DATASETS_SIZES)
def test_random_maximum_spanning_tree_nx(graph_size):
    rmm.reinitialize(managed_memory=True)
    df = utils.random_edgelist(
        e=graph_size,
        ef=16,
        dtypes={"src": np.int32, "dst": np.int32, "weight": float},
        drop_duplicates=True,
        seed=123456,
    )
    gdf = cudf.from_pandas(df)
    # hipgraph
    G = hipgraph.Graph()
    G.from_cudf_edgelist(gdf, source="src", destination="dst", edge_attr="weight")
    # Just for getting relevant timing
    G.view_adj_list()
    t1 = time.time()
    hipgraph.maximum_spanning_tree(G)
    t2 = time.time() - t1
    print("hipGRAPH time : " + str(t2))

    # Nx
    Gnx = nx.from_pandas_edgelist(
        df,
        create_using=nx.Graph(),
        source="src",
        target="dst",
        edge_attr="weight",
    )
    t1 = time.time()
    nx.maximum_spanning_tree(Gnx)
    t3 = time.time() - t1
    print("Nx Time : " + str(t3))
    print("Speedup: " + str(t3 / t2))
