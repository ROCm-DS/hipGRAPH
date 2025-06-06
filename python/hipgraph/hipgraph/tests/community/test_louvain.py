# Copyright (c) 2019-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc

import cudf
import cupyx
import hipgraph
import networkx as nx
import pytest
from hipgraph.datasets import karate_asymmetric
from hipgraph.testing import UNDIRECTED_DATASETS, utils

try:
    import community
except ModuleNotFoundError:
    pytest.exit(
        "community module not found\n"
        "The python-louvain module needs to be installed\n"
        "please run `pip install python-louvain`"
    )


print("Networkx version : {} ".format(nx.__version__))


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


def hipgraph_call(graph_file, edgevals=False, directed=False):
    G = graph_file.get_graph(
        create_using=hipgraph.Graph(directed=directed), ignore_weights=not edgevals
    )
    parts, mod = hipgraph.louvain(G)

    return parts, mod


def networkx_call(M):
    # z = {k: 1.0/M.shape[0] for k in range(M.shape[0])}
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.Graph()
    )
    parts = community.best_partition(Gnx)

    return parts


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", UNDIRECTED_DATASETS)
def test_louvain(graph_file):
    dataset_path = graph_file.get_path()
    M = utils.read_csv_for_nx(dataset_path)
    cu_parts, cu_mod = hipgraph_call(graph_file, edgevals=True)
    nx_parts = networkx_call(M)

    # Calculating modularity scores for comparison
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.Graph()
    )

    cu_parts = cu_parts.to_pandas()
    cu_map = dict(zip(cu_parts["vertex"], cu_parts["partition"]))

    assert set(nx_parts.keys()) == set(cu_map.keys())

    cu_mod_nx = community.modularity(cu_map, Gnx)
    nx_mod = community.modularity(nx_parts, Gnx)

    assert len(cu_parts) == len(nx_parts)
    assert cu_mod > (0.82 * nx_mod)
    assert abs(cu_mod - cu_mod_nx) < 0.0001


@pytest.mark.sg
def test_louvain_directed_graph():
    with pytest.raises(ValueError):
        hipgraph_call(karate_asymmetric, edgevals=True, directed=True)


@pytest.mark.sg
@pytest.mark.parametrize("is_weighted", [True, False])
def test_louvain_csr_graph(is_weighted):
    karate = UNDIRECTED_DATASETS[0]
    df = karate.get_edgelist()

    M = cupyx.scipy.sparse.coo_matrix(
        (df["wgt"].to_cupy(), (df["src"].to_cupy(), df["dst"].to_cupy()))
    )
    M = M.tocsr()

    offsets = cudf.Series(M.indptr)
    indices = cudf.Series(M.indices)
    weights = cudf.Series(M.data)
    G_csr = hipgraph.Graph()
    G_coo = karate.get_graph()

    if not is_weighted:
        weights = None

    G_csr.from_cudf_adjlist(offsets, indices, weights)

    assert G_csr.is_weighted() is is_weighted

    louvain_csr, mod_csr = hipgraph.louvain(G_csr)
    louvain_coo, mod_coo = hipgraph.louvain(G_coo)
    louvain_csr = louvain_csr.sort_values("vertex").reset_index(drop=True)
    result_louvain = (
        louvain_coo.sort_values("vertex")
        .reset_index(drop=True)
        .rename(columns={"partition": "partition_coo"})
    )
    result_louvain["partition_csr"] = louvain_csr["partition"]

    parition_diffs = result_louvain.query("partition_csr != partition_coo")

    assert len(parition_diffs) == 0
    assert mod_csr == mod_coo


@pytest.mark.sg
def test_louvain_nx_graph_with_isolated_nodes():
    # Cluster IDs are expected to unique if all nodes are isolated
    G = nx.Graph()
    G.add_nodes_from(range(5))
    result, _ = hipgraph.louvain(G)
    assert set(result.keys()) == set(G.nodes)
    assert len(set(result.values())) == G.number_of_nodes()

    # A graph with 5 nodes, where 3 of the nodes are isolated
    G.add_edge(1, 2)
    result, _ = hipgraph.louvain(G)
    assert set(result.keys()) == set(G.nodes)
    assert len(set(result.values())) == G.number_of_nodes() - 1
