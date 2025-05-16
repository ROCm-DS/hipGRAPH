# Copyright (c) 2019-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc
import random

import cudf
import hipgraph
import networkx as nx
import pytest
from hipgraph.testing import DEFAULT_DATASETS, utils
from hipgraph.utilities import ensure_hipgraph_obj_for_nx


def hipgraph_call(G, partitions):
    df = hipgraph.spectralModularityMaximizationClustering(
        G, partitions, num_eigen_vects=(partitions - 1)
    )

    score = hipgraph.analyzeClustering_modularity(
        G, partitions, df, "vertex", "cluster"
    )
    return score


def random_call(G, partitions):
    random.seed(0)
    num_verts = G.number_of_vertices()
    assignment = []
    for i in range(num_verts):
        assignment.append(random.randint(0, partitions - 1))

    assignment_cu = cudf.DataFrame(assignment, columns=["cluster"])
    assignment_cu["vertex"] = assignment_cu.index
    assignment_cu = assignment_cu.astype("int32")

    score = hipgraph.analyzeClustering_modularity(
        G, partitions, assignment_cu, "vertex", "cluster"
    )
    return score


PARTITIONS = [2, 4, 8]


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DEFAULT_DATASETS)
@pytest.mark.parametrize("partitions", PARTITIONS)
def test_modularity_clustering(graph_file, partitions):
    gc.collect()

    # Read in the graph and get a hipgraph object
    G = graph_file.get_graph()
    # read_weights_in_sp=False => value column dtype is float64
    G.edgelist.edgelist_df["weights"] = G.edgelist.edgelist_df["weights"].astype(
        "float64"
    )

    # Get the modularity score for partitioning versus random assignment
    cu_score = hipgraph_call(G, partitions)
    rand_score = random_call(G, partitions)

    # Assert that the partitioning has better modularity than the random
    # assignment
    assert cu_score > rand_score


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DEFAULT_DATASETS)
@pytest.mark.parametrize("partitions", PARTITIONS)
def test_modularity_clustering_nx(graph_file, partitions):
    # Read in the graph and get a hipgraph object
    dataset_path = graph_file.get_path()
    csv_data = utils.read_csv_for_nx(dataset_path, read_weights_in_sp=True)

    nxG = nx.from_pandas_edgelist(
        csv_data,
        source="0",
        target="1",
        edge_attr="weight",
        create_using=nx.Graph(),
    )
    assert nx.is_directed(nxG) is False
    assert nx.is_weighted(nxG) is True

    cuG, isNx = ensure_hipgraph_obj_for_nx(nxG)
    assert hipgraph.is_directed(cuG) is False
    assert hipgraph.is_weighted(cuG) is True

    # Get the modularity score for partitioning versus random assignment
    cu_score = hipgraph_call(cuG, partitions)
    rand_score = random_call(cuG, partitions)

    # Assert that the partitioning has better modularity than the random
    # assignment
    assert cu_score > rand_score


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DEFAULT_DATASETS)
@pytest.mark.parametrize("partitions", PARTITIONS)
def test_modularity_clustering_multi_column(graph_file, partitions):
    # Read in the graph and get a hipgraph object
    dataset_path = graph_file.get_path()
    cu_M = utils.read_csv_file(dataset_path, read_weights_in_sp=False)
    cu_M.rename(columns={"0": "src_0", "1": "dst_0"}, inplace=True)
    cu_M["src_1"] = cu_M["src_0"] + 1000
    cu_M["dst_1"] = cu_M["dst_0"] + 1000

    G1 = hipgraph.Graph()
    G1.from_cudf_edgelist(
        cu_M, source=["src_0", "src_1"], destination=["dst_0", "dst_1"], edge_attr="2"
    )

    df1 = hipgraph.spectralModularityMaximizationClustering(
        G1, partitions, num_eigen_vects=(partitions - 1)
    )

    cu_score = hipgraph.analyzeClustering_modularity(
        G1, partitions, df1, ["0_vertex", "1_vertex"], "cluster"
    )

    G2 = hipgraph.Graph()
    G2.from_cudf_edgelist(cu_M, source="src_0", destination="dst_0", edge_attr="2")

    rand_score = random_call(G2, partitions)
    # Assert that the partitioning has better modularity than the random
    # assignment
    assert cu_score > rand_score


# Test to ensure DiGraph objs are not accepted
# Test all combinations of default/managed and pooled/non-pooled allocation


@pytest.mark.sg
def test_digraph_rejected():
    df = cudf.DataFrame()
    df["src"] = cudf.Series(range(10))
    df["dst"] = cudf.Series(range(10))
    df["val"] = cudf.Series(range(10))

    G = hipgraph.Graph(directed=True)
    G.from_cudf_edgelist(
        df, source="src", destination="dst", edge_attr="val", renumber=False
    )

    with pytest.raises(ValueError):
        hipgraph_call(G, 2)
