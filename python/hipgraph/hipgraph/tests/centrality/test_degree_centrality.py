# Copyright (c) 2022-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc

import cudf
import hipgraph
import networkx as nx
import pytest
from hipgraph.testing import UNDIRECTED_DATASETS, utils


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


def topKVertices(degree, col, k):
    top = degree.nlargest(n=k, columns=col)
    top = top.sort_values(by=col, ascending=False)
    return top["vertex"]


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", UNDIRECTED_DATASETS)
def test_degree_centrality_nx(graph_file):
    dataset_path = graph_file.get_path()
    NM = utils.read_csv_for_nx(dataset_path)
    Gnx = nx.from_pandas_edgelist(
        NM,
        create_using=nx.DiGraph(),
        source="0",
        target="1",
    )

    G = hipgraph.utilities.convert_from_nx(Gnx)

    nk = nx.degree_centrality(Gnx)
    ck = hipgraph.degree_centrality(G)

    # Calculating mismatch
    nk = sorted(nk.items(), key=lambda x: x[0])
    ck = ck.sort_values("vertex")
    ck.index = ck["vertex"]
    ck = ck["degree_centrality"]
    err = 0

    assert len(ck) == len(nk)
    for i in range(len(ck)):
        if abs(ck[i] - nk[i][1]) > 0.1 and ck.index[i] == nk[i][0]:
            err = err + 1
    print("Mismatches:", err)
    assert err < (0.1 * len(ck))


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", UNDIRECTED_DATASETS)
def test_degree_centrality_multi_column(graph_file):
    dataset_path = graph_file.get_path()
    cu_M = utils.read_csv_file(dataset_path)
    cu_M.rename(columns={"0": "src_0", "1": "dst_0"}, inplace=True)
    cu_M["src_1"] = cu_M["src_0"] + 1000
    cu_M["dst_1"] = cu_M["dst_0"] + 1000

    G1 = hipgraph.Graph(directed=True)
    G1.from_cudf_edgelist(
        cu_M, source=["src_0", "src_1"], destination=["dst_0", "dst_1"]
    )

    G2 = hipgraph.Graph(directed=True)
    G2.from_cudf_edgelist(cu_M, source="src_0", destination="dst_0")

    k_df_exp = hipgraph.degree_centrality(G2)
    k_df_exp = k_df_exp.sort_values("vertex").reset_index(drop=True)

    nstart = cudf.DataFrame()
    nstart["vertex_0"] = k_df_exp["vertex"]
    nstart["vertex_1"] = nstart["vertex_0"] + 1000
    nstart["values"] = k_df_exp["degree_centrality"]

    k_df_res = hipgraph.degree_centrality(G1)
    k_df_res = k_df_res.sort_values("0_vertex").reset_index(drop=True)
    k_df_res.rename(columns={"0_vertex": "vertex"}, inplace=True)

    top_res = topKVertices(k_df_res, "degree_centrality", 10)
    top_exp = topKVertices(k_df_exp, "degree_centrality", 10)

    assert top_res.equals(top_exp)
