# Copyright (c) 2019-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc

import hipgraph
import networkx as nx
import pytest
from hipgraph.testing import UNDIRECTED_DATASETS, utils

print("Networkx version : {} ".format(nx.__version__))


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


def calc_k_cores(graph_file, directed=True):
    # directed is used to create either a Graph or DiGraph so the returned
    # hipgraph can be compared to nx graph of same type.
    dataset_path = graph_file.get_path()
    NM = utils.read_csv_for_nx(dataset_path)
    G = graph_file.get_graph(
        create_using=hipgraph.Graph(directed=directed), ignore_weights=True
    )
    if directed:
        Gnx = nx.from_pandas_edgelist(
            NM, source="0", target="1", create_using=nx.DiGraph()
        )
    else:
        Gnx = nx.from_pandas_edgelist(
            NM, source="0", target="1", create_using=nx.Graph()
        )
    ck = hipgraph.k_core(G)
    nk = nx.k_core(Gnx)
    return ck, nk


def compare_edges(cg, nxg):
    edgelist_df = cg.view_edge_list()
    src, dest = edgelist_df["src"], edgelist_df["dst"]
    assert cg.edgelist.weights is False
    assert len(src) == nxg.size()
    for i in range(len(src)):
        assert nxg.has_edge(src[i], dest[i])
    return True


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", UNDIRECTED_DATASETS)
def test_k_core_Graph(graph_file):

    cu_kcore, nx_kcore = calc_k_cores(graph_file, False)

    assert compare_edges(cu_kcore, nx_kcore)


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", UNDIRECTED_DATASETS)
def test_k_core_Graph_nx(graph_file):
    dataset_path = graph_file.get_path()
    NM = utils.read_csv_for_nx(dataset_path)
    Gnx = nx.from_pandas_edgelist(NM, source="0", target="1", create_using=nx.Graph())
    nc = nx.k_core(Gnx)
    cc = hipgraph.k_core(Gnx)

    assert nx.is_isomorphic(nc, cc)


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", UNDIRECTED_DATASETS)
def test_k_core_corenumber_multicolumn(graph_file):
    dataset_path = graph_file.get_path()
    cu_M = utils.read_csv_file(dataset_path)
    cu_M.rename(columns={"0": "src_0", "1": "dst_0"}, inplace=True)
    cu_M["src_1"] = cu_M["src_0"] + 1000
    cu_M["dst_1"] = cu_M["dst_0"] + 1000

    G1 = hipgraph.Graph()
    G1.from_cudf_edgelist(
        cu_M, source=["src_0", "src_1"], destination=["dst_0", "dst_1"]
    )

    corenumber_G1 = hipgraph.core_number(G1)
    corenumber_G1.rename(columns={"core_number": "values"}, inplace=True)
    corenumber_G1 = corenumber_G1[["0_vertex", "1_vertex", "values"]]
    corenumber_G1 = None
    ck_res = hipgraph.k_core(G1, core_number=corenumber_G1)
    G2 = hipgraph.Graph()
    G2.from_cudf_edgelist(cu_M, source="src_0", destination="dst_0", renumber=False)

    corenumber_G2 = hipgraph.core_number(G2)
    corenumber_G2.rename(columns={"core_number": "values"}, inplace=True)
    corenumber_G2 = corenumber_G2[["vertex", "values"]]
    ck_exp = hipgraph.k_core(G2, core_number=corenumber_G2)

    # FIXME: Replace with multi-column view_edge_list()
    edgelist_df = ck_res.edgelist.edgelist_df
    edgelist_df_res = ck_res.unrenumber(edgelist_df, "src")
    edgelist_df_res = ck_res.unrenumber(edgelist_df_res, "dst")

    for i in range(len(edgelist_df_res)):
        assert ck_exp.has_edge(
            edgelist_df_res["0_src"].iloc[i], edgelist_df_res["0_dst"].iloc[i]
        )


@pytest.mark.sg
def test_k_core_invalid_input():
    karate = UNDIRECTED_DATASETS[0]
    G = karate.get_graph(create_using=hipgraph.Graph(directed=True))
    with pytest.raises(ValueError):
        hipgraph.k_core(G)

    G = karate.get_graph()
    degree_type = "invalid"
    with pytest.raises(ValueError):
        hipgraph.k_core(G, degree_type=degree_type)
