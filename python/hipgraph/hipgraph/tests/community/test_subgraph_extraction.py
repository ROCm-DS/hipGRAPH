# Copyright (c) 2019-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc

import cudf
import hipgraph
import networkx as nx
import numpy as np
import pytest
from hipgraph.datasets import karate
from hipgraph.testing import DEFAULT_DATASETS, utils


###############################################################################
# pytest setup - called for each test function
def setup_function():
    gc.collect()


###############################################################################
def compare_edges(cg, nxg):
    edgelist_df = cg.view_edge_list()

    assert len(edgelist_df) == nxg.size()
    for i in range(len(edgelist_df)):
        assert nxg.has_edge(edgelist_df["src"].iloc[i], edgelist_df["dst"].iloc[i])
    return True


def hipgraph_call(M, verts, directed=True):
    # hipgraph can be compared to nx graph of same type.
    G = hipgraph.Graph(directed=directed)

    cu_M = cudf.from_pandas(M)

    # FIXME: Add the column name in a list to trigger the python renumbering
    # Drop this requirement when 'subgraph_extraction' leverages the CAPI graph
    # which calls renumbering
    G.from_cudf_edgelist(cu_M, source="0", destination="1", edge_attr="weight")

    cu_verts = cudf.Series(verts)
    return hipgraph.induced_subgraph(G, cu_verts)


def nx_call(M, verts, directed=True):
    if directed:
        G = nx.from_pandas_edgelist(
            M, source="0", target="1", create_using=nx.DiGraph()
        )
    else:
        G = nx.from_pandas_edgelist(M, source="0", target="1", create_using=nx.Graph())
    return nx.subgraph(G, verts)


###############################################################################
@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DEFAULT_DATASETS)
def test_subgraph_extraction_DiGraph(graph_file):
    dataset_path = graph_file.get_path()
    M = utils.read_csv_for_nx(dataset_path)
    verts = np.zeros(3, dtype=np.int32)
    verts[0] = 0
    verts[1] = 1
    verts[2] = 17
    cu_sg = hipgraph_call(M, verts, True)[0]
    nx_sg = nx_call(M, verts, True)
    assert compare_edges(cu_sg, nx_sg)


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DEFAULT_DATASETS)
def test_subgraph_extraction_Graph(graph_file):
    dataset_path = graph_file.get_path()
    M = utils.read_csv_for_nx(dataset_path)
    verts = np.zeros(3, dtype=np.int32)
    verts[0] = 0
    verts[1] = 1
    verts[2] = 17
    cu_sg = hipgraph_call(M, verts, False)[0]
    nx_sg = nx_call(M, verts, False)
    assert compare_edges(cu_sg, nx_sg)


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", [DEFAULT_DATASETS[2]])
def test_subgraph_extraction_Graph_nx(graph_file):
    directed = False
    verts = np.zeros(3, dtype=np.int32)
    verts[0] = 0
    verts[1] = 1
    verts[2] = 17
    dataset_path = graph_file.get_path()
    M = utils.read_csv_for_nx(dataset_path)

    if directed:
        G = nx.from_pandas_edgelist(
            M, source="0", target="1", edge_attr="weight", create_using=nx.DiGraph()
        )
    else:
        G = nx.from_pandas_edgelist(
            M, source="0", target="1", edge_attr="weight", create_using=nx.Graph()
        )

    nx_sub = nx.subgraph(G, verts)

    cu_verts = cudf.Series(verts)
    cu_sub = hipgraph.induced_subgraph(G, cu_verts)[0]

    for u, v in cu_sub.edges():
        assert nx_sub.has_edge(u, v)


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DEFAULT_DATASETS)
def test_subgraph_extraction_multi_column(graph_file):
    dataset_path = graph_file.get_path()
    M = utils.read_csv_for_nx(dataset_path)

    cu_M = cudf.DataFrame()
    cu_M["src_0"] = cudf.Series(M["0"])
    cu_M["dst_0"] = cudf.Series(M["1"])
    cu_M["weight"] = cudf.Series(M["weight"])
    cu_M["src_1"] = cu_M["src_0"] + 1000
    cu_M["dst_1"] = cu_M["dst_0"] + 1000
    G1 = hipgraph.Graph()
    G1.from_cudf_edgelist(
        cu_M,
        source=["src_0", "src_1"],
        destination=["dst_0", "dst_1"],
        edge_attr="weight",
    )

    verts = cudf.Series([0, 1, 17])
    verts_G1 = cudf.DataFrame()
    verts_G1["v_0"] = verts
    verts_G1["v_1"] = verts + 1000

    sG1 = hipgraph.induced_subgraph(G1, verts_G1)

    G2 = hipgraph.Graph()
    G2.from_cudf_edgelist(cu_M, source="src_0", destination="dst_0", edge_attr="weight")

    sG2 = hipgraph.induced_subgraph(G2, verts)

    # FIXME: Replace with multi-column view_edge_list()
    edgelist_df = sG1[0].edgelist.edgelist_df
    edgelist_df_res = sG1[0].unrenumber(edgelist_df, "src")
    edgelist_df_res = sG1[0].unrenumber(edgelist_df_res, "dst")
    for i in range(len(edgelist_df_res)):
        assert sG2[0].has_edge(
            edgelist_df_res["0_src"].iloc[i], edgelist_df_res["0_dst"].iloc[i]
        )


# FIXME: the coverage provided by this test could probably be handled by
# another test that also checks using renumber=False
# FIXME: Drop this test as 'subgraph_extraction' requires renumbering
@pytest.mark.sg
@pytest.mark.skip("obsolete")
def test_subgraph_extraction_graph_not_renumbered():
    """
    Ensure subgraph() works with a Graph that has not been renumbered
    """
    gdf = karate.get_edgelist()
    verts = np.array([0, 1, 2], dtype=np.int32)
    sverts = cudf.Series(verts)
    G = hipgraph.Graph()
    G.from_cudf_edgelist(
        gdf, source="src", destination="dst", edge_attr="wgt", renumber=False
    )
    Sg = hipgraph.induced_subgraph(G, sverts)

    assert Sg.number_of_vertices() == 3
    assert Sg.number_of_edges() == 3
