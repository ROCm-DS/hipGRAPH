# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc
import random

import cudf
import dask_cudf
import hipgraph
import hipgraph.dask as dcg
import pytest
from hipgraph.datasets import karate_asymmetric
from hipgraph.structure.symmetrize import symmetrize
from hipgraph.testing import SMALL_DATASETS
from pylibhipgraph.testing.utils import gen_fixture_params_product

# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


IS_DIRECTED = [True, False]


# =============================================================================
# Pytest fixtures
# =============================================================================

datasets = SMALL_DATASETS + [karate_asymmetric]

fixture_params = gen_fixture_params_product(
    (datasets, "graph_file"),
    (IS_DIRECTED, "directed"),
)


def calc_random_walks(G):
    """
    compute random walks

    parameters
    ----------
    G : hipGRAPH.Graph or networkx.Graph
        The graph can be either directed (DiGraph) or undirected (Graph).
        Weights in the graph are ignored.
        Use weight parameter if weights need to be considered
        (currently not supported)

    Returns
    -------
    vertex_paths : cudf.Series or cudf.DataFrame
        Series containing the vertices of edges/paths in the random walk.

    edge_weight_paths: cudf.Series
        Series containing the edge weights of edges represented by the
        returned vertex_paths

    max_path_length : int
        The maximum path length

    start_vertices : list
        Roots for the random walks

    max_depth : int
    """
    k = random.randint(1, 4)
    random_walks_type = "uniform"
    max_depth = random.randint(2, 4)

    start_vertices = G.nodes().compute().sample(k).reset_index(drop=True)

    vertex_paths, edge_weights, max_path_length = dcg.random_walks(
        G, random_walks_type, start_vertices, max_depth
    )

    return (vertex_paths, edge_weights, max_path_length), start_vertices, max_depth


def check_random_walks(G, path_data, seeds, max_depth, df_G=None):
    invalid_edge = 0
    invalid_edge_wgt_path = 0
    invalid_seeds = 0
    next_path_idx = 0
    invalid_edge_wgt_path = 0
    e_wgt_path_idx = 0
    v_paths = path_data[0].compute()
    e_paths = path_data[1].compute()

    max_path_length = path_data[2]
    sizes = max_path_length

    for _ in range(len(seeds)):
        for i in range(next_path_idx, next_path_idx + sizes):
            src, dst = v_paths.iloc[i], v_paths.iloc[i + 1]

            if i == next_path_idx and src not in seeds.values:
                invalid_seeds += 1
                print("[ERR] Invalid seed: " " src {} != src {}".format(src, seeds))

            else:
                # If everything is good proceed to the next part
                # now check the destination

                # find the src out_degree to ensure it effectively has no outgoing edges
                # No need to check for -1 values, move to the next iteration
                if src != -1:
                    src_degree = G.out_degree([src])["degree"].compute()[0]
                    if dst == -1 and src_degree == 0:
                        if e_paths.values[e_wgt_path_idx] != 0:
                            wgt = e_paths.values[e_wgt_path_idx]
                            print(
                                "[ERR] Invalid edge weight path: "
                                "Edge src {} dst {} has wgt 0 "
                                "But got wgt {}".format(src, dst, wgt)
                            )
                            invalid_edge_wgt_path += 1
                    else:
                        exp_edge = df_G.loc[
                            (df_G["src"] == (src)) & (df_G["dst"] == (dst))
                        ].reset_index(drop=True)

                        if len(exp_edge) == 0:
                            print(
                                "[ERR] Invalid edge: "
                                "There is no edge src {} dst {}".format(src, dst)
                            )
                            invalid_edge += 1
                        else:
                            # This is a valid edge, check the edge_wgt_path
                            if e_paths.values[e_wgt_path_idx] != 1:
                                wgt = e_paths.values[e_wgt_path_idx]
                                print(
                                    "[ERR] Invalid edge weight path: "
                                    "Edge src {} dst {} has wgt 1 "
                                    "But got wgt {}".format(src, dst, wgt)
                                )
                                invalid_edge_wgt_path += 1
                else:
                    # v_path: src == -1, dst == -1 => e_wgt_path=0 otherwise ERROR
                    if e_paths.values[e_wgt_path_idx] != 0:
                        wgt = e_paths.values[e_wgt_path_idx]
                        print(
                            "[ERR] Invalid edge weight path: "
                            "Edge src {} dst {} has wgt 0 "
                            "But got wgt {}".format(src, dst, wgt)
                        )
                        invalid_edge_wgt_path += 1

            e_wgt_path_idx += 1
        next_path_idx += sizes + 1

    assert invalid_edge == 0
    assert invalid_seeds == 0
    assert invalid_edge_wgt_path == 0
    assert max_path_length == max_depth


@pytest.fixture(scope="module", params=fixture_params)
def input_graph(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    parameters = dict(zip(("graph_file", "directed"), request.param))
    input_data_path = parameters["graph_file"].get_path()
    directed = parameters["directed"]

    chunksize = dcg.get_chunksize(input_data_path)
    ddf = dask_cudf.read_csv(
        input_data_path,
        blocksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )
    dg = hipgraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(
        ddf,
        source="src",
        destination="dst",
        edge_attr="value",
        renumber=True,
        store_transposed=True,
    )

    return dg


@pytest.mark.mg
def test_dask_mg_random_walks(dask_client, input_graph):
    path_data, seeds, max_depth = calc_random_walks(input_graph)
    df_G = input_graph.input_df.compute().reset_index(drop=True)

    # FIXME: leverages the deprecated symmetrize call
    source_col, dest_col, value_col = symmetrize(
        df_G, "src", "dst", "value", symmetrize=not input_graph.is_directed()
    )

    df = cudf.DataFrame()
    df["src"] = source_col
    df["dst"] = dest_col
    df["value"] = value_col

    check_random_walks(input_graph, path_data, seeds, max_depth, df)
