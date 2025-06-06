# Copyright (c) 2020-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc

import cudf
import hipgraph
import numpy as np
import pytest
from hipgraph.datasets import karate
from hipgraph.testing import utils


@pytest.mark.sg
def test_bfs_paths():
    with pytest.raises(ValueError) as ErrorMsg:
        gc.collect()
        G = karate.get_graph()

        # run BFS starting at vertex 17
        df = hipgraph.bfs(G, 16)

        # Get the path to vertex 1
        p_df = hipgraph.utils.get_traversed_path(df, 0)

        assert len(p_df) == 3

        # Get path to vertex 0 - which is not in graph
        p_df = hipgraph.utils.get_traversed_path(df, 100)

        assert "not in the result set" in str(ErrorMsg)


@pytest.mark.sg
def test_bfs_paths_array():
    with pytest.raises(ValueError) as ErrorMsg:
        gc.collect()
        G = karate.get_graph()

        # run BFS starting at vertex 17
        df = hipgraph.bfs(G, 16)

        # Get the path to vertex 1
        answer = hipgraph.utils.get_traversed_path_list(df, 0)

        assert len(answer) == 3

        # Get path to vertex 0 - which is not in graph
        answer = hipgraph.utils.get_traversed_path_list(df, 100)

        assert "not in the result set" in str(ErrorMsg)


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", utils.DATASETS)
@pytest.mark.skip(reason="Skipping large tests")
def test_get_traversed_cost(graph_file):
    cu_M = utils.read_csv_file(graph_file)

    noise = cudf.Series(np.random.randint(10, size=(cu_M.shape[0])))
    cu_M["info"] = cu_M["2"] + noise

    G = hipgraph.Graph()
    G.from_cudf_edgelist(cu_M, source="0", destination="1", edge_attr="info")

    # run SSSP starting at vertex 17
    df = hipgraph.sssp(G, 16)

    answer = hipgraph.utilities.path_retrieval.get_traversed_cost(
        df, 16, cu_M["0"], cu_M["1"], cu_M["info"]
    )

    df = df.sort_values(by="vertex").reset_index()
    answer = answer.sort_values(by="vertex").reset_index()

    assert df.shape[0] == answer.shape[0]
    assert np.allclose(df["distance"], answer["info"])
