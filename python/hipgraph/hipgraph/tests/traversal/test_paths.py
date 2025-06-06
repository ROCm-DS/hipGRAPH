# Copyright (c) 2019-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import math
import sys
from tempfile import NamedTemporaryFile

import cudf
import cupy
import hipgraph
import numpy as np
import pytest
from cudf.testing.testing import assert_series_equal
from cupyx.scipy.sparse import coo_matrix as cupy_coo_matrix
from hipgraph.testing import get_resultset, load_resultset

CONNECTED_GRAPH = """1,5,3
1,4,1
1,2,1
1,6,2
1,7,2
4,5,1
2,3,1
7,6,2
"""

DISCONNECTED_GRAPH = CONNECTED_GRAPH + "8,9,4"


# Single value or callable golden results are not added as a Resultset
paths_golden_results = {
    "shortest_path_length_1_1": 0,
    "shortest_path_length_1_5": 2.0,
    "shortest_path_length_1_3": 2.0,
    "shortest_path_length_1_6": 2.0,
    "shortest_path_length_-1_1": ValueError,
    "shortest_path_length_1_10": ValueError,
    "shortest_path_length_0_42": ValueError,
    "shortest_path_length_1_8": 3.4028235e38,
}


# Fixture that loads all golden results necessary to run hipgraph tests if the
# tests are not already present in the designated results directory. Most of the
# time, this will only check if the module-specific mapping file exists.
@pytest.fixture(scope="module")
def load_traversal_results():
    load_resultset(
        "traversal", "https://data.rapids.ai/hipgraph/results/resultsets.tar.gz"
    )


@pytest.fixture
def graphs(request):
    with NamedTemporaryFile(mode="w+", suffix=".csv") as graph_tf:
        graph_tf.writelines(request.param)
        graph_tf.seek(0)

        cudf_df = cudf.read_csv(
            graph_tf.name,
            names=["src", "dst", "data"],
            delimiter=",",
            dtype=["int32", "int32", "float64"],
        )
        hipgraph_G = hipgraph.Graph()
        hipgraph_G.from_cudf_edgelist(
            cudf_df, source="src", destination="dst", edge_attr="data"
        )

        # construct cupy coo_matrix graph
        i = []
        j = []
        weights = []
        for index in range(cudf_df.shape[0]):
            vertex1 = cudf_df.iloc[index]["src"]
            vertex2 = cudf_df.iloc[index]["dst"]
            weight = cudf_df.iloc[index]["data"]
            i += [vertex1, vertex2]
            j += [vertex2, vertex1]
            weights += [weight, weight]
        i = cupy.array(i)
        j = cupy.array(j)
        weights = cupy.array(weights)
        largest_vertex = max(cupy.amax(i), cupy.amax(j))
        cupy_df = cupy_coo_matrix(
            (weights, (i, j)), shape=(largest_vertex + 1, largest_vertex + 1)
        )

        yield hipgraph_G, cupy_df


@pytest.mark.sg
@pytest.mark.parametrize("graphs", [CONNECTED_GRAPH], indirect=True)
def test_connected_graph_shortest_path_length(graphs):
    hipgraph_G, cupy_df = graphs

    path_1_to_1_length = hipgraph.shortest_path_length(hipgraph_G, 1, 1)
    # FIXME: aren't the first two assertions in each batch redundant?
    assert path_1_to_1_length == 0.0
    assert path_1_to_1_length == paths_golden_results["shortest_path_length_1_1"]
    assert path_1_to_1_length == hipgraph.shortest_path_length(cupy_df, 1, 1)

    path_1_to_5_length = hipgraph.shortest_path_length(hipgraph_G, 1, 5)
    assert path_1_to_5_length == 2.0
    assert path_1_to_5_length == paths_golden_results["shortest_path_length_1_5"]
    assert path_1_to_5_length == hipgraph.shortest_path_length(cupy_df, 1, 5)

    path_1_to_3_length = hipgraph.shortest_path_length(hipgraph_G, 1, 3)
    assert path_1_to_3_length == 2.0
    assert path_1_to_3_length == paths_golden_results["shortest_path_length_1_3"]
    assert path_1_to_3_length == hipgraph.shortest_path_length(cupy_df, 1, 3)

    path_1_to_6_length = hipgraph.shortest_path_length(hipgraph_G, 1, 6)
    assert path_1_to_6_length == 2.0
    assert path_1_to_6_length == paths_golden_results["shortest_path_length_1_6"]
    assert path_1_to_6_length == hipgraph.shortest_path_length(cupy_df, 1, 6)


@pytest.mark.sg
@pytest.mark.parametrize("graphs", [CONNECTED_GRAPH], indirect=True)
def test_shortest_path_length_invalid_source(graphs):
    hipgraph_G, cupy_df = graphs

    with pytest.raises(ValueError):
        hipgraph.shortest_path_length(hipgraph_G, -1, 1)

    result = paths_golden_results["shortest_path_length_-1_1"]
    if callable(result):
        with pytest.raises(ValueError):
            raise result()

    with pytest.raises(ValueError):
        hipgraph.shortest_path_length(cupy_df, -1, 1)


@pytest.mark.sg
@pytest.mark.parametrize("graphs", [DISCONNECTED_GRAPH], indirect=True)
def test_shortest_path_length_invalid_target(graphs):
    hipgraph_G, cupy_df = graphs

    with pytest.raises(ValueError):
        hipgraph.shortest_path_length(hipgraph_G, 1, 10)

    result = paths_golden_results["shortest_path_length_1_10"]
    if callable(result):
        with pytest.raises(ValueError):
            raise result()

    with pytest.raises(ValueError):
        hipgraph.shortest_path_length(cupy_df, 1, 10)


@pytest.mark.sg
@pytest.mark.parametrize("graphs", [CONNECTED_GRAPH], indirect=True)
def test_shortest_path_length_invalid_vertexes(graphs):
    hipgraph_G, cupy_df = graphs

    with pytest.raises(ValueError):
        hipgraph.shortest_path_length(hipgraph_G, 0, 42)

    result = paths_golden_results["shortest_path_length_0_42"]
    if callable(result):
        with pytest.raises(ValueError):
            raise result()

    with pytest.raises(ValueError):
        hipgraph.shortest_path_length(cupy_df, 0, 42)


@pytest.mark.sg
@pytest.mark.parametrize("graphs", [DISCONNECTED_GRAPH], indirect=True)
def test_shortest_path_length_no_path(graphs):
    hipgraph_G, cupy_df = graphs

    # FIXME: In case there is no path between two vertices, the
    # result can be either the max of float32 or float64
    max_float_32 = (2 - math.pow(2, -23)) * math.pow(2, 127)

    path_1_to_8 = hipgraph.shortest_path_length(hipgraph_G, 1, 8)
    assert path_1_to_8 == sys.float_info.max

    golden_path_1_to_8 = paths_golden_results["shortest_path_length_1_8"]
    golden_path_1_to_8 = np.float32(golden_path_1_to_8)
    assert golden_path_1_to_8 in [
        max_float_32,
        path_1_to_8,
    ]
    assert path_1_to_8 == hipgraph.shortest_path_length(cupy_df, 1, 8)


@pytest.mark.sg
@pytest.mark.parametrize("graphs", [DISCONNECTED_GRAPH], indirect=True)
def test_shortest_path_length_no_target(graphs, load_traversal_results):
    hipgraph_G, cupy_df = graphs

    hipgraph_path_1_to_all = (
        hipgraph.shortest_path_length(hipgraph_G, 1)
        .sort_values("vertex")
        .reset_index(drop=True)
    )
    golden_path_1_to_all = get_resultset(
        resultset_name="traversal",
        algo="shortest_path_length",
        graph_dataset="DISCONNECTED",
        graph_directed=str(True),
        source="1",
        weight="weight",
    )
    cupy_path_1_to_all = hipgraph.shortest_path_length(cupy_df, 1)

    # Cast networkx graph on hipgraph vertex column type from str to int.
    # SSSP preserves vertex type, convert for comparison
    assert_series_equal(
        hipgraph_path_1_to_all["distance"],
        cupy_path_1_to_all["distance"],
        check_names=False,
        check_dtype=False,
    )

    # results for vertex 8 and 9 are not returned
    assert hipgraph_path_1_to_all.shape[0] == len(golden_path_1_to_all) + 2
    for index in range(hipgraph_path_1_to_all.shape[0]):

        vertex = hipgraph_path_1_to_all["vertex"][index].item()
        distance = hipgraph_path_1_to_all["distance"][index].item()

        # verify hipgraph against networkx
        if vertex in {8, 9}:
            # Networkx does not return distances for these vertexes.
            assert distance == sys.float_info.max
        else:
            assert (
                distance
                == golden_path_1_to_all.loc[
                    golden_path_1_to_all.vertex == vertex
                ].distance.iloc[0]
            )
