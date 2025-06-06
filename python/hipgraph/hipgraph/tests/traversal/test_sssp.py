# Copyright (c) 2019-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc

import cudf
import cupy as cp
import cupyx
import hipgraph
import numpy as np
import pandas as pd
import pytest
from cupyx.scipy.sparse import coo_matrix as cp_coo_matrix
from cupyx.scipy.sparse import csc_matrix as cp_csc_matrix
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from hipgraph.testing import (
    SMALL_DATASETS,
    UNDIRECTED_DATASETS,
    get_resultset,
    load_resultset,
    utils,
)
from pylibhipgraph.testing.utils import gen_fixture_params_product
from scipy.sparse import coo_matrix as sp_coo_matrix
from scipy.sparse import csc_matrix as sp_csc_matrix
from scipy.sparse import csr_matrix as sp_csr_matrix

# Map of hipGRAPH input types to the expected output type for hipGRAPH
# connected_components calls.
hipGRAPH_input_output_map = {
    hipgraph.Graph: cudf.DataFrame,
    cp_coo_matrix: tuple,
    cp_csr_matrix: tuple,
    cp_csc_matrix: tuple,
    sp_coo_matrix: tuple,
    sp_csr_matrix: tuple,
    sp_csc_matrix: tuple,
}
cupy_types = [cp_coo_matrix, cp_csr_matrix, cp_csc_matrix]


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


# =============================================================================
# Helper functions
# =============================================================================
def hipgraph_call(gpu_benchmark_callable, input_G_or_matrix, source, edgevals=True):
    """
    Call hipgraph.sssp on input_G_or_matrix, then convert the result to a
    standard format (dictionary of vertex IDs to (distance, predecessor)
    tuples) for easy checking in the test code.
    """
    result = gpu_benchmark_callable(hipgraph.sssp, input_G_or_matrix, source)

    input_type = type(input_G_or_matrix)
    expected_return_type = hipGRAPH_input_output_map[type(input_G_or_matrix)]
    assert type(result) is expected_return_type

    # Convert cudf and pandas: DF of 3 columns: (vertex, distance, predecessor)
    if expected_return_type in [cudf.DataFrame, pd.DataFrame]:
        if expected_return_type is pd.DataFrame:
            result = cudf.from_pandas(result)

        if np.issubdtype(result["distance"].dtype, np.integer):
            max_val = np.iinfo(result["distance"].dtype).max
        else:
            max_val = np.finfo(result["distance"].dtype).max
        verts = result["vertex"].to_numpy()
        dists = result["distance"].to_numpy()
        preds = result["predecessor"].to_numpy()

    # A CuPy/SciPy input means the return value will be a 2-tuple of:
    #   distance: cupy.ndarray
    #      ndarray of shortest distances between source and vertex.
    #   predecessor: cupy.ndarray
    #      ndarray of predecessors of a vertex on the path from source, which
    #      can be used to reconstruct the shortest paths.
    elif expected_return_type is tuple:
        if input_type in cupy_types:
            assert type(result[0]) is cp.ndarray
            assert type(result[1]) is cp.ndarray
        else:
            assert type(result[0]) is np.ndarray
            assert type(result[1]) is np.ndarray

        if np.issubdtype(result[0].dtype, np.integer):
            max_val = np.iinfo(result[0].dtype).max
        else:
            max_val = np.finfo(result[0].dtype).max

        # Get unique verts from input since they are not incuded in output
        if type(input_G_or_matrix) in [
            cp_csr_matrix,
            cp_csc_matrix,
            sp_csr_matrix,
            sp_csc_matrix,
        ]:
            coo = input_G_or_matrix.tocoo(copy=False)
        else:
            coo = input_G_or_matrix
        verts = sorted(set([n.item() for n in coo.col] + [n.item() for n in coo.row]))
        dists = [n.item() for n in result[0]]
        preds = [n.item() for n in result[1]]
        assert len(verts) == len(dists) == len(preds)

    else:
        raise RuntimeError(f"unsupported return type: {expected_return_type}")

    result_dict = dict(zip(verts, zip(dists, preds)))
    return result_dict, max_val


def resultset_call(graph_file, source, load_results, edgevals=True):
    dataset_path = graph_file.get_path()
    dataset_name = graph_file.metadata["name"]

    if edgevals is False:
        # FIXME: no test coverage if edgevals is False, this assertion is never reached
        assert False
        golden_paths = get_resultset(
            resultset_name="traversal",
            algo="single_source_shortest_path_length",
            graph_dataset=dataset_name,
            graph_directed=str(True),
            source=str(source),
        )
    else:
        # FIXME: The golden results (nx) below doesn't return accurate results as it
        # seems to not support 'weights'. It matches hipGRAPH result only if the weight
        # column is 1s.
        golden_paths = get_resultset(
            resultset_name="traversal",
            algo="single_source_dijkstra_path_length",
            graph_dataset=dataset_name,
            graph_directed=str(True),
            source=str(source),
        )
    golden_paths = cudf.Series(
        golden_paths.distance.values, index=golden_paths.vertex
    ).to_dict()

    G = graph_file.get_graph(
        create_using=hipgraph.Graph(directed=True), ignore_weights=not edgevals
    )

    return (G, dataset_path, graph_file, source, golden_paths)


# =============================================================================
# Pytest fixtures
# =============================================================================

# Call gen_fixture_params_product() to calculate the cartesian product of
# multiple lists of params. This is required since parameterized fixtures do
# not do this automatically (unlike multiply-parameterized tests). The 2nd
# item in the tuple is a label for the param value used when displaying the
# full test name.
# FIXME: tests with datasets like 'netscience' which has a weight column different
# than than 1's fail because it looks like netwokX doesn't consider weights during
# the computation.
DATASETS = [pytest.param(d) for d in SMALL_DATASETS]
SOURCES = [pytest.param(1)]
fixture_params = gen_fixture_params_product((DATASETS, "ds"), (SOURCES, "src"))
fixture_params_single_dataset = gen_fixture_params_product(
    ([DATASETS[0]], "ds"), (SOURCES, "src")
)


# Fixture that loads all golden results necessary to run hipgraph tests if the
# tests are not already present in the designated results directory. Most of the
# time, this will only check if the module-specific mapping file exists.
@pytest.fixture(scope="module")
def load_traversal_results():
    load_resultset(
        "traversal", "https://data.rapids.ai/hipgraph/results/resultsets.tar.gz"
    )


@pytest.fixture(scope="module", params=fixture_params)
def dataset_source_goldenresults(request):
    # request.param is a tuple of params from fixture_params. When expanded
    # with *, will be passed to resultset_call() as args (graph_file, source)
    return resultset_call(*(request.param), load_traversal_results)


@pytest.fixture(scope="module", params=fixture_params_single_dataset)
def single_dataset_source_goldenresults(request):
    return resultset_call(*(request.param), load_traversal_results)


@pytest.fixture(scope="module", params=fixture_params)
def dataset_source_goldenresults_weighted(request):
    return resultset_call(*(request.param), load_traversal_results, edgevals=True)


@pytest.fixture(scope="module", params=fixture_params_single_dataset)
def single_dataset_source_goldenresults_weighted(request):
    return resultset_call(*(request.param), load_traversal_results, edgevals=True)


# =============================================================================
# Tests
# =============================================================================
@pytest.mark.sg
@pytest.mark.parametrize("hipgraph_input_type", utils.HIPGRAPH_DIR_INPUT_TYPES)
def test_sssp(gpubenchmark, dataset_source_goldenresults, hipgraph_input_type):
    # Extract the params generated from the fixture
    (G, dataset_path, _, source, golden_paths) = dataset_source_goldenresults

    if not isinstance(hipgraph_input_type, hipgraph.Graph):
        input_G_or_matrix = utils.create_obj_from_csv(
            dataset_path, hipgraph_input_type, edgevals=True
        )
    else:
        input_G_or_matrix = G
    cu_paths, max_val = hipgraph_call(gpubenchmark, input_G_or_matrix, source)

    # Calculating mismatch
    err = 0
    for vid in cu_paths:
        # Validate vertices that are reachable
        # NOTE : If distance type is float64 then cu_paths[vid][0]
        # should be compared against np.finfo(np.float64).max)
        if cu_paths[vid][0] != max_val:
            if cu_paths[vid][0] != golden_paths[vid]:
                err = err + 1
            # check pred dist + 1 = current dist (since unweighted)
            pred = cu_paths[vid][1]
            if vid != source and cu_paths[pred][0] + 1 != cu_paths[vid][0]:
                err = err + 1
        else:
            if vid in golden_paths.keys():
                err = err + 1

    assert err == 0


@pytest.mark.sg
@pytest.mark.parametrize("hipgraph_input_type", utils.HIPGRAPH_DIR_INPUT_TYPES)
def test_sssp_invalid_start(
    gpubenchmark, dataset_source_goldenresults, hipgraph_input_type
):
    (G, _, _, source, _) = dataset_source_goldenresults
    el = G.view_edge_list()

    newval = max(el.src.max(), el.dst.max()) + 1
    source = newval

    with pytest.raises(ValueError):
        hipgraph_call(gpubenchmark, G, source)


@pytest.mark.sg
@pytest.mark.parametrize("hipgraph_input_type", utils.MATRIX_INPUT_TYPES)
def test_sssp_nonnative_inputs_matrix(
    gpubenchmark, single_dataset_source_goldenresults, hipgraph_input_type
):
    test_sssp(gpubenchmark, single_dataset_source_goldenresults, hipgraph_input_type)


@pytest.mark.sg
@pytest.mark.parametrize("directed", [True, False])
def test_sssp_nonnative_inputs_graph(single_dataset_source_goldenresults, directed):
    (_, _, graph_file, source, golden_paths) = single_dataset_source_goldenresults
    dataset_name = graph_file.metadata["name"]
    result = get_resultset(
        resultset_name="traversal",
        algo="sssp_nonnative",
        graph_dataset=dataset_name,
        graph_directed=str(directed),
        source=str(source),
    )
    if np.issubdtype(result["distance"].dtype, np.integer):
        max_val = np.iinfo(result["distance"].dtype).max
    else:
        max_val = np.finfo(result["distance"].dtype).max
    verts = result["vertex"].to_numpy()
    dists = result["distance"].to_numpy()
    preds = result["predecessor"].to_numpy()
    cu_paths = dict(zip(verts, zip(dists, preds)))

    # Calculating mismatch
    err = 0
    for vid in cu_paths:
        # Validate vertices that are reachable
        # NOTE : If distance type is float64 then cu_paths[vid][0]
        # should be compared against np.finfo(np.float64).max)
        if cu_paths[vid][0] != max_val:
            if cu_paths[vid][0] != golden_paths[vid]:
                err = err + 1
            # check pred dist + 1 = current dist (since unweighted)
            pred = cu_paths[vid][1]
            if vid != source and cu_paths[pred][0] + 1 != cu_paths[vid][0]:
                err = err + 1
        else:
            if vid in golden_paths.keys():
                err = err + 1

    assert err == 0


@pytest.mark.sg
@pytest.mark.parametrize("hipgraph_input_type", utils.HIPGRAPH_DIR_INPUT_TYPES)
def test_sssp_edgevals(
    gpubenchmark, dataset_source_goldenresults_weighted, hipgraph_input_type
):
    # Extract the params generated from the fixture
    (G, _, _, source, golden_paths) = dataset_source_goldenresults_weighted
    input_G_or_matrix = G

    cu_paths, max_val = hipgraph_call(
        gpubenchmark, input_G_or_matrix, source, edgevals=True
    )

    # Calculating mismatch
    err = 0
    for vid in cu_paths:
        # Validate vertices that are reachable
        # NOTE : If distance type is float64 then cu_paths[vid][0]
        # should be compared against np.finfo(np.float64).max)
        distances = hipgraph.sssp(G, source=vid)
        if cu_paths[vid][0] != max_val:
            if cu_paths[vid][0] != golden_paths[vid]:
                err = err + 1
            # check pred dist + edge_weight = current dist
            if vid != source:
                pred = cu_paths[vid][1]
                if G.has_edge(pred, vid):
                    edge_weight = distances[distances["vertex"] == pred].iloc[0, 0]
                if cu_paths[pred][0] + edge_weight != cu_paths[vid][0]:
                    err = err + 1
        else:
            if vid in golden_paths.keys():
                err = err + 1
    assert err == 0


@pytest.mark.sg
@pytest.mark.parametrize(
    "hipgraph_input_type", utils.NX_DIR_INPUT_TYPES + utils.MATRIX_INPUT_TYPES
)
def test_sssp_edgevals_nonnative_inputs(
    gpubenchmark, single_dataset_source_goldenresults_weighted, hipgraph_input_type
):
    test_sssp_edgevals(
        gpubenchmark, single_dataset_source_goldenresults_weighted, hipgraph_input_type
    )


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DATASETS)
@pytest.mark.parametrize("source", SOURCES)
def test_sssp_data_type_conversion(graph_file, source):
    dataset_path = graph_file.get_path()
    dataset_name = graph_file.metadata["name"]
    cu_M = utils.read_csv_file(dataset_path)

    # hipgraph call with int32 weights
    cu_M["2"] = cu_M["2"].astype(np.int32)
    G = hipgraph.Graph(directed=True)
    G.from_cudf_edgelist(cu_M, source="0", destination="1", edge_attr="2")
    # assert hipgraph weights is int32
    assert G.edgelist.edgelist_df["weights"].dtype == np.int32
    df = hipgraph.sssp(G, source)
    max_val = np.finfo(df["distance"].dtype).max
    verts_np = df["vertex"].to_numpy()
    dist_np = df["distance"].to_numpy()
    pred_np = df["predecessor"].to_numpy()
    cu_paths = dict(zip(verts_np, zip(dist_np, pred_np)))
    golden_paths = get_resultset(
        resultset_name="traversal",
        algo="single_source_dijkstra_path_length",
        graph_dataset=dataset_name,
        graph_directed=str(True),
        source=str(source),
        test="data_type_conversion",
    )
    golden_paths = cudf.Series(
        golden_paths.distance.values, index=golden_paths.vertex
    ).to_dict()

    # Calculating mismatch
    err = 0
    for vid in cu_paths:
        # Validate vertices that are reachable
        # NOTE : If distance type is float64 then cu_paths[vid][0]
        # should be compared against np.finfo(np.float64).max)
        distances = hipgraph.sssp(G, source=vid)
        if cu_paths[vid][0] != max_val:
            if cu_paths[vid][0] != golden_paths[vid]:
                err = err + 1
            # check pred dist + edge_weight = current dist
            if vid != source:
                pred = cu_paths[vid][1]
                if G.has_edge(pred, vid):
                    edge_weight = distances[distances["vertex"] == pred].iloc[0, 0]
                if cu_paths[pred][0] + edge_weight != cu_paths[vid][0]:
                    err = err + 1
        else:
            if vid in golden_paths.keys():
                err = err + 1

    assert err == 0


@pytest.mark.sg
def test_sssp_golden_edge_attr(load_traversal_results):
    df = get_resultset(
        resultset_name="traversal", algo="sssp_nonnative", test="network_edge_attr"
    )
    df = df.set_index("vertex")
    assert df.loc[0, "distance"] == 0
    assert df.loc[1, "distance"] == 10
    assert df.loc[2, "distance"] == 30


@pytest.mark.sg
def test_scipy_api_compat():
    graph_file = SMALL_DATASETS[0]
    dataset_path = graph_file.get_path()
    input_hipgraph_graph = graph_file.get_graph()
    input_coo_matrix = utils.create_obj_from_csv(
        dataset_path, cp_coo_matrix, edgevals=True
    )

    # Ensure scipy-only options are rejected for hipgraph inputs
    with pytest.raises(TypeError):
        hipgraph.shortest_path(input_hipgraph_graph, source=0, directed=False)
    with pytest.raises(TypeError):
        hipgraph.shortest_path(input_hipgraph_graph, source=0, unweighted=False)
    with pytest.raises(TypeError):
        hipgraph.shortest_path(input_hipgraph_graph, source=0, overwrite=False)
    with pytest.raises(TypeError):
        hipgraph.shortest_path(
            input_hipgraph_graph, source=0, return_predecessors=False
        )

    # Ensure hipgraph-compatible options work as expected
    # cannot set both source and indices, but must set one
    with pytest.raises(TypeError):
        hipgraph.shortest_path(input_hipgraph_graph, source=0, indices=0)
    with pytest.raises(TypeError):
        hipgraph.shortest_path(input_hipgraph_graph)
    with pytest.raises(ValueError):
        hipgraph.shortest_path(input_hipgraph_graph, source=0, method="BF")
    hipgraph.shortest_path(input_hipgraph_graph, indices=0)
    with pytest.raises(ValueError):
        hipgraph.shortest_path(input_hipgraph_graph, indices=[0, 1, 2])
    hipgraph.shortest_path(input_hipgraph_graph, source=0, method="auto")

    # Ensure SciPy options for matrix inputs work as expected
    # cannot set both source and indices, but must set one
    with pytest.raises(TypeError):
        hipgraph.shortest_path(input_coo_matrix, source=0, indices=0)
    with pytest.raises(TypeError):
        hipgraph.shortest_path(input_coo_matrix)
    with pytest.raises(ValueError):
        hipgraph.shortest_path(input_coo_matrix, source=0, method="BF")
    hipgraph.shortest_path(input_coo_matrix, source=0, method="auto")

    with pytest.raises(ValueError):
        hipgraph.shortest_path(input_coo_matrix, source=0, directed=3)
    hipgraph.shortest_path(input_coo_matrix, source=0, directed=True)
    hipgraph.shortest_path(input_coo_matrix, source=0, directed=False)

    with pytest.raises(ValueError):
        hipgraph.shortest_path(input_coo_matrix, source=0, return_predecessors=3)
    (distances, preds) = hipgraph.shortest_path(
        input_coo_matrix, source=0, return_predecessors=True
    )
    distances = hipgraph.shortest_path(
        input_coo_matrix, source=0, return_predecessors=False
    )
    assert type(distances) is not tuple

    with pytest.raises(ValueError):
        hipgraph.shortest_path(input_coo_matrix, source=0, unweighted=False)
    hipgraph.shortest_path(input_coo_matrix, source=0, unweighted=True)

    with pytest.raises(ValueError):
        hipgraph.shortest_path(input_coo_matrix, source=0, overwrite=True)
    hipgraph.shortest_path(input_coo_matrix, source=0, overwrite=False)

    with pytest.raises(ValueError):
        hipgraph.shortest_path(input_coo_matrix, indices=[0, 1, 2])
    hipgraph.shortest_path(input_coo_matrix, indices=0)


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", UNDIRECTED_DATASETS)
def test_sssp_csr_graph(graph_file):
    df = graph_file.get_edgelist()

    M = cupyx.scipy.sparse.coo_matrix(
        (df["wgt"].to_cupy(), (df["src"].to_cupy(), df["dst"].to_cupy()))
    )
    M = M.tocsr()

    offsets = cudf.Series(M.indptr)
    indices = cudf.Series(M.indices)
    weights = cudf.Series(M.data)
    G_csr = hipgraph.Graph()
    G_coo = graph_file.get_graph()

    source = G_coo.select_random_vertices(num_vertices=1)[0]

    print("source = ", source)

    G_csr.from_cudf_adjlist(offsets, indices, weights)

    result_csr = hipgraph.sssp(G_csr, source)
    result_coo = hipgraph.sssp(G_coo, source)

    result_csr = result_csr.sort_values("vertex").reset_index(drop=True)
    result_sssp = (
        result_coo.sort_values("vertex")
        .reset_index(drop=True)
        .rename(columns={"distance": "distance_coo", "predecessor": "predecessor_coo"})
    )
    result_sssp["distance_csr"] = result_csr["distance"]
    result_sssp["predecessor_csr"] = result_csr["predecessor"]

    distance_diffs = result_sssp.query("distance_csr != distance_coo")
    predecessor_diffs = result_sssp.query("predecessor_csr != predecessor_coo")

    assert len(distance_diffs) == 0
    assert len(predecessor_diffs) == 0


@pytest.mark.sg
def test_sssp_unweighted_graph():
    karate = SMALL_DATASETS[0]
    G = karate.get_graph(ignore_weights=True)

    error_msg = (
        "'SSSP' requires the input graph to be weighted."
        "'BFS' should be used instead of 'SSSP' for unweighted graphs."
    )

    with pytest.raises(RuntimeError, match=error_msg):
        hipgraph.sssp(G, 1)
