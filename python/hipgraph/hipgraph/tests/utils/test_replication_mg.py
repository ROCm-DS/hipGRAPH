# Copyright (c) 2020-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc

import cudf
import hipgraph
import hipgraph.dask.structure.replication as replication
import hipgraph.testing.utils as utils
import pytest
from cudf.testing import assert_frame_equal, assert_series_equal
from hipgraph.dask.common.mg_utils import is_single_gpu

DATASETS_OPTIONS = utils.DATASETS_SMALL
DIRECTED_GRAPH_OPTIONS = [False, True]


@pytest.mark.mg
@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize(
    "input_data_path", DATASETS_OPTIONS, ids=[f"dataset={d}" for d in DATASETS_OPTIONS]
)
def test_replicate_cudf_dataframe_with_weights(input_data_path, dask_client):
    gc.collect()
    df = cudf.read_csv(
        input_data_path,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )
    worker_to_futures = replication.replicate_cudf_dataframe(df)
    for worker in worker_to_futures:
        replicated_df = worker_to_futures[worker].result()
        assert_frame_equal(df, replicated_df)


@pytest.mark.mg
@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize(
    "input_data_path", DATASETS_OPTIONS, ids=[f"dataset={d}" for d in DATASETS_OPTIONS]
)
def test_replicate_cudf_dataframe_no_weights(input_data_path, dask_client):
    gc.collect()
    df = cudf.read_csv(
        input_data_path,
        delimiter=" ",
        names=["src", "dst"],
        dtype=["int32", "int32"],
    )
    worker_to_futures = replication.replicate_cudf_dataframe(df)
    for worker in worker_to_futures:
        replicated_df = worker_to_futures[worker].result()
        assert_frame_equal(df, replicated_df)


@pytest.mark.mg
@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize(
    "input_data_path", DATASETS_OPTIONS, ids=[f"dataset={d}" for d in DATASETS_OPTIONS]
)
def test_replicate_cudf_series(input_data_path, dask_client):
    gc.collect()
    df = cudf.read_csv(
        input_data_path,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )
    for column in df.columns.values:
        series = df[column]
        worker_to_futures = replication.replicate_cudf_series(series)
        for worker in worker_to_futures:
            replicated_series = worker_to_futures[worker].result()
            assert_series_equal(series, replicated_series, check_names=False)
        # FIXME: If we do not clear this dictionary, when comparing
        # results for the 2nd column, one of the workers still
        # has a value from the 1st column
        worker_to_futures = {}


@pytest.mark.skip(reason="no way of currently testing this")
@pytest.mark.parametrize(
    "graph_file", DATASETS_OPTIONS, ids=[f"dataset={d}" for d in DATASETS_OPTIONS]
)
@pytest.mark.mg
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_enable_batch_no_context(graph_file, directed):
    gc.collect()
    G = utils.generate_hipgraph_graph_from_file(graph_file, directed)
    assert G.batch_enabled is False, "Internal property should be False"
    with pytest.raises(Exception):
        G.enable_batch()


@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize(
    "graph_file", DATASETS_OPTIONS, ids=[f"dataset={d}" for d in DATASETS_OPTIONS]
)
@pytest.mark.mg
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_enable_batch_no_context_view_adj(graph_file, directed, dask_client):
    gc.collect()
    G = utils.generate_hipgraph_graph_from_file(graph_file, directed)
    assert G.batch_enabled is False, "Internal property should be False"
    G.view_adj_list()


@pytest.mark.mg
@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize(
    "graph_file", DATASETS_OPTIONS, ids=[f"dataset={d}" for d in DATASETS_OPTIONS]
)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_enable_batch_context_then_views(graph_file, directed, dask_client):
    gc.collect()
    G = utils.generate_hipgraph_graph_from_file(graph_file, directed)
    assert G.batch_enabled is False, "Internal property should be False"
    G.enable_batch()
    assert G.batch_enabled is True, "Internal property should be True"
    assert G.batch_edgelists is not None, (
        "The graph should have " "been created with an " "edgelist"
    )
    assert G.batch_adjlists is None
    G.view_adj_list()
    assert G.batch_adjlists is not None

    assert G.batch_transposed_adjlists is None
    G.view_transposed_adj_list()
    assert G.batch_transposed_adjlists is not None


@pytest.mark.mg
@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize(
    "graph_file", DATASETS_OPTIONS, ids=[f"dataset={d}" for d in DATASETS_OPTIONS]
)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_enable_batch_view_then_context(graph_file, directed, dask_client):
    gc.collect()
    G = utils.generate_hipgraph_graph_from_file(graph_file, directed)

    assert G.batch_adjlists is None
    G.view_adj_list()
    assert G.batch_adjlists is None

    assert G.batch_transposed_adjlists is None
    G.view_transposed_adj_list()
    assert G.batch_transposed_adjlists is None

    assert G.batch_enabled is False, "Internal property should be False"
    G.enable_batch()
    assert G.batch_enabled is True, "Internal property should be True"
    assert G.batch_edgelists is not None, (
        "The graph should have " "been created with an " "edgelist"
    )
    assert G.batch_adjlists is not None
    assert G.batch_transposed_adjlists is not None


@pytest.mark.mg
@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize(
    "graph_file", DATASETS_OPTIONS, ids=[f"dataset={d}" for d in DATASETS_OPTIONS]
)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_enable_batch_context_no_context_views(graph_file, directed, dask_client):
    gc.collect()
    G = utils.generate_hipgraph_graph_from_file(graph_file, directed)
    assert G.batch_enabled is False, "Internal property should be False"
    G.enable_batch()
    assert G.batch_enabled is True, "Internal property should be True"
    assert G.batch_edgelists is not None, (
        "The graph should have " "been created with an " "edgelist"
    )
    G.view_edge_list()
    G.view_adj_list()
    G.view_transposed_adj_list()


@pytest.mark.mg
@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize(
    "graph_file", DATASETS_OPTIONS, ids=[f"dataset={d}" for d in DATASETS_OPTIONS]
)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_enable_batch_edgelist_replication(graph_file, directed, dask_client):
    gc.collect()
    G = utils.generate_hipgraph_graph_from_file(graph_file, directed)
    G.enable_batch()
    df = G.edgelist.edgelist_df
    for i in range(G.batch_edgelists.npartitions):
        replicated_df = G.batch_edgelists.get_partition(i).compute()
        assert_frame_equal(df, replicated_df)


@pytest.mark.mg
@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize(
    "graph_file", DATASETS_OPTIONS, ids=[f"dataset={d}" for d in DATASETS_OPTIONS]
)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_enable_batch_adjlist_replication_weights(graph_file, directed, dask_client):
    gc.collect()
    df = cudf.read_csv(
        graph_file,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )
    G = hipgraph.Graph(directed=directed)
    G.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="value")
    G.enable_batch()
    G.view_adj_list()
    adjlist = G.adjlist
    offsets = adjlist.offsets
    indices = adjlist.indices
    weights = adjlist.weights
    for worker in G.batch_adjlists:
        (rep_offsets, rep_indices, rep_weights) = G.batch_adjlists[worker]
        assert_series_equal(offsets, rep_offsets.result(), check_names=False)
        assert_series_equal(indices, rep_indices.result(), check_names=False)
        assert_series_equal(weights, rep_weights.result(), check_names=False)


@pytest.mark.mg
@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize(
    "graph_file", DATASETS_OPTIONS, ids=[f"dataset={d}" for d in DATASETS_OPTIONS]
)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_enable_batch_adjlist_replication_no_weights(graph_file, directed, dask_client):
    gc.collect()
    df = cudf.read_csv(
        graph_file,
        delimiter=" ",
        names=["src", "dst"],
        dtype=["int32", "int32"],
    )
    G = hipgraph.Graph(directed=directed)
    G.from_cudf_edgelist(df, source="src", destination="dst")
    G.enable_batch()
    G.view_adj_list()
    adjlist = G.adjlist
    offsets = adjlist.offsets
    indices = adjlist.indices
    weights = adjlist.weights
    for worker in G.batch_adjlists:
        (rep_offsets, rep_indices, rep_weights) = G.batch_adjlists[worker]
        assert_series_equal(offsets, rep_offsets.result(), check_names=False)
        assert_series_equal(indices, rep_indices.result(), check_names=False)
        assert weights is None and rep_weights is None
