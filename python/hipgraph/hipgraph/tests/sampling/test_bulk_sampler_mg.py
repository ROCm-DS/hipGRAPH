# Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import os
import re
import shutil

import cudf
import cupy
import dask_cudf
import hipgraph
import pytest
from hipgraph.datasets import email_Eu_core, karate
from hipgraph.gnn import BulkSampler
from hipgraph.utilities.utils import create_directory_with_overwrite


@pytest.mark.mg
def test_bulk_sampler_simple(dask_client, scratch_dir):
    el = karate.get_edgelist().reset_index().rename(columns={"index": "eid"})
    el["eid"] = el["eid"].astype("int32")
    el["etp"] = cupy.int32(0)

    G = hipgraph.Graph(directed=True)
    G.from_dask_cudf_edgelist(
        dask_cudf.from_cudf(el, npartitions=2),
        source="src",
        destination="dst",
        edge_attr=["wgt", "eid", "etp"],
    )

    samples_path = os.path.join(scratch_dir, "mg_test_bulk_sampler_simple")
    create_directory_with_overwrite(samples_path)

    bs = BulkSampler(
        batch_size=2,
        output_path=samples_path,
        graph=G,
        fanout_vals=[2, 2],
        with_replacement=False,
    )

    batches = dask_cudf.from_cudf(
        cudf.DataFrame(
            {
                "start": cudf.Series([0, 5, 10, 15], dtype="int32"),
                "batch": cudf.Series([0, 0, 1, 1], dtype="int32"),
            }
        ),
        npartitions=2,
    )

    bs.add_batches(batches, start_col_name="start", batch_col_name="batch")
    bs.flush()

    recovered_samples = cudf.read_parquet(samples_path)
    assert "map" not in recovered_samples.columns

    for b in batches["batch"].unique().compute().values_host.tolist():
        assert b in recovered_samples["batch_id"].values_host.tolist()

    shutil.rmtree(samples_path)


@pytest.mark.mg
def test_bulk_sampler_mg_graph_sg_input(dask_client, scratch_dir):
    el = karate.get_edgelist().reset_index().rename(columns={"index": "eid"})
    el["eid"] = el["eid"].astype("int32")
    el["etp"] = cupy.int32(0)

    G = hipgraph.Graph(directed=True)
    G.from_dask_cudf_edgelist(
        dask_cudf.from_cudf(el, npartitions=2),
        source="src",
        destination="dst",
        edge_attr=["wgt", "eid", "etp"],
    )

    samples_path = os.path.join(scratch_dir, "mg_test_bulk_sampler_mg_graph_sg_input")
    create_directory_with_overwrite(samples_path)

    bs = BulkSampler(
        batch_size=2,
        output_path=samples_path,
        graph=G,
        fanout_vals=[2, 2],
        with_replacement=False,
    )

    batches = cudf.DataFrame(
        {
            "start": cudf.Series([0, 5, 10, 15], dtype="int32"),
            "batch": cudf.Series([0, 0, 1, 1], dtype="int32"),
        }
    )

    bs.add_batches(batches, start_col_name="start", batch_col_name="batch")
    bs.flush()

    recovered_samples = cudf.read_parquet(samples_path)
    assert "map" not in recovered_samples.columns

    for b in batches["batch"].unique().values_host.tolist():
        assert b in recovered_samples["batch_id"].values_host.tolist()

    shutil.rmtree(samples_path)


@pytest.mark.mg
@pytest.mark.parametrize("mg_input", [True, False])
def test_bulk_sampler_partitions(dask_client, scratch_dir, mg_input):
    el = karate.get_edgelist().reset_index().rename(columns={"index": "eid"})
    el["eid"] = el["eid"].astype("int32")
    el["etp"] = cupy.int32(0)

    G = hipgraph.Graph(directed=True)
    G.from_dask_cudf_edgelist(
        dask_cudf.from_cudf(el, npartitions=2),
        source="src",
        destination="dst",
        edge_attr=["wgt", "eid", "etp"],
    )

    samples_path = os.path.join(scratch_dir, "test_bulk_sampler_partitions_mg")
    if os.path.exists(samples_path):
        shutil.rmtree(samples_path)
    os.makedirs(samples_path)

    bs = BulkSampler(
        batch_size=3,
        output_path=samples_path,
        graph=G,
        fanout_vals=[2, 2],
        with_replacement=False,
        batches_per_partition=2,
        renumber=True,
    )

    batches = cudf.DataFrame(
        {
            "start": cudf.Series([0, 5, 6, 10, 15, 17, 18, 9, 23], dtype="int32"),
            "batch": cudf.Series([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype="int32"),
        }
    )

    if mg_input:
        batches = dask_cudf.from_cudf(batches, npartitions=4)

    bs.add_batches(batches, start_col_name="start", batch_col_name="batch")
    bs.flush()

    for file in os.listdir(samples_path):
        start_batch_id, end_batch_id = [
            int(x) for x in re.match(r"batch=([0-9]+)-([0-9]+).parquet", file).groups()
        ]

        recovered_samples = cudf.read_parquet(os.path.join(samples_path, file))
        recovered_map = recovered_samples["map"]
        recovered_samples = recovered_samples.drop("map", axis=1).dropna()

        for current_batch_id in range(start_batch_id, end_batch_id + 1):
            map_start_ix = recovered_map.iloc[current_batch_id - start_batch_id]
            map_end_ix = recovered_map.iloc[current_batch_id - start_batch_id + 1]
            map_current_batch = recovered_map.iloc[map_start_ix:map_end_ix]
            n_unique = cudf.concat(
                [
                    recovered_samples[
                        recovered_samples.batch_id == current_batch_id
                    ].sources,
                    recovered_samples[
                        recovered_samples.batch_id == current_batch_id
                    ].destinations,
                ]
            ).nunique()
            assert len(map_current_batch) == n_unique


@pytest.mark.mg
def test_bulk_sampler_empty_batches(dask_client, scratch_dir):
    edgelist = dask_cudf.from_cudf(
        cudf.DataFrame(
            {
                "src": [0, 0, 1, 2, 3, 4, 5, 6, 4, 4],
                "dst": [3, 2, 0, 7, 8, 9, 1, 2, 8, 1],
            }
        ),
        npartitions=2,
    )

    batches = dask_cudf.from_cudf(
        cudf.DataFrame(
            {
                "start": [0, 1, 2, 7, 8, 9, 3, 2, 7],
                "batch": cudf.Series([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype="int32"),
            }
        ),
        npartitions=2,
    )

    G = hipgraph.Graph(directed=True)
    G.from_dask_cudf_edgelist(edgelist, source="src", destination="dst")

    samples_path = os.path.join(scratch_dir, "mg_test_bulk_sampler_empty_batches")
    create_directory_with_overwrite(samples_path)

    bs = BulkSampler(
        batch_size=3,
        output_path=samples_path,
        graph=G,
        fanout_vals=[-1, -1],
        with_replacement=False,
        batches_per_partition=6,
        renumber=False,
    )
    bs.add_batches(batches, start_col_name="start", batch_col_name="batch")
    bs.flush()

    assert len(os.listdir(samples_path)) == 1

    df = cudf.read_parquet(os.path.join(samples_path, "batch=0-1.parquet"))

    assert df[
        (df.batch_id == 0) & (df.hop_id == 0)
    ].destinations.sort_values().values_host.tolist() == [0, 2, 3, 7]

    assert df[
        (df.batch_id == 0) & (df.hop_id == 1)
    ].destinations.sort_values().values_host.tolist() == [2, 3, 7, 8]

    assert df[
        (df.batch_id == 1) & (df.hop_id == 0)
    ].destinations.sort_values().values_host.tolist() == [7, 8]

    assert len(df[(df.batch_id == 1) & (df.hop_id == 1)]) == 0

    assert df.batch_id.max() == 1

    shutil.rmtree(samples_path)


@pytest.mark.mg
@pytest.mark.parametrize("mg_input", [True, False])
def test_bulk_sampler_csr(dask_client, scratch_dir, mg_input):
    nworkers = len(dask_client.scheduler_info()["workers"])
    el = dask_cudf.from_cudf(email_Eu_core.get_edgelist(), npartitions=nworkers * 2)

    G = hipgraph.Graph(directed=True)
    G.from_dask_cudf_edgelist(el, source="src", destination="dst")

    samples_path = os.path.join(scratch_dir, "mg_test_bulk_sampler_csr")
    create_directory_with_overwrite(samples_path)

    bs = BulkSampler(
        batch_size=7,
        output_path=samples_path,
        graph=G,
        fanout_vals=[5, 4, 3],
        with_replacement=False,
        batches_per_partition=7,
        renumber=True,
        use_legacy_names=False,
        compression="CSR",
        compress_per_hop=True,
        prior_sources_behavior="carryover",
        deduplicate_sources=True,
        include_hop_column=False,
    )

    seeds = G.select_random_vertices(62, 1000)
    batch_ids = cudf.Series(
        cupy.repeat(cupy.arange(int(1000 / 7) + 1, dtype="int32"), 7)[:1000]
    ).sort_values()

    batch_df = cudf.DataFrame(
        {
            "seed": seeds.compute().values,
            "batch": batch_ids,
        }
    )

    if mg_input:
        batch_df = dask_cudf.from_cudf(batch_df, npartitions=2)

    bs.add_batches(batch_df, start_col_name="seed", batch_col_name="batch")
    bs.flush()

    assert len(os.listdir(samples_path)) == 21

    for file in os.listdir(samples_path):
        df = cudf.read_parquet(os.path.join(samples_path, file))

        assert df.major_offsets.dropna().iloc[-1] - df.major_offsets.iloc[0] == len(df)

    shutil.rmtree(samples_path)
