# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import os
import shutil

import cudf
import cupy
import pytest
from hipgraph.datasets import karate
from hipgraph.gnn import DistSampleWriter, UniformNeighborSampler
from hipgraph.gnn.data_loading.bulk_sampler_io import create_df_from_disjoint_arrays
from hipgraph.utilities.utils import (
    MissingModule,
    create_directory_with_overwrite,
    import_optional,
)
from pylibhipgraph import GraphProperties, ResourceHandle, SGGraph

torch = import_optional("torch")
if not isinstance(torch, MissingModule):
    if torch.cuda.is_available():
        from rmm.allocators.torch import rmm_torch_allocator

        torch.cuda.change_current_allocator(rmm_torch_allocator)
    else:
        pytest.skip("CUDA-enabled PyTorch is unavailable", allow_module_level=True)


@pytest.fixture
def karate_graph() -> SGGraph:
    el = karate.get_edgelist().reset_index().rename(columns={"index": "eid"})
    G = SGGraph(
        ResourceHandle(),
        GraphProperties(is_multigraph=True, is_symmetric=False),
        el.src.astype("int64"),
        el.dst.astype("int64"),
        edge_id_array=el.eid,
    )

    return G


@pytest.mark.sg
@pytest.mark.parametrize("equal_input_size", [True, False])
@pytest.mark.parametrize("fanout", [[2, 2], [4, 4], [4, 2, 1]])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_dist_sampler_simple(
    scratch_dir, karate_graph, batch_size, fanout, equal_input_size
):
    G = karate_graph

    samples_path = os.path.join(scratch_dir, "test_bulk_sampler_simple")
    create_directory_with_overwrite(samples_path)

    writer = DistSampleWriter(samples_path)

    sampler = UniformNeighborSampler(G, writer, fanout=fanout)

    seeds = cupy.array([0, 5, 10, 15], dtype="int64")

    sampler.sample_from_nodes(
        seeds, batch_size=batch_size, assume_equal_input_size=equal_input_size
    )

    recovered_samples = cudf.read_parquet(samples_path)
    print(recovered_samples)
    original_el = karate.get_edgelist()

    for b in range(len(seeds) // batch_size):
        el_start = recovered_samples.label_hop_offsets.iloc[b * len(fanout)]
        el_end = recovered_samples.label_hop_offsets.iloc[(b + 1) * len(fanout)]
        print(el_start, el_end)
        src = recovered_samples.majors.iloc[el_start:el_end]
        dst = recovered_samples.minors.iloc[el_start:el_end]
        edge_id = recovered_samples.edge_id.iloc[el_start:el_end]

        map_start = recovered_samples.renumber_map_offsets[b]
        map_end = recovered_samples.renumber_map_offsets[b + 1]
        renumber_map = recovered_samples["map"].iloc[map_start:map_end]

        src = renumber_map.iloc[src.values]
        dst = renumber_map.iloc[dst.values]

        for i in range(len(edge_id)):
            assert original_el.src.iloc[edge_id.iloc[i]] == src.iloc[i]
            assert original_el.dst.iloc[edge_id.iloc[i]] == dst.iloc[i]

    shutil.rmtree(samples_path)


@pytest.mark.sg
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.parametrize("seeds_per_call", [4, 5, 10])
@pytest.mark.parametrize("compression", ["COO", "CSR"])
def test_dist_sampler_buffered_in_memory(
    scratch_dir: str, karate_graph: SGGraph, seeds_per_call: int, compression: str
):
    G = karate_graph

    samples_path = os.path.join(scratch_dir, "test_bulk_sampler_buffered_in_memory")
    create_directory_with_overwrite(samples_path)

    seeds = cupy.arange(10, dtype="int64")

    unbuffered_sampler = UniformNeighborSampler(
        G,
        writer=DistSampleWriter(samples_path),
        local_seeds_per_call=seeds_per_call,
        compression=compression,
    )

    buffered_sampler = UniformNeighborSampler(
        G,
        writer=None,
        local_seeds_per_call=seeds_per_call,
        compression=compression,
    )

    unbuffered_results = unbuffered_sampler.sample_from_nodes(
        seeds,
        batch_size=4,
    )

    unbuffered_results = [
        (create_df_from_disjoint_arrays(r[0]), r[1], r[2]) for r in unbuffered_results
    ]

    buffered_results = buffered_sampler.sample_from_nodes(seeds, batch_size=4)
    buffered_results = [
        (create_df_from_disjoint_arrays(r[0]), r[1], r[2]) for r in buffered_results
    ]

    assert len(buffered_results) == len(unbuffered_results)

    for k in range(len(buffered_results)):
        br, bs, be = buffered_results[k]
        ur, us, ue = unbuffered_results[k]

        assert bs == us
        assert be == ue

        for col in ur.columns:
            assert (br[col].dropna() == ur[col].dropna()).all()

    shutil.rmtree(samples_path)
