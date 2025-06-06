# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import os
import shutil
from typing import Any

import cudf
import cupy
import pytest
from hipgraph.datasets import karate
from hipgraph.gnn import (
    DistSampleWriter,
    UniformNeighborSampler,
    hipgraph_comms_create_unique_id,
    hipgraph_comms_get_raft_handle,
    hipgraph_comms_init,
    hipgraph_comms_shutdown,
)
from hipgraph.gnn.data_loading.bulk_sampler_io import create_df_from_disjoint_arrays
from hipgraph.utilities.utils import (
    MissingModule,
    create_directory_with_overwrite,
    import_optional,
)
from pylibhipgraph import GraphProperties, MGGraph, ResourceHandle

torch = import_optional("torch")
if __name__ == "__main__" and not isinstance(torch, MissingModule):
    if torch.cuda.is_available():
        from rmm.allocators.torch import rmm_torch_allocator

        torch.cuda.change_current_allocator(rmm_torch_allocator)
    else:
        pytest.skip("CUDA-enabled PyTorch is unavailable", allow_module_level=True)


def karate_mg_graph(rank, world_size):
    el = karate.get_edgelist().reset_index().rename(columns={"index": "eid"})
    split = cupy.array_split(cupy.arange(len(el)), world_size)[rank]
    el = el.iloc[split]

    G = MGGraph(
        ResourceHandle(hipgraph_comms_get_raft_handle().getHandle()),
        GraphProperties(is_multigraph=True, is_symmetric=False),
        [el.src.astype("int64")],
        [el.dst.astype("int64")],
        edge_id_array=[el.eid],
    )

    return G


def init_pytorch(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def run_test_dist_sampler_simple(
    rank,
    world_size,
    uid,
    samples_path,
    batch_size,
    seeds_per_rank,
    fanout,
    equal_input_size,
    seeds_per_call,
):
    init_pytorch(rank, world_size)
    hipgraph_comms_init(rank, world_size, uid, device=rank)

    G = karate_mg_graph(rank, world_size)

    writer = DistSampleWriter(samples_path)

    sampler = UniformNeighborSampler(
        G, writer, fanout=fanout, local_seeds_per_call=seeds_per_call
    )

    seeds = cupy.random.randint(0, 34, seeds_per_rank, dtype="int64")

    from time import perf_counter

    start_time = perf_counter()
    sampler.sample_from_nodes(
        seeds, batch_size=batch_size, assume_equal_input_size=equal_input_size
    )
    end_time = perf_counter()

    print("time:", end_time - start_time)

    hipgraph_comms_shutdown()


@pytest.mark.mg
@pytest.mark.parametrize("equal_input_size", [True, False])
@pytest.mark.parametrize("fanout", [[4, 4], [4, 2, 1]])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seeds_per_rank", [8, 1])
@pytest.mark.parametrize("seeds_per_call", [4, 8])
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not installed")
def test_dist_sampler_simple(
    scratch_dir, batch_size, seeds_per_rank, fanout, equal_input_size, seeds_per_call
):
    uid = hipgraph_comms_create_unique_id()

    samples_path = os.path.join(scratch_dir, "test_bulk_sampler_mg_simple")
    create_directory_with_overwrite(samples_path)

    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        run_test_dist_sampler_simple,
        args=(
            world_size,
            uid,
            samples_path,
            batch_size,
            seeds_per_rank,
            fanout,
            equal_input_size,
            seeds_per_call,
        ),
        nprocs=world_size,
    )

    for file in os.listdir(samples_path):
        recovered_samples = cudf.read_parquet(os.path.join(samples_path, file))
        original_el = karate.get_edgelist()

        for b in range(len(recovered_samples.renumber_map_offsets.dropna()) - 1):
            el_start = int(recovered_samples.label_hop_offsets.iloc[b * len(fanout)])
            el_end = int(
                recovered_samples.label_hop_offsets.iloc[(b + 1) * len(fanout)]
            )
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


def run_test_dist_sampler_uneven(
    rank, world_size, uid, samples_path, batch_size, fanout, seeds_per_call
):
    init_pytorch(rank, world_size)
    hipgraph_comms_init(rank, world_size, uid, device=rank)

    G = karate_mg_graph(rank, world_size)

    writer = DistSampleWriter(samples_path)

    sampler = UniformNeighborSampler(
        G, writer, fanout=fanout, local_seeds_per_call=seeds_per_call
    )

    num_seeds = 8 + rank
    seeds = cupy.random.randint(0, 34, num_seeds, dtype="int64")

    from time import perf_counter

    start_time = perf_counter()
    sampler.sample_from_nodes(
        seeds, batch_size=batch_size, assume_equal_input_size=False
    )
    end_time = perf_counter()

    print("time:", end_time - start_time)

    hipgraph_comms_shutdown()


@pytest.mark.mg
@pytest.mark.parametrize("fanout", [[4, 4], [4, 2, 1]])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seeds_per_call", [4, 8, 16])
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not installed")
def test_dist_sampler_uneven(scratch_dir, batch_size, fanout, seeds_per_call):
    uid = hipgraph_comms_create_unique_id()

    samples_path = os.path.join(scratch_dir, "test_bulk_sampler_mg_uneven")
    create_directory_with_overwrite(samples_path)

    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        run_test_dist_sampler_uneven,
        args=(world_size, uid, samples_path, batch_size, fanout, seeds_per_call),
        nprocs=world_size,
    )

    for file in os.listdir(samples_path):
        recovered_samples = cudf.read_parquet(os.path.join(samples_path, file))
        original_el = karate.get_edgelist()

        for b in range(len(recovered_samples.renumber_map_offsets.dropna()) - 1):
            el_start = int(recovered_samples.label_hop_offsets.iloc[b * len(fanout)])
            el_end = int(
                recovered_samples.label_hop_offsets.iloc[(b + 1) * len(fanout)]
            )
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


def run_test_dist_sampler_buffered_in_memory(
    rank: int,
    world_size: int,
    uid: Any,
    samples_path: str,
    seeds_per_call: int,
    compression: str,
):
    init_pytorch(rank, world_size)
    hipgraph_comms_init(rank, world_size, uid, device=rank)

    G = karate_mg_graph(rank, world_size)

    num_seeds = 8
    seeds = cupy.random.randint(0, 34, num_seeds, dtype="int64")

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


@pytest.mark.mg
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.parametrize("seeds_per_call", [4, 5, 10])
@pytest.mark.parametrize("compression", ["COO", "CSR"])
def test_dist_sampler_buffered_in_memory(scratch_dir, seeds_per_call, compression):
    uid = hipgraph_comms_create_unique_id()

    samples_path = os.path.join(scratch_dir, "test_bulk_sampler_buffered_in_memory_mg")
    create_directory_with_overwrite(samples_path)

    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        run_test_dist_sampler_buffered_in_memory,
        args=(world_size, uid, samples_path, seeds_per_call, compression),
        nprocs=world_size,
    )

    shutil.rmtree(samples_path)
