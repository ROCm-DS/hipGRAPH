# Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import os

import numba.cuda
import numpy as np
import pytest
from hipgraph.gnn import FeatureStore
from hipgraph.utilities.utils import MissingModule, import_optional

pylibwholegraph = import_optional("pylibwholegraph")
wmb = import_optional("pylibwholegraph.binding.wholememory_binding")
torch = import_optional("torch")
wgth = import_optional("pylibwholegraph.torch")


def get_cudart_version():
    major, minor = numba.cuda.runtime.get_version()
    return major * 1000 + minor * 10


def runtest(rank: int, world_size: int):
    torch.cuda.set_device(rank)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    pylibwholegraph.torch.initialize.init(
        rank,
        world_size,
        rank,
        world_size,
    )
    wm_comm = wgth.get_global_communicator()

    generator = np.random.default_rng(62)
    arr = (
        generator.integers(low=0, high=100, size=100_000)
        .reshape(10_000, -1)
        .astype("float64")
    )

    fs = FeatureStore(backend="wholegraph")
    fs.add_data(arr, "type2", "feat1")
    wm_comm.barrier()

    indices_to_fetch = np.random.randint(low=0, high=len(arr), size=1024)
    output_fs = fs.get_data(indices_to_fetch, type_name="type2", feat_name="feat1")
    assert isinstance(output_fs, torch.Tensor)
    assert output_fs.is_cuda
    expected = arr[indices_to_fetch]
    np.testing.assert_array_equal(output_fs.cpu().numpy(), expected)

    pylibwholegraph.torch.initialize.finalize()


@pytest.mark.sg
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.skipif(
    isinstance(pylibwholegraph, MissingModule), reason="wholegraph not available"
)
@pytest.mark.skipif(
    get_cudart_version() < 11080, reason="not compatible with CUDA < 11.8"
)
def test_feature_storage_wholegraph_backend():
    world_size = torch.cuda.device_count()
    print("gpu count:", world_size)
    assert world_size > 0

    print("ignoring gpu count and running on 1 GPU only")

    torch.multiprocessing.spawn(runtest, args=(1,), nprocs=1)


@pytest.mark.mg
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.skipif(
    isinstance(pylibwholegraph, MissingModule), reason="wholegraph not available"
)
@pytest.mark.skipif(
    get_cudart_version() < 11080, reason="not compatible with CUDA < 11.8"
)
def test_feature_storage_wholegraph_backend_mg():
    world_size = torch.cuda.device_count()
    print("gpu count:", world_size)
    assert world_size > 0

    torch.multiprocessing.spawn(runtest, args=(world_size,), nprocs=world_size)
