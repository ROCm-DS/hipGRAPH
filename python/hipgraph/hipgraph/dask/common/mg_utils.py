# Copyright (c) 2020-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import gc
import os

import numba.cuda

# FIXME: this raft import breaks the library if ucx-py is
# not available. They are necessary only when doing MG work.
from hipgraph.dask.common.read_utils import MissingUCXPy

try:
    from raft_dask.common.utils import default_client
except ImportError as err:
    # FIXME: Generalize since err.name is arr when
    # libnuma.so.1 is not available
    if err.name == "ucp" or err.name == "arr":
        default_client = MissingUCXPy()
    else:
        raise


# FIXME: We currently look for the default client from dask, as such is the
# if there is a dask client running without any GPU we will still try
# to run MG using this client, it also implies that more  work will be
# required  in order to run an MG Batch in Combination with mutli-GPU Graph
def get_client():
    try:
        client = default_client()
    except ValueError:
        client = None
    return client


def prepare_worker_to_parts(data, client=None):
    if client is None:
        client = get_client()
    for placeholder, worker in enumerate(client.has_what().keys()):
        if worker not in data.worker_to_parts:
            data.worker_to_parts[worker] = [placeholder]
    return data


def is_single_gpu():
    ngpus = len(numba.cuda.gpus)
    if ngpus > 1:
        return False
    else:
        return True


def get_visible_devices():
    _visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if _visible_devices is None:
        # FIXME: We assume that if the variable is unset there is only one GPU
        visible_devices = ["0"]
    else:
        visible_devices = _visible_devices.strip().split(",")
    return visible_devices


def run_gc_on_dask_cluster(client):
    gc.collect()
    client.run(gc.collect)
