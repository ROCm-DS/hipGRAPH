# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import math

from hipgraph.dask.comms.comms_wrapper import init_subcomms
from pylibraft.common.handle import Handle
from raft_dask.common.comms_utils import inject_comms_on_handle_coll_only
from raft_dask.common.nccl import nccl
from rmm._cuda.gpu import getDevice, setDevice

__nccl_comms = None
__raft_handle = None


def nccl_init(rank: int, world_size: int, uid: int):
    try:
        ni = nccl()
        ni.init(world_size, uid, rank)
        return ni
    except Exception as ex:
        raise RuntimeError(f"A nccl error occurred: {ex}")


def make_raft_handle(
    rank, world_size, nccl_comms, n_streams_per_handle=0, verbose=False
):
    handle = Handle(n_streams=n_streams_per_handle)
    inject_comms_on_handle_coll_only(handle, nccl_comms, world_size, rank, verbose)

    return handle


def __get_2D_div(ngpus):
    prows = int(math.sqrt(ngpus))
    while ngpus % prows != 0:
        prows = prows - 1
    return prows, int(ngpus / prows)


def hipgraph_comms_init(rank, world_size, uid, device=0):
    global __nccl_comms, __raft_handle
    if __nccl_comms is not None or __raft_handle is not None:
        raise RuntimeError("hipGRAPH has already been initialized!")

    # TODO add options for rmm initialization

    global __old_device
    __old_device = getDevice()
    setDevice(device)

    nccl_comms = nccl_init(rank, world_size, uid)
    # FIXME should we use n_streams_per_handle=1 here?
    raft_handle = make_raft_handle(rank, world_size, nccl_comms, verbose=True)

    pcols, _ = __get_2D_div(world_size)
    init_subcomms(raft_handle, pcols)

    __nccl_comms = nccl_comms
    __raft_handle = raft_handle


def hipgraph_comms_shutdown():
    global __raft_handle, __nccl_comms, __old_device

    __nccl_comms.destroy()
    setDevice(__old_device)

    del __raft_handle
    del __nccl_comms
    del __old_device


def hipgraph_comms_create_unique_id():
    return nccl.get_unique_id()


def hipgraph_comms_get_raft_handle():
    global __raft_handle
    return __raft_handle
