# Copyright (c) 2018-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# FIXME: these raft imports break the library if ucx-py is
# not available. They are necessary only when doing MG work.
from hipgraph.dask.common.read_utils import MissingUCXPy

try:
    from raft_dask.common.comms import Comms as raftComms
    from raft_dask.common.comms import get_raft_comm_state
except ImportError as err:
    # FIXME: Generalize since err.name is arr when
    # libnuma.so.1 is not available
    if err.name == "ucp" or err.name == "arr":
        raftComms = MissingUCXPy()
        get_raft_comm_state = MissingUCXPy()
    else:
        raise
import math

from dask.distributed import default_client, get_worker
from hipgraph.dask.common import read_utils
from hipgraph.dask.comms.comms_wrapper import init_subcomms as c_init_subcomms
from pylibraft.common.handle import Handle

__instance = None
__default_handle = None
__subcomm = None


def __get_2D_div(ngpus):
    prows = int(math.sqrt(ngpus))
    while ngpus % prows != 0:
        prows = prows - 1
    return prows, int(ngpus / prows)


def subcomm_init(prows, pcols, partition_type):
    sID = get_session_id()
    ngpus = get_n_workers()
    if prows is None and pcols is None:
        if partition_type == 1:
            pcols, prows = __get_2D_div(ngpus)
        else:
            prows, pcols = __get_2D_div(ngpus)
    else:
        if prows is not None and pcols is not None:
            if ngpus != prows * pcols:
                raise Exception(
                    "prows*pcols should be equal to the\
 number of processes"
                )
        elif prows is not None:
            if ngpus % prows != 0:
                raise Exception(
                    "prows must be a factor of the number\
 of processes"
                )
            pcols = int(ngpus / prows)
        elif pcols is not None:
            if ngpus % pcols != 0:
                raise Exception(
                    "pcols must be a factor of the number\
 of processes"
                )
            prows = int(ngpus / pcols)

    client = default_client()
    client.run(_subcomm_init, sID, pcols)
    global __subcomm
    __subcomm = (prows, pcols, partition_type)


def _subcomm_init(sID, partition_row_size, dask_worker=None):
    handle = get_handle(sID, dask_worker)
    c_init_subcomms(handle, partition_row_size)


def initialize(comms=None, p2p=False, prows=None, pcols=None, partition_type=1):
    """
    Initialize a communicator for multi-node/multi-gpu communications.  It is
    expected to be called right after client initialization for running
    multi-GPU algorithms (this wraps raft comms that manages underlying NCCL
    and UCX comms handles across the workers of a Dask cluster).

    It is recommended to also call `destroy()` when the comms are no longer
    needed so the underlying resources can be cleaned up.

    Parameters
    ----------
    comms : raft Comms, optional (default=None)
        A pre-initialized raft communicator. If provided, this is used for mnmg
        communications. If not provided, default comms are initialized as per
        client information.

    p2p : bool, optional (default=False)
        Initialize UCX endpoints if True.

    prows : int, optional (default=None)
        Specifies the number of rows when performing a 2D partitioning of the
        input graph. If specified, this must be a factor of the total number of
        parallel processes. When specified with pcols, prows*pcols should be
        equal to the total number of parallel processes.

    pcols : int, optional (default=None)
        Specifies the number of columns when performing a 2D partitioning of
        the input graph. If specified, this must be a factor of the total
        number of parallel processes. When specified with prows, prows*pcols
        should be equal to the total number of parallel processes.

    partition_type : int, optional (default=1)
        Valid values are currently 1 or any int other than 1. A value of 1 (the
        default) represents a partitioning resulting in prows*pcols
        partitions. A non-1 value currently results in a partitioning of
        p*pcols partitions, where p is the number of GPUs.

    Examples
    --------
    >>> from dask.distributed import Client
    >>> from dask_cuda import LocalCUDACluster
    >>> import hipgraph.dask.comms as Comms
    >>> cluster = LocalCUDACluster()
    >>> client = Client(cluster)
    >>> Comms.initialize(p2p=True)
    >>> # DO WORK HERE
    >>> # All done, clean up
    >>> Comms.destroy()
    >>> client.close()
    >>> cluster.close()

    """

    global __instance
    if __instance is None:
        global __default_handle
        __default_handle = None
        if comms is None:
            # Initialize communicator
            __instance = raftComms(comms_p2p=p2p)
            __instance.init()
            # Initialize subcommunicator
            subcomm_init(prows, pcols, partition_type)
        else:
            __instance = comms
    else:
        raise Exception("Communicator is already initialized")


def is_initialized():
    """
    Returns True if comms was initialized, False otherwise.
    """
    global __instance
    if __instance is not None:
        return True
    else:
        return False


def get_comms():
    """
    Returns raft Comms instance
    """
    global __instance
    return __instance


def get_workers():
    """
    Returns the workers in the Comms instance, or None if Comms is not
    initialized.
    """
    if is_initialized():
        global __instance
        return __instance.worker_addresses


def get_session_id():
    """
    Returns the sessionId for finding sessionstate of workers, or None if Comms
    is not initialized.
    """
    if is_initialized():
        global __instance
        return __instance.sessionId


def get_2D_partition():
    """
    Returns a tuple representing the 2D partition information: (prows, pcols,
    partition_type)
    """
    global __subcomm
    if __subcomm is not None:
        return __subcomm


def destroy():
    """
    Shuts down initialized comms and cleans up resources.
    """
    global __instance
    if is_initialized():
        __instance.destroy()
        __instance = None


def get_default_handle():
    """
    Returns the default handle. This does not perform nccl initialization.
    """
    global __default_handle
    if __default_handle is None:
        __default_handle = Handle()
    return __default_handle


# Functions to be called from within workers


def get_handle(sID, dask_worker=None):
    """
    Returns the handle from within the worker using the sessionstate.
    """
    if dask_worker is None:
        dask_worker = get_worker()
    sessionstate = get_raft_comm_state(sID, dask_worker)
    return sessionstate["handle"]


def get_worker_id(sID, dask_worker=None):
    """
    Returns the worker's sessionId from within the worker.
    """
    if dask_worker is None:
        dask_worker = get_worker()
    sessionstate = get_raft_comm_state(sID, dask_worker)
    return sessionstate["wid"]


# FIXME: There are several similar instances of utility functions for getting
# the number of workers, including:
#   * get_n_workers() (from hipgraph.dask.common.read_utils)
#   * len(get_visible_devices())
#   * len(numba.cuda.gpus)
# Consider consolidating these or emphasizing why different
# functions/techniques are needed.
def get_n_workers(sID=None, dask_worker=None):
    if sID is None:
        return read_utils.get_n_workers()
    else:
        if dask_worker is None:
            dask_worker = get_worker()
        sessionstate = get_raft_comm_state(sID, dask_worker)
        return sessionstate["nworkers"]


def rank_to_worker(client):
    """
    Return a mapping of ranks to dask workers.
    """
    workers = client.scheduler_info()["workers"].keys()
    worker_info = __instance.worker_info(workers)
    rank_to_worker = {}
    for w in worker_info:
        rank_to_worker[worker_info[w]["rank"]] = w

    return rank_to_worker
