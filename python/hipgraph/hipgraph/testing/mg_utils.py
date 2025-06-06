# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import os
import tempfile
import time
from pprint import pformat

import numpy as np
from dask import persist
from dask.base import is_dask_collection
from dask.distributed import Client, default_client, wait
from dask_cuda import LocalCUDACluster
from dask_cuda.initialize import initialize
from hipgraph.dask.common.mg_utils import get_visible_devices
from hipgraph.dask.comms import comms as Comms
from hipgraph.generators import rmat


def start_dask_client(
    protocol=None,
    rmm_async=False,
    rmm_pool_size=None,
    dask_worker_devices=None,
    jit_unspill=False,
    worker_class=None,
    device_memory_limit=0.8,
    p2p=True,
):
    """
    Creates a new dask client, and possibly also a cluster, and returns them as
    a tuple (client, cluster).

    If the env var SCHEDULER_FILE is set, it is assumed to contain the path to
    a JSON file generated by a running dask scheduler that can be used to
    configure the new dask client (the new client object returned will be a
    client to that scheduler), and the value of cluster will be None. If
    SCHEDULER_FILE is not set, a new LocalCUDACluster will be created and
    returned as the value of cluster.

    If the env var DASK_WORKER_DEVICES is set, it will be assumed to be a list
    of comma-separated GPU devices (ex. "0,1,2" for those 3 devices) for the
    LocalCUDACluster to use when setting up individual workers (1 worker per
    device). If not set, the parameter dask_worker_devices will be used the
    same way instead. If neither are set, the new LocalCUDACluster instance
    will default to one worker per device visible to this process.

    If the env var DASK_LOCAL_DIRECTORY is set, it will be used as the
    "local_directory" arg to LocalCUDACluster, for all temp files generated.

    Upon successful creation of a client (either to a LocalCUDACluster or
    otherwise), the hipgraph.dask.comms.comms singleton is initialized using
    "p2p=True".

    Parameters
    ----------
    protocol : str or None, default None
        The "protocol" arg to LocalCUDACluster (ex. "tcp"), see docs for
        dask_cuda.LocalCUDACluster for details. This parameter is ignored if
        the env var SCHEDULER_FILE is set which implies the dask cluster has
        already been created.

    rmm_pool_size : int, str or None, default None
        The "rmm_pool_size" arg to LocalCUDACluster (ex. "20GB"), see docs for
        dask_cuda.LocalCUDACluster for details. This parameter is ignored if
        the env var SCHEDULER_FILE is set which implies the dask cluster has
        already been created.

    dask_worker_devices : str, list of int, or None, default None
        GPUs to restrict activity to. Can be a string (like ``"0,1,2,3"``),
        list (like ``[0, 1, 2, 3]``), or ``None`` to use all available GPUs.
        This parameter is overridden by the value of env var
        DASK_WORKER_DEVICES. This parameter is ignored if the env var
        SCHEDULER_FILE is set which implies the dask cluster has already been
        created.

    jit_unspill : bool or None, default None
        The "jit_unspill" arg to LocalCUDACluster to enable just-in-time
        spilling, see docs for dask_cuda.LocalCUDACluster for details. This
        parameter is ignored if the env var SCHEDULER_FILE is set which implies
        the dask cluster has already been created.

    device_memory_limit : int, float, str, or None, default 0.8
        The "device_memory_limit" arg to LocalCUDACluster to determine when
        workers start spilling to host memory, see docs for
        dask_cuda.LocalCUDACluster for details. This parameter is ignored if
        the env var SCHEDULER_FILE is set which implies the dask cluster has
        already been created.

    p2p : bool, optional (default=True)
        Initialize UCX endpoints if True.
    """
    dask_scheduler_file = os.environ.get("SCHEDULER_FILE")
    dask_local_directory = os.getenv("DASK_LOCAL_DIRECTORY")
    # Allow the DASK_WORKER_DEVICES env var to override a value passed in. If
    # neither are set, this will be None.
    dask_worker_devices = os.getenv("DASK_WORKER_DEVICES", dask_worker_devices)

    cluster = None
    client = None
    tempdir_object = None

    if dask_scheduler_file:
        if protocol is not None:
            print(
                f"WARNING: {protocol=} is ignored in start_dask_client() when using "
                "dask SCHEDULER_FILE"
            )
        if rmm_pool_size is not None:
            print(
                f"WARNING: {rmm_pool_size=} is ignored in start_dask_client() when "
                "using dask SCHEDULER_FILE"
            )
        if dask_worker_devices is not None:
            print(
                f"WARNING: {dask_worker_devices=} is ignored in start_dask_client() "
                "when using dask SCHEDULER_FILE"
            )
        initialize()
        client = Client(scheduler_file=dask_scheduler_file)
        # FIXME: use proper logging, INFO or DEBUG level
        print("\nDask client created using " f"{dask_scheduler_file}")
    else:
        if dask_local_directory is None:
            # The tempdir created by tempdir_object should be cleaned up once
            # tempdir_object is deleted.
            tempdir_object = tempfile.TemporaryDirectory()
            local_directory = tempdir_object.name
        else:
            local_directory = dask_local_directory

        cluster = LocalCUDACluster(
            local_directory=local_directory,
            protocol=protocol,
            rmm_pool_size=rmm_pool_size,
            rmm_async=rmm_async,
            CUDA_VISIBLE_DEVICES=dask_worker_devices,
            jit_unspill=jit_unspill,
            worker_class=worker_class,
            device_memory_limit=device_memory_limit,
        )
        client = Client(cluster)

        if dask_worker_devices is None:
            num_workers = len(get_visible_devices())
        else:
            if isinstance(dask_worker_devices, list):
                num_workers = len(dask_worker_devices)
            else:
                # FIXME: this assumes a properly formatted string with commas
                num_workers = len(dask_worker_devices.split(","))

        client.wait_for_workers(num_workers)
        # Add a reference to tempdir_object to the client to prevent it from
        # being deleted when this function returns. This will be deleted in
        # stop_dask_client()
        client.tempdir_object = tempdir_object
        # FIXME: use proper logging, INFO or DEBUG level
        print("\nDask client/cluster created using LocalCUDACluster")

    Comms.initialize(p2p=p2p)

    return (client, cluster)


def stop_dask_client(client, cluster=None):
    """
    Shutdown/cleanup a client and possibly cluster object returned from
    start_dask_client(). This also stops the hipgraph.dask.comms.comms
    singleton.
    """
    Comms.destroy()
    client.close()
    if cluster:
        cluster.close()
    # Remove a TemporaryDirectory object that may have been assigned to the
    # client, which should remove it and all the contents from disk.
    if hasattr(client, "tempdir_object"):
        del client.tempdir_object
    # FIXME: use proper logging, INFO or DEBUG level
    print("\nDask client closed.")


def restart_client(client):
    """
    Restart the Dask client
    """
    Comms.destroy()
    client.restart()
    client = client.run(enable_spilling)
    Comms.initialize(p2p=True)


def enable_spilling():
    import cudf

    cudf.set_option("spill", True)


def generate_edgelist_rmat(
    scale,
    edgefactor,
    seed=None,
    unweighted=False,
    mg=True,
):
    """
    Returns a cudf/dask_cudf DataFrame created using the R-MAT graph generator.

    The resulting graph is weighted with random values of a uniform distribution
    from the interval [0, 1)
    Args:
        scale:
            scale is used to determine the number of vertices to be generated (num_verts
            = 2^scale), which is also used to determine the data type for the vertex ID
            values in the DataFrame.
        edgefactor:
            edgefactor determies the number of edges (num_edges = num_edges*edgefactor)
        seed:
            seed, if specified, will be used as the seed to the RNG.
        unweighted:
            unweighted determines if the resulting edgelist will have randomly-generated
            weights ranging in value between [0, 1). If True, an edgelist with only 2
            columns is returned.
        mg:
            mg determines if the resulting edgelist will be a multi-GPU edgelist.
            If True, returns a dask_cudf.DataFrame and
            if False, returns a cudf.DataFrame.
    """
    ddf = rmat(
        scale,
        (2**scale) * edgefactor,
        0.57,  # from Graph500
        0.19,  # from Graph500
        0.19,  # from Graph500
        seed or 42,
        clip_and_flip=False,
        scramble_vertex_ids=True,
        create_using=None,  # return edgelist instead of Graph instance
        mg=mg,
    )
    if not unweighted:
        rng = np.random.default_rng(seed)
        ddf["weight"] = ddf.map_partitions(lambda df: rng.random(size=len(df)))
    return ddf


def set_statistics_adaptor():
    """
    Sets the current device resource to a StatisticsResourceAdaptor
    """
    import rmm

    rmm.mr.set_current_device_resource(
        rmm.mr.StatisticsResourceAdaptor(rmm.mr.get_current_device_resource())
    )


def _get_allocation_counts():
    """
    Returns the allocation counts from the current device resource
    """
    import rmm

    mr = rmm.mr.get_current_device_resource()
    if not hasattr(mr, "allocation_counts"):
        if hasattr(mr, "upstream_mr"):
            return _get_allocation_counts(mr.upstream_mr)
        else:
            return -1
    else:
        return mr.allocation_counts


def persist_dask_object(arg):
    """
    Persist if it is a dask object
    """
    if is_dask_collection(arg) or hasattr(arg, "persist"):
        arg = persist(arg)
        wait(arg)
        arg = arg[0]
    return arg


# Function to convert bytes into human readable format
def sizeof_fmt(num, suffix="B"):
    if isinstance(num, str):
        if num[-2:] == "GB":
            return num[:-2] + "G"
        elif num[-2:] == "MB":
            return num[:-2] + "M"
        elif num[-2:] == "KB":
            return num[:-2] + "K"
        else:
            raise ValueError("unknown unit")
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


def _parse_allocation_counts(allocation_counts):
    """
    Parses the allocation counts from the current device resource
    into human readable format
    """
    return {k: sizeof_fmt(v) for k, v in allocation_counts.items() if "bytes" in k}


# Decorator to set the statistics adaptor
# and calls the allocation_counts function
def get_allocation_counts_dask_lazy(return_allocations=False, logging=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            client = default_client()
            client.run(set_statistics_adaptor)
            st = time.time()
            return_val = func(*args, **kwargs)
            et = time.time()
            allocation_counts = client.run(_get_allocation_counts)
            if logging:
                _print_allocation_statistics(
                    func, args, kwargs, et - st, allocation_counts
                )
            client.run(set_statistics_adaptor)
            if return_allocations:
                return return_val, allocation_counts
            else:
                return return_val

        return wrapper

    return decorator


def get_allocation_counts_dask_persist(return_allocations=False, logging=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            args = [persist_dask_object(a) for a in args]
            kwargs = {k: persist_dask_object(v) for k, v in kwargs.items()}
            client = default_client()
            client.run(set_statistics_adaptor)
            st = time.time()
            return_val = func(*args, **kwargs)
            return_val = persist_dask_object(return_val)
            if isinstance(return_val, (list, tuple)):
                return_val = [persist_dask_object(d) for d in return_val]
            et = time.time()
            allocation_counts = client.run(_get_allocation_counts)
            if logging:
                _print_allocation_statistics(
                    func, args, kwargs, et - st, allocation_counts
                )
            client.run(set_statistics_adaptor)
            if return_allocations:
                return return_val, allocation_counts
            else:
                return return_val

        return wrapper

    return decorator


def _get_allocation_stats_string(func, args, kwargs, execution_time, allocation_counts):
    allocation_counts_parsed = {
        worker_id: _parse_allocation_counts(worker_allocations)
        for worker_id, worker_allocations in allocation_counts.items()
    }
    return (
        f"function:  {func.__name__}\n"
        + f"function args: {args} kwargs: {kwargs}\n"
        + f"execution_time: {execution_time}\n"
        + "allocation_counts:\n"
        + f"{pformat(allocation_counts_parsed, indent=4, width=1, compact=True)}"
    )


def _print_allocation_statistics(func, args, kwargs, execution_time, allocation_counts):
    print(
        _get_allocation_stats_string(
            func, args, kwargs, execution_time, allocation_counts
        )
    )


def get_peak_output_ratio_across_workers(allocation_counts):
    peak_ratio = -1
    for w_allocations in allocation_counts.values():
        w_peak_ratio = w_allocations["peak_bytes"] / w_allocations["current_bytes"]
        peak_ratio = max(w_peak_ratio, peak_ratio)
    return peak_ratio
