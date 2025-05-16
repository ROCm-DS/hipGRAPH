# Copyright (c) 2019-2022, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT


def get_n_workers():
    from dask.distributed import default_client

    client = default_client()
    return len(client.scheduler_info()["workers"])


def get_chunksize(input_path):
    """
    Calculate the appropriate chunksize for dask_cudf.read_csv
    to get a number of partitions equal to the number of GPUs.

    Examples
    --------
    >>> import hipgraph.dask as dcg
    >>> chunksize = dcg.get_chunksize(datasets_path / 'netscience.csv')

    """

    import math
    import os
    from glob import glob

    input_files = sorted(glob(str(input_path)))
    if len(input_files) == 1:
        size = os.path.getsize(input_files[0])
        chunksize = math.ceil(size / get_n_workers())
    else:
        size = [os.path.getsize(_file) for _file in input_files]
        chunksize = max(size)
    return chunksize


class MissingUCXPy:
    def __call__(self, *args, **kwargs):
        raise ModuleNotFoundError(
            "ucx-py could not be imported but is required for MG operations"
        )
