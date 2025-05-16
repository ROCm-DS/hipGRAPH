# Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

from hipgraph.gnn.data_loading.bulk_sampler import BulkSampler
from hipgraph.gnn.data_loading.dist_io import (
    BufferedSampleReader,
    DistSampleReader,
    DistSampleWriter,
)
from hipgraph.gnn.data_loading.dist_sampler import DistSampler, NeighborSampler


def UniformNeighborSampler(*args, **kwargs):
    return NeighborSampler(
        *args,
        **kwargs,
        biased=False,
    )


def BiasedNeighborSampler(*args, **kwargs):
    return NeighborSampler(
        *args,
        **kwargs,
        biased=True,
    )
