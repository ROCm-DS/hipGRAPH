# Copyright (c) 2020-2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from hipgraph.dask.comms.comms cimport init_subcomm as c_init_subcomm
from pylibraft.common.handle cimport *


def init_subcomms(handle, row_comm_size):
    cdef size_t handle_size_t = <size_t>handle.getHandle()
    handle_ = <handle_t*>handle_size_t
    c_init_subcomm(handle_[0], row_comm_size)
