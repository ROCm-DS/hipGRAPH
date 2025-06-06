// Copyright (c) 2020-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <rmm/device_uvector.hpp>

#include <iostream>

namespace raft::sparse::solver::detail
{

    template <typename idx_t>
    __device__ idx_t get_1D_idx()
    {
        return blockIdx.x * blockDim.x + threadIdx.x;
    }

} // namespace raft::sparse::solver::detail
