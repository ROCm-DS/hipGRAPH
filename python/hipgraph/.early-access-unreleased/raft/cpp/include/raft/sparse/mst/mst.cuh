// Copyright (c) 2020-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/**
 * This file is deprecated and will be removed in release 22.06.
 * Please use the cuh version instead.
 */
#pragma once

#ifndef RAFT_HIDE_DEPRECATION_WARNINGS
#pragma message(__FILE__ " is deprecated and will be removed in a future release." \
                         " Please use the raft/sparse/solver version instead.")
#endif

#include <raft/sparse/mst/mst_solver.cuh>
#include <raft/sparse/solver/mst.cuh>

namespace raft::mst
{
    using raft::sparse::solver::mst;
}
