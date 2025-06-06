// Copyright (c) 2023, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

// Defines a named requirement "has_cutlass_op"
#include <raft/distance/detail/distance_ops/cutlass.cuh>

// The distance operations:
#include <raft/distance/detail/distance_ops/canberra.cuh>
#include <raft/distance/detail/distance_ops/correlation.cuh>
#include <raft/distance/detail/distance_ops/cosine.cuh>
#include <raft/distance/detail/distance_ops/hamming.cuh>
#include <raft/distance/detail/distance_ops/hellinger.cuh>
#include <raft/distance/detail/distance_ops/jensen_shannon.cuh>
#include <raft/distance/detail/distance_ops/kl_divergence.cuh>
#include <raft/distance/detail/distance_ops/l1.cuh>
#include <raft/distance/detail/distance_ops/l2_exp.cuh>
#include <raft/distance/detail/distance_ops/l2_unexp.cuh>
#include <raft/distance/detail/distance_ops/l_inf.cuh>
#include <raft/distance/detail/distance_ops/lp_unexp.cuh>
#include <raft/distance/detail/distance_ops/russel_rao.cuh>
