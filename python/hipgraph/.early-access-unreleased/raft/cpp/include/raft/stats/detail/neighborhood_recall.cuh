// Copyright (c) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/*
 * Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/math.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#include <hip/std/atomic>
#include <hip/std/optional>
#else
#include <cub/cub.cuh>
#include <cuda/atomic>

#include <optional>
#endif

#include <cstddef>

namespace raft::stats::detail
{

    template <typename IndicesValueType,
              typename DistanceValueType,
              typename IndexType,
              typename ScalarType>
    RAFT_KERNEL neighborhood_recall(
        raft::device_matrix_view<const IndicesValueType, IndexType, raft::row_major> indices,
        raft::device_matrix_view<const IndicesValueType, IndexType, raft::row_major> ref_indices,
        hip::std::optional<
            raft::device_matrix_view<const DistanceValueType, IndexType, raft::row_major>>
            distances,
        hip::std::optional<
            raft::device_matrix_view<const DistanceValueType, IndexType, raft::row_major>>
                                             ref_distances,
        raft::device_scalar_view<ScalarType> recall_score,
        DistanceValueType const              eps)
    {
        auto constexpr kThreadsPerBlock = 32;
        IndexType const row_idx         = blockIdx.x;
        auto const      lane_idx        = threadIdx.x % kThreadsPerBlock;

        // Each warp stores a recall score computed across the columns per row
        IndexType thread_recall_score = 0;
        for(IndexType col_idx = lane_idx; col_idx < indices.extent(1); col_idx += kThreadsPerBlock)
        {
            for(IndexType ref_col_idx = 0; ref_col_idx < ref_indices.extent(1); ref_col_idx++)
            {
                if(indices(row_idx, col_idx) == ref_indices(row_idx, ref_col_idx))
                {
                    thread_recall_score += 1;
                    break;
                }
                else if(distances.has_value())
                {
                    auto              dist     = distances.value()(row_idx, col_idx);
                    auto              ref_dist = ref_distances.value()(row_idx, ref_col_idx);
                    DistanceValueType diff     = raft::abs(dist - ref_dist);
                    DistanceValueType m        = std::max(raft::abs(dist), raft::abs(ref_dist));
                    DistanceValueType ratio    = diff > eps ? diff / m : diff;

                    if(ratio <= eps)
                    {
                        thread_recall_score += 1;
                        break;
                    }
                }
            }
        }

        // Reduce across a warp for row score
        typedef cub::BlockReduce<IndexType, kThreadsPerBlock> BlockReduce;

        __shared__ typename BlockReduce::TempStorage temp_storage;

        ScalarType row_recall_score = BlockReduce(temp_storage).Sum(thread_recall_score);

        // Reduce across all rows for global score
        if(lane_idx == 0)
        {
            cuda::atomic_ref<ScalarType, cuda::thread_scope_device> device_recall_score{
                *recall_score.data_handle()};
            std::size_t const total_count = indices.extent(0) * indices.extent(1);
            device_recall_score.fetch_add(row_recall_score / total_count);
        }
    }

    template <typename IndicesValueType,
              typename DistanceValueType,
              typename IndexType,
              typename ScalarType>
    void neighborhood_recall(
        raft::resources const&                                                       res,
        raft::device_matrix_view<const IndicesValueType, IndexType, raft::row_major> indices,
        raft::device_matrix_view<const IndicesValueType, IndexType, raft::row_major> ref_indices,
        hip::std::optional<
            raft::device_matrix_view<const DistanceValueType, IndexType, raft::row_major>>
            distances,
        hip::std::optional<
            raft::device_matrix_view<const DistanceValueType, IndexType, raft::row_major>>
                                             ref_distances,
        raft::device_scalar_view<ScalarType> recall_score,
        DistanceValueType const              eps)
    {
        // One warp per row, launch a warp-width block per-row kernel
        auto constexpr kThreadsPerBlock = 32;
        auto const num_blocks           = indices.extent(0);

        neighborhood_recall<<<num_blocks,
                              kThreadsPerBlock,
                              0,
                              raft::resource::get_cuda_stream(res)>>>(
            indices, ref_indices, distances, ref_distances, recall_score, eps);
    }

} // end namespace raft::stats::detail
