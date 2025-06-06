// Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/detail/utils.h>
#include <raft/sparse/op/sort.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/device_atomics.cuh>

#include <rmm/device_uvector.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cuda_runtime.h>
#include <raft/cusparse.h>
#else
#include <cuda_runtime.h>

#include <cusparse_v2.h>
#endif
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <stdio.h>

#include <algorithm>
#include <iostream>

namespace raft
{
    namespace sparse
    {
        namespace op
        {
            namespace detail
            {

                template <typename value_idx>
                RAFT_KERNEL compute_duplicates_diffs_kernel(const value_idx* rows,
                                                            const value_idx* cols,
                                                            value_idx*       diff,
                                                            size_t           nnz)
                {
                    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                    if(tid >= nnz)
                        return;

                    value_idx d = 1;
                    if(tid == 0 || (rows[tid - 1] == rows[tid] && cols[tid - 1] == cols[tid]))
                        d = 0;
                    diff[tid] = d;
                }

                template <typename value_idx, typename value_t>
                RAFT_KERNEL max_duplicates_kernel(const value_idx* src_rows,
                                                  const value_idx* src_cols,
                                                  const value_t*   src_vals,
                                                  const value_idx* index,
                                                  value_idx*       out_rows,
                                                  value_idx*       out_cols,
                                                  value_t*         out_vals,
                                                  size_t           nnz)
                {
                    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

                    if(tid < nnz)
                    {
                        value_idx idx = index[tid];
                        atomicMax(&out_vals[idx], src_vals[tid]);
                        out_rows[idx] = src_rows[tid];
                        out_cols[idx] = src_cols[tid];
                    }
                }

                /**
 * Computes a mask from a sorted COO matrix where 0's denote
 * duplicate values and 1's denote new values. This mask can
 * be useful for computing an exclusive scan to pre-build offsets
 * for reducing duplicates, such as when symmetrizing
 * or taking the min of each duplicated value.
 *
 * Note that this function always marks the first value as 0 so that
 * a cumulative sum can be performed as a follow-on. However, even
 * if the mask is used directly, any duplicates should always have a
 * 1 when first encountered so it can be assumed that the first element
 * is always a 1 otherwise.
 *
 * @tparam value_idx
 * @param[out] mask output mask, size nnz
 * @param[in] rows COO rows array, size nnz
 * @param[in] cols COO cols array, size nnz
 * @param[in] nnz number of nonzeros in input arrays
 * @param[in] stream cuda ops will be ordered wrt this stream
 */
                template <typename value_idx>
                void compute_duplicates_mask(value_idx*       mask,
                                             const value_idx* rows,
                                             const value_idx* cols,
                                             size_t           nnz,
                                             cudaStream_t     stream)
                {
                    RAFT_CUDA_TRY(cudaMemsetAsync(mask, 0, nnz * sizeof(value_idx), stream));

                    compute_duplicates_diffs_kernel<<<raft::ceildiv(nnz, (size_t)256),
                                                      256,
                                                      0,
                                                      stream>>>(rows, cols, mask, nnz);
                }

                /**
 * Performs a COO reduce of duplicate columns per row, taking the max weight
 * for duplicate columns in each row. This function assumes the input COO
 * has been sorted by both row and column but makes no assumption on
 * the sorting of values.
 * @tparam value_idx
 * @tparam value_t
 * @param[out] out output COO, the nnz will be computed allocate() will be called in this function.
 * @param[in] rows COO rows array, size nnz
 * @param[in] cols COO cols array, size nnz
 * @param[in] vals COO vals array, size nnz
 * @param[in] nnz number of nonzeros in COO input arrays
 * @param[in] m number of rows in COO input matrix
 * @param[in] n number of columns in COO input matrix
 * @param[in] stream cuda ops will be ordered wrt this stream
 */
                template <typename value_idx, typename value_t>
                void max_duplicates(raft::resources const&                 handle,
                                    raft::sparse::COO<value_t, value_idx>& out,
                                    const value_idx*                       rows,
                                    const value_idx*                       cols,
                                    const value_t*                         vals,
                                    size_t                                 nnz,
                                    size_t                                 m,
                                    size_t                                 n)
                {
                    auto stream        = resource::get_cuda_stream(handle);
                    auto thrust_policy = resource::get_thrust_policy(handle);

                    // compute diffs & take exclusive scan
                    rmm::device_uvector<value_idx> diff(nnz + 1, stream);

                    compute_duplicates_mask(diff.data(), rows, cols, nnz, stream);

                    thrust::exclusive_scan(
                        thrust_policy, diff.data(), diff.data() + diff.size(), diff.data());

                    // compute final size
                    value_idx size = 0;
                    raft::update_host(&size, diff.data() + (diff.size() - 1), 1, stream);
                    resource::sync_stream(handle, stream);
                    size++;

                    out.allocate(size, m, n, true, stream);

                    // perform reduce
                    max_duplicates_kernel<<<raft::ceildiv(nnz, (size_t)256), 256, 0, stream>>>(
                        rows, cols, vals, diff.data() + 1, out.rows(), out.cols(), out.vals(), nnz);
                }

            }; // END namespace detail
        }; // END namespace op
    }; // END namespace sparse
}; // END namespace raft
