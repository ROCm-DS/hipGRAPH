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

#include <raft/sparse/coo.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/detail/utils.h>
#include <raft/sparse/linalg/degree.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cuda_runtime.h>
#include <raft/cusparse.h>
#else
#include <cuda_runtime.h>

#include <cusparse_v2.h>
#endif
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <algorithm>
#include <cstdio>
#include <iostream>

namespace raft
{
    namespace sparse
    {
        namespace op
        {
            namespace detail
            {

                template <int TPB_X, typename T>
                RAFT_KERNEL coo_remove_scalar_kernel(const int* rows,
                                                     const int* cols,
                                                     const T*   vals,
                                                     int        nnz,
                                                     int*       crows,
                                                     int*       ccols,
                                                     T*         cvals,
                                                     int*       ex_scan,
                                                     int*       cur_ex_scan,
                                                     int        m,
                                                     T          scalar)
                {
                    int row = (blockIdx.x * TPB_X) + threadIdx.x;

                    if(row < m)
                    {
                        int start       = cur_ex_scan[row];
                        int stop        = get_stop_idx(row, m, nnz, cur_ex_scan);
                        int cur_out_idx = ex_scan[row];

                        for(int idx = start; idx < stop; idx++)
                        {
                            if(vals[idx] != scalar)
                            {
                                crows[cur_out_idx] = rows[idx];
                                ccols[cur_out_idx] = cols[idx];
                                cvals[cur_out_idx] = vals[idx];
                                ++cur_out_idx;
                            }
                        }
                    }
                }

                /**
 * @brief Removes the values matching a particular scalar from a COO formatted sparse matrix.
 *
 * @param rows: input array of rows (size n)
 * @param cols: input array of cols (size n)
 * @param vals: input array of vals (size n)
 * @param nnz: size of current rows/cols/vals arrays
 * @param crows: compressed array of rows
 * @param ccols: compressed array of cols
 * @param cvals: compressed array of vals
 * @param cnnz: array of non-zero counts per row
 * @param cur_cnnz array of counts per row
 * @param scalar: scalar to remove from arrays
 * @param n: number of rows in dense matrix
 * @param d_alloc device allocator for temporary buffers
 * @param stream: cuda stream to use
 */
                template <int TPB_X, typename T>
                void coo_remove_scalar(const int*   rows,
                                       const int*   cols,
                                       const T*     vals,
                                       int          nnz,
                                       int*         crows,
                                       int*         ccols,
                                       T*           cvals,
                                       int*         cnnz,
                                       int*         cur_cnnz,
                                       T            scalar,
                                       int          n,
                                       cudaStream_t stream)
                {
                    rmm::device_uvector<int> ex_scan(n, stream);
                    rmm::device_uvector<int> cur_ex_scan(n, stream);

                    RAFT_CUDA_TRY(cudaMemsetAsync(ex_scan.data(), 0, n * sizeof(int), stream));
                    RAFT_CUDA_TRY(cudaMemsetAsync(cur_ex_scan.data(), 0, n * sizeof(int), stream));

                    thrust::device_ptr<int> dev_cnnz = thrust::device_pointer_cast(cnnz);
                    thrust::device_ptr<int> dev_ex_scan
                        = thrust::device_pointer_cast(ex_scan.data());
                    thrust::exclusive_scan(
                        rmm::exec_policy(stream), dev_cnnz, dev_cnnz + n, dev_ex_scan);
                    RAFT_CUDA_TRY(cudaPeekAtLastError());

                    thrust::device_ptr<int> dev_cur_cnnz = thrust::device_pointer_cast(cur_cnnz);
                    thrust::device_ptr<int> dev_cur_ex_scan
                        = thrust::device_pointer_cast(cur_ex_scan.data());
                    thrust::exclusive_scan(
                        rmm::exec_policy(stream), dev_cur_cnnz, dev_cur_cnnz + n, dev_cur_ex_scan);
                    RAFT_CUDA_TRY(cudaPeekAtLastError());

                    dim3 grid(raft::ceildiv(n, TPB_X), 1, 1);
                    dim3 blk(TPB_X, 1, 1);

                    coo_remove_scalar_kernel<TPB_X><<<grid, blk, 0, stream>>>(rows,
                                                                              cols,
                                                                              vals,
                                                                              nnz,
                                                                              crows,
                                                                              ccols,
                                                                              cvals,
                                                                              dev_ex_scan.get(),
                                                                              dev_cur_ex_scan.get(),
                                                                              n,
                                                                              scalar);
                    RAFT_CUDA_TRY(cudaPeekAtLastError());
                }

                /**
 * @brief Removes the values matching a particular scalar from a COO formatted sparse matrix.
 *
 * @param in: input COO matrix
 * @param out: output COO matrix
 * @param scalar: scalar to remove from arrays
 * @param stream: cuda stream to use
 */
                template <int TPB_X, typename T>
                void coo_remove_scalar(COO<T>* in, COO<T>* out, T scalar, cudaStream_t stream)
                {
                    rmm::device_uvector<int> row_count_nz(in->n_rows, stream);
                    rmm::device_uvector<int> row_count(in->n_rows, stream);

                    RAFT_CUDA_TRY(
                        cudaMemsetAsync(row_count_nz.data(), 0, in->n_rows * sizeof(int), stream));
                    RAFT_CUDA_TRY(
                        cudaMemsetAsync(row_count.data(), 0, in->n_rows * sizeof(int), stream));

                    linalg::coo_degree(in->rows(), in->nnz, row_count.data(), stream);
                    RAFT_CUDA_TRY(cudaPeekAtLastError());

                    linalg::coo_degree_scalar(
                        in->rows(), in->vals(), in->nnz, scalar, row_count_nz.data(), stream);
                    RAFT_CUDA_TRY(cudaPeekAtLastError());

                    thrust::device_ptr<int> d_row_count_nz
                        = thrust::device_pointer_cast(row_count_nz.data());
                    int out_nnz = thrust::reduce(
                        rmm::exec_policy(stream), d_row_count_nz, d_row_count_nz + in->n_rows);

                    out->allocate(out_nnz, in->n_rows, in->n_cols, false, stream);

                    coo_remove_scalar<TPB_X, T>(in->rows(),
                                                in->cols(),
                                                in->vals(),
                                                in->nnz,
                                                out->rows(),
                                                out->cols(),
                                                out->vals(),
                                                row_count_nz.data(),
                                                row_count.data(),
                                                scalar,
                                                in->n_rows,
                                                stream);
                    RAFT_CUDA_TRY(cudaPeekAtLastError());
                }

                /**
 * @brief Removes zeros from a COO formatted sparse matrix.
 *
 * @param in: input COO matrix
 * @param out: output COO matrix
 * @param stream: cuda stream to use
 */
                template <int TPB_X, typename T>
                void coo_remove_zeros(COO<T>* in, COO<T>* out, cudaStream_t stream)
                {
                    coo_remove_scalar<TPB_X, T>(in, out, T(0.0), stream);
                }

            }; // namespace detail
        }; // namespace op
    }; // end NAMESPACE sparse
}; // end NAMESPACE raft
