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

#include <raft/common/nvtx.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/norm_types.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/detail/utils.h>
#include <raft/sparse/op/row_op.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cuda_runtime.h>

#include <hipblas/hipblas.h>
#else
#include <cuda_runtime.h>

#include <cusparse_v2.h>
#endif
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <stdio.h>

#include <iostream>
#include <limits>

namespace raft
{
    namespace sparse
    {
        namespace linalg
        {
            namespace detail
            {

                template <int TPB_X = 64, typename T>
                RAFT_KERNEL csr_row_normalize_l1_kernel(
                    // @TODO: This can be done much more parallel by
                    // having threads in a warp compute the sum in parallel
                    // over each row and then divide the values in parallel.
                    const int* ia, // csr row ex_scan (sorted by row)
                    const T*   vals,
                    int        nnz, // array of values and number of non-zeros
                    int        m, // num rows in csr
                    T*         result)
                { // output array

                    // row-based matrix 1 thread per row
                    int row = (blockIdx.x * TPB_X) + threadIdx.x;

                    // sum all vals_arr for row and divide each val by sum
                    if(row < m)
                    {
                        int start_idx = ia[row];
                        int stop_idx  = 0;
                        if(row < m - 1)
                        {
                            stop_idx = ia[row + 1];
                        }
                        else
                            stop_idx = nnz;

                        T sum = T(0.0);
                        for(int j = start_idx; j < stop_idx; j++)
                        {
                            sum = sum + fabs(vals[j]);
                        }

                        for(int j = start_idx; j < stop_idx; j++)
                        {
                            if(sum != 0.0)
                            {
                                T val     = vals[j];
                                result[j] = val / sum;
                            }
                            else
                            {
                                result[j] = 0.0;
                            }
                        }
                    }
                }

                /**
 * @brief Perform L1 normalization on the rows of a given CSR-formatted sparse matrix
 *
 * @param ia: row_ind array
 * @param vals: data array
 * @param nnz: size of data array
 * @param m: size of row_ind array
 * @param result: l1 normalized data array
 * @param stream: cuda stream to use
 */
                template <int TPB_X = 64, typename T>
                void csr_row_normalize_l1(const int* ia, // csr row ex_scan (sorted by row)
                                          const T*   vals,
                                          int        nnz, // array of values and number of non-zeros
                                          int        m, // num rows in csr
                                          T*         result,
                                          cudaStream_t stream)
                { // output array

                    dim3 grid(raft::ceildiv(m, TPB_X), 1, 1);
                    dim3 blk(TPB_X, 1, 1);

                    csr_row_normalize_l1_kernel<TPB_X, T>
                        <<<grid, blk, 0, stream>>>(ia, vals, nnz, m, result);
                    RAFT_CUDA_TRY(cudaGetLastError());
                }

                template <int TPB_X = 64, typename T>
                RAFT_KERNEL csr_row_normalize_max_kernel(
                    // @TODO: This can be done much more parallel by
                    // having threads in a warp compute the sum in parallel
                    // over each row and then divide the values in parallel.
                    const int* ia, // csr row ind array (sorted by row)
                    const T*   vals,
                    int        nnz, // array of values and number of non-zeros
                    int        m, // num total rows in csr
                    T*         result)
                { // output array

                    // row-based matrix 1 thread per row
                    int row = (blockIdx.x * TPB_X) + threadIdx.x;

                    // find max across columns and divide
                    if(row < m)
                    {
                        int start_idx = ia[row];
                        int stop_idx  = 0;
                        if(row < m - 1)
                        {
                            stop_idx = ia[row + 1];
                        }
                        else
                            stop_idx = nnz;

                        T max = std::numeric_limits<float>::min();
                        for(int j = start_idx; j < stop_idx; j++)
                        {
                            if(vals[j] > max)
                                max = vals[j];
                        }

                        // divide nonzeros in current row by max
                        for(int j = start_idx; j < stop_idx; j++)
                        {
                            if(max != 0.0 && max > std::numeric_limits<float>::min())
                            {
                                T val     = vals[j];
                                result[j] = val / max;
                            }
                            else
                            {
                                result[j] = 0.0;
                            }
                        }
                    }
                }

                /**
 * @brief Perform L_inf normalization on a given CSR-formatted sparse matrix
 *
 * @param ia: row_ind array
 * @param vals: data array
 * @param nnz: size of data array
 * @param m: size of row_ind array
 * @param result: l1 normalized data array
 * @param stream: cuda stream to use
 */

                template <int TPB_X = 64, typename T>
                void csr_row_normalize_max(const int* ia, // csr row ind array (sorted by row)
                                           const T*   vals,
                                           int nnz, // array of values and number of non-zeros
                                           int m, // num total rows in csr
                                           T*  result,
                                           cudaStream_t stream)
                {
                    dim3 grid(raft::ceildiv(m, TPB_X), 1, 1);
                    dim3 blk(TPB_X, 1, 1);

                    csr_row_normalize_max_kernel<TPB_X, T>
                        <<<grid, blk, 0, stream>>>(ia, vals, nnz, m, result);
                    RAFT_CUDA_TRY(cudaGetLastError());
                }

                template <typename Type,
                          typename IdxType      = int,
                          typename MainLambda   = raft::identity_op,
                          typename ReduceLambda = raft::add_op,
                          typename FinalLambda  = raft::identity_op>
                void csr_row_op_wrapper(const IdxType* ia,
                                        const Type*    data,
                                        IdxType        nnz,
                                        IdxType        N,
                                        Type           init,
                                        Type*          norm,
                                        cudaStream_t   stream,
                                        MainLambda     main_op   = raft::identity_op(),
                                        ReduceLambda   reduce_op = raft::add_op(),
                                        FinalLambda    final_op  = raft::identity_op())
                {
                    op::csr_row_op<IdxType>(
                        ia,
                        N,
                        nnz,
                        [data, init, norm, main_op, reduce_op, final_op] __device__(
                            IdxType row, IdxType start_idx, IdxType stop_idx) {
                            norm[row] = init;
                            for(IdxType i = start_idx; i < stop_idx; i++)
                                norm[row] = final_op(reduce_op(norm[row], main_op(data[i])));
                        },
                        stream);
                }

                template <typename Type, typename IdxType, typename Lambda>
                void rowNormCsrCaller(const IdxType*         ia,
                                      const Type*            data,
                                      IdxType                nnz,
                                      IdxType                N,
                                      Type*                  norm,
                                      raft::linalg::NormType type,
                                      Lambda                 fin_op,
                                      cudaStream_t           stream)
                {
                    switch(type)
                    {
                    case raft::linalg::NormType::L1Norm:
                        csr_row_op_wrapper(ia,
                                           data,
                                           nnz,
                                           N,
                                           (Type)0,
                                           norm,
                                           stream,
                                           raft::abs_op(),
                                           raft::add_op(),
                                           fin_op);
                        break;
                    case raft::linalg::NormType::L2Norm:
                        csr_row_op_wrapper(ia,
                                           data,
                                           nnz,
                                           N,
                                           (Type)0,
                                           norm,
                                           stream,
                                           raft::sq_op(),
                                           raft::add_op(),
                                           fin_op);
                        break;
                    case raft::linalg::NormType::LinfNorm:
                        csr_row_op_wrapper(ia,
                                           data,
                                           nnz,
                                           N,
                                           (Type)0,
                                           norm,
                                           stream,
                                           raft::abs_op(),
                                           raft::max_op(),
                                           fin_op);
                        break;
                    default:
                        THROW("Unsupported norm type: %d", type);
                    };
                }

            }; // end NAMESPACE detail
        }; // end NAMESPACE linalg
    }; // end NAMESPACE sparse
}; // end NAMESPACE raft
