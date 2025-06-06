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

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/detail/utils.h>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

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
        namespace convert
        {
            namespace detail
            {

                template <typename value_t>
                RAFT_KERNEL csr_to_dense_warp_per_row_kernel(int            n_cols,
                                                             const value_t* csrVal,
                                                             const int*     csrRowPtr,
                                                             const int*     csrColInd,
                                                             value_t*       a)
                {
                    int row = blockIdx.x;
                    int tid = threadIdx.x;

                    int colStart = csrRowPtr[row];
                    int colEnd   = csrRowPtr[row + 1];
                    int rowNnz   = colEnd - colStart;

                    for(int i = tid; i < rowNnz; i += blockDim.x)
                    {
                        int colIdx = colStart + i;
                        if(colIdx < colEnd)
                        {
                            int col               = csrColInd[colIdx];
                            a[row * n_cols + col] = csrVal[colIdx];
                        }
                    }
                }

                /**
 * Convert CSR arrays to a dense matrix in either row-
 * or column-major format. A custom kernel is used when
 * row-major output is desired since cusparse does not
 * output row-major.
 * @tparam value_idx : data type of the CSR index arrays
 * @tparam value_t : data type of the CSR value array
 * @param[in] handle : cusparse handle for conversion
 * @param[in] nrows : number of rows in CSR
 * @param[in] ncols : number of columns in CSR
 * @param[in] nnz : the number of nonzeros in CSR
 * @param[in] csr_indptr : CSR row index pointer array
 * @param[in] csr_indices : CSR column indices array
 * @param[in] csr_data : CSR data array
 * @param[in] lda : Leading dimension (used for col-major only)
 * @param[out] out : Dense output array of size nrows * ncols
 * @param[in] stream : Cuda stream for ordering events
 * @param[in] row_major : Is row-major output desired?
 */
                template <typename value_idx, typename value_t>
                void csr_to_dense(cusparseHandle_t handle,
                                  value_idx        nrows,
                                  value_idx        ncols,
                                  value_idx        nnz,
                                  const value_idx* csr_indptr,
                                  const value_idx* csr_indices,
                                  const value_t*   csr_data,
                                  value_idx        lda,
                                  value_t*         out,
                                  cudaStream_t     stream,
                                  bool             row_major = true)
                {
                    if(!row_major)
                    {
                        /**
     * If we need col-major, use cusparse.
     */
                        cusparseMatDescr_t out_mat;
                        RAFT_CUSPARSE_TRY(cusparseCreateMatDescr(&out_mat));
                        RAFT_CUSPARSE_TRY(
                            cusparseSetMatIndexBase(out_mat, CUSPARSE_INDEX_BASE_ZERO));
                        RAFT_CUSPARSE_TRY(
                            cusparseSetMatType(out_mat, CUSPARSE_MATRIX_TYPE_GENERAL));

                        size_t buffer_size;
                        RAFT_CUSPARSE_TRY(
                            raft::sparse::detail::cusparsecsr2dense_buffersize(handle,
                                                                               nrows,
                                                                               ncols,
                                                                               nnz,
                                                                               out_mat,
                                                                               csr_data,
                                                                               csr_indptr,
                                                                               csr_indices,
                                                                               out,
                                                                               lda,
                                                                               &buffer_size,
                                                                               stream));

                        rmm::device_uvector<char> buffer(buffer_size, stream);

                        RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecsr2dense(handle,
                                                                                  nrows,
                                                                                  ncols,
                                                                                  nnz,
                                                                                  out_mat,
                                                                                  csr_data,
                                                                                  csr_indptr,
                                                                                  csr_indices,
                                                                                  out,
                                                                                  lda,
                                                                                  buffer.data(),
                                                                                  stream));

                        RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyMatDescr(out_mat));
                    }
                    else
                    {
                        int blockdim = block_dim(ncols);
                        RAFT_CUDA_TRY(
                            cudaMemsetAsync(out, 0, nrows * ncols * sizeof(value_t), stream));
                        csr_to_dense_warp_per_row_kernel<<<nrows, blockdim, 0, stream>>>(
                            ncols, csr_data, csr_indptr, csr_indices, out);
                    }
                }

            }; // namespace detail
        }; // end NAMESPACE convert
    }; // end NAMESPACE sparse
}; // end NAMESPACE raft
