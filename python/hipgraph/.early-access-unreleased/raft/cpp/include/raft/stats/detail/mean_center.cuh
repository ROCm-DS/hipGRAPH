// Copyright (c) 2022, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/vectorized.cuh>

namespace raft
{
    namespace stats
    {
        namespace detail
        {

            /**
 * @brief Center the input matrix wrt its mean
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB threads per block of the cuda kernel launched
 * @param out the output mean-centered matrix
 * @param data input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param rowMajor whether input is row or col major
 * @param bcastAlongRows whether to broadcast vector along rows or columns
 * @param stream cuda stream where to launch work
 */
            template <typename Type, typename IdxType = int, int TPB = 256>
            void meanCenter(Type*        out,
                            const Type*  data,
                            const Type*  mu,
                            IdxType      D,
                            IdxType      N,
                            bool         rowMajor,
                            bool         bcastAlongRows,
                            cudaStream_t stream)
            {
                raft::linalg::matrixVectorOp(
                    out, data, mu, D, N, rowMajor, bcastAlongRows, raft::sub_op{}, stream);
            }

            /**
 * @brief Add the input matrix wrt its mean
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB threads per block of the cuda kernel launched
 * @param out the output mean-added matrix
 * @param data input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param rowMajor whether input is row or col major
 * @param bcastAlongRows whether to broadcast vector along rows or columns
 * @param stream cuda stream where to launch work
 */
            template <typename Type, typename IdxType = int, int TPB = 256>
            void meanAdd(Type*        out,
                         const Type*  data,
                         const Type*  mu,
                         IdxType      D,
                         IdxType      N,
                         bool         rowMajor,
                         bool         bcastAlongRows,
                         cudaStream_t stream)
            {
                raft::linalg::matrixVectorOp(
                    out, data, mu, D, N, rowMajor, bcastAlongRows, raft::add_op{}, stream);
            }

        }; // end namespace detail
    }; // end namespace stats
}; // end namespace raft
