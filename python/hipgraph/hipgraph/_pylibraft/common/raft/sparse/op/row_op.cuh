// Copyright (c) 2019-2023, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#ifndef __SPARSE_ROW_OP_H
#define __SPARSE_ROW_OP_H
#pragma once

#include <raft/core/resources.hpp>
#include <raft/sparse/op/detail/row_op.cuh>

namespace raft
{
    namespace sparse
    {
        namespace op
        {

            /**
 * @brief Perform a custom row operation on a CSR matrix in batches.
 * @tparam T numerical type of row_ind array
 * @tparam TPB_X number of threads per block to use for underlying kernel
 * @tparam Lambda type of custom operation function
 * @param row_ind the CSR row_ind array to perform parallel operations over
 * @param n_rows total number vertices in graph
 * @param nnz number of non-zeros
 * @param op custom row operation functor accepting the row and beginning index.
 * @param stream cuda stream to use
 */
            template <typename Index_, typename Lambda = auto(Index_, Index_, Index_)->void>
            void csr_row_op(
                const Index_* row_ind, Index_ n_rows, Index_ nnz, Lambda op, cudaStream_t stream)
            {
                detail::csr_row_op<Index_, 128, Lambda>(row_ind, n_rows, nnz, op, stream);
            }

        }; // namespace op
    }; // end NAMESPACE sparse
}; // end NAMESPACE raft

#endif
