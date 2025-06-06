// Copyright (c) 2019-2023, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <raft/core/resource/cusparse_handle.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/linalg/detail/transpose.h>

namespace raft
{
    namespace sparse
    {
        namespace linalg
        {

            /**
 * Transpose a set of CSR arrays into a set of CSC arrays.
 * @tparam value_idx : data type of the CSR index arrays
 * @tparam value_t : data type of the CSR data array
 * @param[in] handle : used for invoking cusparse
 * @param[in] csr_indptr : CSR row index array
 * @param[in] csr_indices : CSR column indices array
 * @param[in] csr_data : CSR data array
 * @param[out] csc_indptr : CSC row index array
 * @param[out] csc_indices : CSC column indices array
 * @param[out] csc_data : CSC data array
 * @param[in] csr_nrows : Number of rows in CSR
 * @param[in] csr_ncols : Number of columns in CSR
 * @param[in] nnz : Number of nonzeros of CSR
 * @param[in] stream : Cuda stream for ordering events
 */
            template <typename value_idx, typename value_t>
            void csr_transpose(raft::resources const& handle,
                               const value_idx*       csr_indptr,
                               const value_idx*       csr_indices,
                               const value_t*         csr_data,
                               value_idx*             csc_indptr,
                               value_idx*             csc_indices,
                               value_t*               csc_data,
                               value_idx              csr_nrows,
                               value_idx              csr_ncols,
                               value_idx              nnz,
                               cudaStream_t           stream)
            {
                detail::csr_transpose(resource::get_cusparse_handle(handle),
                                      csr_indptr,
                                      csr_indices,
                                      csr_data,
                                      csc_indptr,
                                      csc_indices,
                                      csc_data,
                                      csr_nrows,
                                      csr_ncols,
                                      nnz,
                                      stream);
            }

        }; // end NAMESPACE linalg
    }; // end NAMESPACE sparse
}; // end NAMESPACE raft
