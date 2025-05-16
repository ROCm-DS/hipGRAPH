// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0
/* ************************************************************************
* Copyright (C) 2018-2021 Advanced Micro Devices, Inc. All rights Reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*
* ************************************************************************ */

#include "hipgraph.hpp"

#include <hipgraph/hipgraph.h>

namespace hipgraph
{

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    template <>
    hipgraphStatus_t hipgraphXgthr(hipgraphHandle_t    handle,
                                   int                 nnz,
                                   const float*        y,
                                   float*              x_val,
                                   const int*          x_ind,
                                   hipgraphIndexBase_t idx_base)
    {
        return hipgraphSgthr(handle, nnz, y, x_val, x_ind, idx_base);
    }

    template <>
    hipgraphStatus_t hipgraphXgthr(hipgraphHandle_t    handle,
                                   int                 nnz,
                                   const double*       y,
                                   double*             x_val,
                                   const int*          x_ind,
                                   hipgraphIndexBase_t idx_base)
    {
        return hipgraphDgthr(handle, nnz, y, x_val, x_ind, idx_base);
    }

#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    template <>
    hipgraphStatus_t hipgraphXcsrmv(hipgraphHandle_t         handle,
                                    hipgraphOperation_t      trans,
                                    int                      m,
                                    int                      n,
                                    int                      nnz,
                                    const float*             alpha,
                                    const hipgraphMatDescr_t descr,
                                    const float*             csr_val,
                                    const int*               csr_row_ptr,
                                    const int*               csr_col_ind,
                                    const float*             x,
                                    const float*             beta,
                                    float*                   y)
    {
        return hipgraphScsrmv(
            handle, trans, m, n, nnz, alpha, descr, csr_val, csr_row_ptr, csr_col_ind, x, beta, y);
    }

    template <>
    hipgraphStatus_t hipgraphXcsrmv(hipgraphHandle_t         handle,
                                    hipgraphOperation_t      trans,
                                    int                      m,
                                    int                      n,
                                    int                      nnz,
                                    const double*            alpha,
                                    const hipgraphMatDescr_t descr,
                                    const double*            csr_val,
                                    const int*               csr_row_ptr,
                                    const int*               csr_col_ind,
                                    const double*            x,
                                    const double*            beta,
                                    double*                  y)
    {
        return hipgraphDcsrmv(
            handle, trans, m, n, nnz, alpha, descr, csr_val, csr_row_ptr, csr_col_ind, x, beta, y);
    }

#endif

    template <>
    hipgraphStatus_t hipgraphXnnz(hipgraphHandle_t         handle,
                                  hipgraphDirection_t      dirA,
                                  int                      m,
                                  int                      n,
                                  const hipgraphMatDescr_t descrA,
                                  const float*             A,
                                  int                      lda,
                                  int*                     nnzPerRowColumn,
                                  int*                     nnzTotalDevHostPtr)
    {
        return hipgraphSnnz(
            handle, dirA, m, n, descrA, A, lda, nnzPerRowColumn, nnzTotalDevHostPtr);
    }

    template <>
    hipgraphStatus_t hipgraphXnnz(hipgraphHandle_t         handle,
                                  hipgraphDirection_t      dirA,
                                  int                      m,
                                  int                      n,
                                  const hipgraphMatDescr_t descrA,
                                  const double*            A,
                                  int                      lda,
                                  int*                     nnzPerRowColumn,
                                  int*                     nnzTotalDevHostPtr)
    {
        return hipgraphDnnz(
            handle, dirA, m, n, descrA, A, lda, nnzPerRowColumn, nnzTotalDevHostPtr);
    }

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    template <>
    hipgraphStatus_t hipgraphXcsr2csc(hipgraphHandle_t    handle,
                                      int                 m,
                                      int                 n,
                                      int                 nnz,
                                      const float*        csr_val,
                                      const int*          csr_row_ptr,
                                      const int*          csr_col_ind,
                                      float*              csc_val,
                                      int*                csc_row_ind,
                                      int*                csc_col_ptr,
                                      hipgraphAction_t    copy_values,
                                      hipgraphIndexBase_t idx_base)
    {
        return hipgraphScsr2csc(handle,
                                m,
                                n,
                                nnz,
                                csr_val,
                                csr_row_ptr,
                                csr_col_ind,
                                csc_val,
                                csc_row_ind,
                                csc_col_ptr,
                                copy_values,
                                idx_base);
    }

    template <>
    hipgraphStatus_t hipgraphXcsr2csc(hipgraphHandle_t    handle,
                                      int                 m,
                                      int                 n,
                                      int                 nnz,
                                      const double*       csr_val,
                                      const int*          csr_row_ptr,
                                      const int*          csr_col_ind,
                                      double*             csc_val,
                                      int*                csc_row_ind,
                                      int*                csc_col_ptr,
                                      hipgraphAction_t    copy_values,
                                      hipgraphIndexBase_t idx_base)
    {
        return hipgraphDcsr2csc(handle,
                                m,
                                n,
                                nnz,
                                csr_val,
                                csr_row_ptr,
                                csr_col_ind,
                                csc_val,
                                csc_row_ind,
                                csc_col_ptr,
                                copy_values,
                                idx_base);
    }

#endif

    template <>
    hipgraphStatus_t hipgraphXcsrcolor(hipgraphHandle_t         handle,
                                       int                      m,
                                       int                      nnz,
                                       const hipgraphMatDescr_t descrA,
                                       const float*             csrValA,
                                       const int*               csrRowPtrA,
                                       const int*               csrColIndA,
                                       const float*             fractionToColor,
                                       int*                     ncolors,
                                       int*                     coloring,
                                       int*                     reordering,
                                       hipgraphColorInfo_t      info)
    {
        return hipgraphScsrcolor(handle,
                                 m,
                                 nnz,
                                 descrA,
                                 csrValA,
                                 csrRowPtrA,
                                 csrColIndA,
                                 fractionToColor,
                                 ncolors,
                                 coloring,
                                 reordering,
                                 info);
    }

    template <>
    hipgraphStatus_t hipgraphXcsrcolor(hipgraphHandle_t         handle,
                                       int                      m,
                                       int                      nnz,
                                       const hipgraphMatDescr_t descrA,
                                       const double*            csrValA,
                                       const int*               csrRowPtrA,
                                       const int*               csrColIndA,
                                       const double*            fractionToColor,
                                       int*                     ncolors,
                                       int*                     coloring,
                                       int*                     reordering,
                                       hipgraphColorInfo_t      info)
    {
        return hipgraphDcsrcolor(handle,
                                 m,
                                 nnz,
                                 descrA,
                                 csrValA,
                                 csrRowPtrA,
                                 csrColIndA,
                                 fractionToColor,
                                 ncolors,
                                 coloring,
                                 reordering,
                                 info);
    }

} // namespace hipgraph
