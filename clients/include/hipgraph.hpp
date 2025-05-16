// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
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

#pragma once
#ifndef _HIPGRAPH_HPP_
#define _HIPGRAPH_HPP_

#include <hipgraph/hipgraph.h>

namespace hipgraph
{

    template <typename T>
    struct floating_traits
    {
        using data_t = T;
    };

    template <typename T>
    using floating_data_t = typename floating_traits<T>::data_t;

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    template <typename T>
    hipgraphStatus_t hipgraphXcsrmv(hipgraphHandle_t         handle,
                                    hipgraphOperation_t      trans,
                                    int                      m,
                                    int                      n,
                                    int                      nnz,
                                    const T*                 alpha,
                                    const hipgraphMatDescr_t descr,
                                    const T*                 csr_val,
                                    const int*               csr_row_ptr,
                                    const int*               csr_col_ind,
                                    const T*                 x,
                                    const T*                 beta,
                                    T*                       y);
#endif

    template <typename T>
    hipgraphStatus_t hipgraphXnnz(hipgraphHandle_t         handle,
                                  hipgraphDirection_t      dirA,
                                  int                      m,
                                  int                      n,
                                  const hipgraphMatDescr_t descrA,
                                  const T*                 A,
                                  int                      lda,
                                  int*                     nnzPerRowColumn,
                                  int*                     nnzTotalDevHostPtr);

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    template <typename T>
    hipgraphStatus_t hipgraphXcsr2csc(hipgraphHandle_t    handle,
                                      int                 m,
                                      int                 n,
                                      int                 nnz,
                                      const T*            csr_val,
                                      const int*          csr_row_ptr,
                                      const int*          csr_col_ind,
                                      T*                  csc_val,
                                      int*                csc_row_ind,
                                      int*                csc_col_ptr,
                                      hipgraphAction_t    copy_values,
                                      hipgraphIndexBase_t idx_base);
#endif
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    template <typename T>
    hipgraphStatus_t hipgraphXgthr(hipgraphHandle_t    handle,
                                   int                 nnz,
                                   const T*            y,
                                   T*                  x_val,
                                   const int*          x_ind,
                                   hipgraphIndexBase_t idx_base);
#endif

    template <typename T>
    hipgraphStatus_t hipgraphXcsrcolor(hipgraphHandle_t          handle,
                                       int                       m,
                                       int                       nnz,
                                       const hipgraphMatDescr_t  descrA,
                                       const T*                  csrValA,
                                       const int*                csrRowPtrA,
                                       const int*                csrColIndA,
                                       const floating_data_t<T>* fractionToColor,
                                       int*                      ncolors,
                                       int*                      coloring,
                                       int*                      reordering,
                                       hipgraphColorInfo_t       info);

} // namespace hipgraph

#endif // _HIPGRAPH_HPP_
