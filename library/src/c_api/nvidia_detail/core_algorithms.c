// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*! \file */
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "common.h"
#include <cugraph_c/core_algorithms.h>
#include "hipgraph/hipgraph_c/core_algorithms.h"

hipgraph_error_code_t
    hipgraph_core_result_create(const hipgraph_resource_handle_t*         handle,
                                hipgraph_type_erased_device_array_view_t* vertices,
                                hipgraph_type_erased_device_array_view_t* core_numbers,
                                hipgraph_core_result_t**                  core_result,
                                hipgraph_error_t**                        error)
{
    cugraph_error_code_t err;
    err = cugraph_core_result_create((const cugraph_resource_handle_t*)handle,
                                     (cugraph_type_erased_device_array_view_t*)vertices,
                                     (cugraph_type_erased_device_array_view_t*)core_numbers,
                                     (cugraph_core_result_t**)core_result,
                                     (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_core_result_get_vertices(hipgraph_core_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_core_result_get_vertices(
        (cugraph_core_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_core_result_get_core_numbers(hipgraph_core_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_core_result_get_core_numbers(
        (cugraph_core_result_t*)result);
}

void hipgraph_core_result_free(hipgraph_core_result_t* result)
{
    cugraph_core_result_free((cugraph_core_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_k_core_result_get_src_vertices(hipgraph_k_core_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_k_core_result_get_src_vertices(
        (cugraph_k_core_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_k_core_result_get_dst_vertices(hipgraph_k_core_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_k_core_result_get_dst_vertices(
        (cugraph_k_core_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_k_core_result_get_weights(hipgraph_k_core_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_k_core_result_get_weights(
        (cugraph_k_core_result_t*)result);
}

void hipgraph_k_core_result_free(hipgraph_k_core_result_t* result)
{
    cugraph_k_core_result_free((cugraph_k_core_result_t*)result);
}

hipgraph_error_code_t hipgraph_core_number(const hipgraph_resource_handle_t* handle,
                                           hipgraph_graph_t*                 graph,
                                           hipgraph_k_core_degree_type_t     degree_type,
                                           hipgraph_bool_t                   do_expensive_check,
                                           hipgraph_core_result_t**          result,
                                           hipgraph_error_t**                error)
{
    cugraph_error_code_t err;
    err = cugraph_core_number((const cugraph_resource_handle_t*)handle,
                              (cugraph_graph_t*)graph,
                              (cugraph_k_core_degree_type_t)degree_type,
                              (bool_t)do_expensive_check,
                              (cugraph_core_result_t**)result,
                              (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
}

hipgraph_error_code_t hipgraph_k_core(const hipgraph_resource_handle_t* handle,
                                      hipgraph_graph_t*                 graph,
                                      size_t                            k,
                                      hipgraph_k_core_degree_type_t     degree_type,
                                      const hipgraph_core_result_t*     core_result,
                                      hipgraph_bool_t                   do_expensive_check,
                                      hipgraph_k_core_result_t**        result,
                                      hipgraph_error_t**                error)
{
    cugraph_error_code_t err;
    err = cugraph_k_core((const cugraph_resource_handle_t*)handle,
                         (cugraph_graph_t*)graph,
                         k,
                         (cugraph_k_core_degree_type_t)degree_type,
                         (const cugraph_core_result_t*)core_result,
                         (bool_t)do_expensive_check,
                         (cugraph_k_core_result_t**)result,
                         (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
}
