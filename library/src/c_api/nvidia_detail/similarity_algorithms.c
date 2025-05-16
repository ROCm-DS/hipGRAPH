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
#include <cugraph_c/similarity_algorithms.h>
#include "hipgraph/hipgraph_c/similarity_algorithms.h"

hipgraph_vertex_pairs_t*
    hipgraph_similarity_result_get_vertex_pairs(hipgraph_similarity_result_t* result)
{
    return (hipgraph_vertex_pairs_t*)cugraph_similarity_result_get_vertex_pairs(
        (cugraph_similarity_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_similarity_result_get_similarity(hipgraph_similarity_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_similarity_result_get_similarity(
        (cugraph_similarity_result_t*)result);
}
void hipgraph_similarity_result_free(hipgraph_similarity_result_t* result)
{
    cugraph_similarity_result_free((cugraph_similarity_result_t*)result);
}

hipgraph_error_code_t hipgraph_jaccard_coefficients(const hipgraph_resource_handle_t* handle,
                                                    hipgraph_graph_t*                 graph,
                                                    const hipgraph_vertex_pairs_t*    vertex_pairs,
                                                    hipgraph_bool_t                   use_weight,
                                                    hipgraph_bool_t do_expensive_check,
                                                    hipgraph_similarity_result_t** result,
                                                    hipgraph_error_t**             error)
{
    cugraph_error_code_t err;
    err = cugraph_jaccard_coefficients((const cugraph_resource_handle_t*)handle,
                                       (cugraph_graph_t*)graph,
                                       (const cugraph_vertex_pairs_t*)vertex_pairs,
                                       (bool_t)use_weight,
                                       (bool_t)do_expensive_check,
                                       (cugraph_similarity_result_t**)result,
                                       (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
}

hipgraph_error_code_t hipgraph_sorensen_coefficients(const hipgraph_resource_handle_t* handle,
                                                     hipgraph_graph_t*                 graph,
                                                     const hipgraph_vertex_pairs_t*    vertex_pairs,
                                                     hipgraph_bool_t                   use_weight,
                                                     hipgraph_bool_t do_expensive_check,
                                                     hipgraph_similarity_result_t** result,
                                                     hipgraph_error_t**             error)
{
    cugraph_error_code_t err;
    err = cugraph_sorensen_coefficients((const cugraph_resource_handle_t*)handle,
                                        (cugraph_graph_t*)graph,
                                        (const cugraph_vertex_pairs_t*)vertex_pairs,
                                        (bool_t)use_weight,
                                        (bool_t)do_expensive_check,
                                        (cugraph_similarity_result_t**)result,
                                        (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
}

hipgraph_error_code_t hipgraph_overlap_coefficients(const hipgraph_resource_handle_t* handle,
                                                    hipgraph_graph_t*                 graph,
                                                    const hipgraph_vertex_pairs_t*    vertex_pairs,
                                                    hipgraph_bool_t                   use_weight,
                                                    hipgraph_bool_t do_expensive_check,
                                                    hipgraph_similarity_result_t** result,
                                                    hipgraph_error_t**             error)
{
    cugraph_error_code_t err;
    err = cugraph_overlap_coefficients((const cugraph_resource_handle_t*)handle,
                                       (cugraph_graph_t*)graph,
                                       (const cugraph_vertex_pairs_t*)vertex_pairs,
                                       (bool_t)use_weight,
                                       (bool_t)do_expensive_check,
                                       (cugraph_similarity_result_t**)result,
                                       (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
}

hipgraph_error_code_t hipgraph_all_pairs_jaccard_coefficients(
    const hipgraph_resource_handle_t*               handle,
    hipgraph_graph_t*                               graph,
    const hipgraph_type_erased_device_array_view_t* vertices,
    hipgraph_bool_t                                 use_weight,
    size_t                                          topk,
    hipgraph_bool_t                                 do_expensive_check,
    hipgraph_similarity_result_t**                  result,
    hipgraph_error_t**                              error)
{
    cugraph_error_code_t err;
    err = cugraph_all_pairs_jaccard_coefficients(
        (const cugraph_resource_handle_t*)handle,
        (cugraph_graph_t*)graph,
        (const cugraph_type_erased_device_array_view_t*)vertices,
        (bool_t)use_weight,
        topk,
        (bool_t)do_expensive_check,
        (cugraph_similarity_result_t**)result,
        (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
}

hipgraph_error_code_t hipgraph_all_pairs_sorensen_coefficients(
    const hipgraph_resource_handle_t*               handle,
    hipgraph_graph_t*                               graph,
    const hipgraph_type_erased_device_array_view_t* vertices,
    hipgraph_bool_t                                 use_weight,
    size_t                                          topk,
    hipgraph_bool_t                                 do_expensive_check,
    hipgraph_similarity_result_t**                  result,
    hipgraph_error_t**                              error)
{
    cugraph_error_code_t err;
    err = cugraph_all_pairs_sorensen_coefficients(
        (const cugraph_resource_handle_t*)handle,
        (cugraph_graph_t*)graph,
        (const cugraph_type_erased_device_array_view_t*)vertices,
        (bool_t)use_weight,
        topk,
        (bool_t)do_expensive_check,
        (cugraph_similarity_result_t**)result,
        (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
}

hipgraph_error_code_t hipgraph_all_pairs_overlap_coefficients(
    const hipgraph_resource_handle_t*               handle,
    hipgraph_graph_t*                               graph,
    const hipgraph_type_erased_device_array_view_t* vertices,
    hipgraph_bool_t                                 use_weight,
    size_t                                          topk,
    hipgraph_bool_t                                 do_expensive_check,
    hipgraph_similarity_result_t**                  result,
    hipgraph_error_t**                              error)
{
    cugraph_error_code_t err;
    err = cugraph_all_pairs_overlap_coefficients(
        (const cugraph_resource_handle_t*)handle,
        (cugraph_graph_t*)graph,
        (const cugraph_type_erased_device_array_view_t*)vertices,
        (bool_t)use_weight,
        topk,
        (bool_t)do_expensive_check,
        (cugraph_similarity_result_t**)result,
        (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
}
