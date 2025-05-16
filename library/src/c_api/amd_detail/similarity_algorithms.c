// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*
 * Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 */

/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common.h"
#include <rocgraph/rocgraph.h>
#include "hipgraph/hipgraph_c/similarity_algorithms.h"

hipgraph_vertex_pairs_t*
    hipgraph_similarity_result_get_vertex_pairs(hipgraph_similarity_result_t* result)
{
    return (hipgraph_vertex_pairs_t*)rocgraph_similarity_result_get_vertex_pairs(
        (rocgraph_similarity_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_similarity_result_get_similarity(hipgraph_similarity_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_similarity_result_get_similarity(
        (rocgraph_similarity_result_t*)result);
}

void hipgraph_similarity_result_free(hipgraph_similarity_result_t* result)
{
    rocgraph_similarity_result_free((rocgraph_similarity_result_t*)result);
}

hipgraph_error_code_t hipgraph_jaccard_coefficients(const hipgraph_resource_handle_t* handle,
                                                    hipgraph_graph_t*                 graph,
                                                    const hipgraph_vertex_pairs_t*    vertex_pairs,
                                                    hipgraph_bool_t                   use_weight,
                                                    hipgraph_bool_t do_expensive_check,
                                                    hipgraph_similarity_result_t** result,
                                                    hipgraph_error_t**             error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_use_weight = hipgraph_bool_t2rocgraph_bool(use_weight);
    if(hghelper_rocgraph_bool_is_invalid(rg_use_weight))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status
        = rocgraph_jaccard_coefficients((const rocgraph_handle_t*)handle,
                                        (rocgraph_graph_t*)graph,
                                        (const rocgraph_vertex_pairs_t*)vertex_pairs,
                                        rg_use_weight,
                                        rg_do_expensive_check,
                                        (rocgraph_similarity_result_t**)result,
                                        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t hipgraph_sorensen_coefficients(const hipgraph_resource_handle_t* handle,
                                                     hipgraph_graph_t*                 graph,
                                                     const hipgraph_vertex_pairs_t*    vertex_pairs,
                                                     hipgraph_bool_t                   use_weight,
                                                     hipgraph_bool_t do_expensive_check,
                                                     hipgraph_similarity_result_t** result,
                                                     hipgraph_error_t**             error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_use_weight = hipgraph_bool_t2rocgraph_bool(use_weight);
    if(hghelper_rocgraph_bool_is_invalid(rg_use_weight))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status
        = rocgraph_sorensen_coefficients((const rocgraph_handle_t*)handle,
                                         (rocgraph_graph_t*)graph,
                                         (const rocgraph_vertex_pairs_t*)vertex_pairs,
                                         rg_use_weight,
                                         rg_do_expensive_check,
                                         (rocgraph_similarity_result_t**)result,
                                         (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t hipgraph_overlap_coefficients(const hipgraph_resource_handle_t* handle,
                                                    hipgraph_graph_t*                 graph,
                                                    const hipgraph_vertex_pairs_t*    vertex_pairs,
                                                    hipgraph_bool_t                   use_weight,
                                                    hipgraph_bool_t do_expensive_check,
                                                    hipgraph_similarity_result_t** result,
                                                    hipgraph_error_t**             error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_use_weight = hipgraph_bool_t2rocgraph_bool(use_weight);
    if(hghelper_rocgraph_bool_is_invalid(rg_use_weight))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status
        = rocgraph_overlap_coefficients((const rocgraph_handle_t*)handle,
                                        (rocgraph_graph_t*)graph,
                                        (const rocgraph_vertex_pairs_t*)vertex_pairs,
                                        rg_use_weight,
                                        rg_do_expensive_check,
                                        (rocgraph_similarity_result_t**)result,
                                        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
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
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_use_weight = hipgraph_bool_t2rocgraph_bool(use_weight);
    if(hghelper_rocgraph_bool_is_invalid(rg_use_weight))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_all_pairs_jaccard_coefficients(
        (const rocgraph_handle_t*)handle,
        (rocgraph_graph_t*)graph,
        (const rocgraph_type_erased_device_array_view_t*)vertices,
        rg_use_weight,
        topk,
        rg_do_expensive_check,
        (rocgraph_similarity_result_t**)result,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
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
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_use_weight = hipgraph_bool_t2rocgraph_bool(use_weight);
    if(hghelper_rocgraph_bool_is_invalid(rg_use_weight))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_all_pairs_sorensen_coefficients(
        (const rocgraph_handle_t*)handle,
        (rocgraph_graph_t*)graph,
        (const rocgraph_type_erased_device_array_view_t*)vertices,
        rg_use_weight,
        topk,
        rg_do_expensive_check,
        (rocgraph_similarity_result_t**)result,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
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
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_use_weight = hipgraph_bool_t2rocgraph_bool(use_weight);
    if(hghelper_rocgraph_bool_is_invalid(rg_use_weight))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_all_pairs_overlap_coefficients(
        (const rocgraph_handle_t*)handle,
        (rocgraph_graph_t*)graph,
        (const rocgraph_type_erased_device_array_view_t*)vertices,
        rg_use_weight,
        topk,
        rg_do_expensive_check,
        (rocgraph_similarity_result_t**)result,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}
