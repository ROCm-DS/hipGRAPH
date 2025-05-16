// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*
 * Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 */

/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include "hipgraph/hipgraph_c/core_algorithms.h"

hipgraph_error_code_t
    hipgraph_core_result_create(const hipgraph_resource_handle_t*         handle,
                                hipgraph_type_erased_device_array_view_t* vertices,
                                hipgraph_type_erased_device_array_view_t* core_numbers,
                                hipgraph_core_result_t**                  core_result,
                                hipgraph_error_t**                        error)
{
    rocgraph_status rg_status
        = rocgraph_core_result_create((const rocgraph_handle_t*)handle,
                                      (rocgraph_type_erased_device_array_view_t*)vertices,
                                      (rocgraph_type_erased_device_array_view_t*)core_numbers,
                                      (rocgraph_core_result_t**)core_result,
                                      (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_core_result_get_vertices(hipgraph_core_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_core_result_get_vertices(
        (rocgraph_core_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_core_result_get_core_numbers(hipgraph_core_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_core_result_get_core_numbers(
        (rocgraph_core_result_t*)result);
}

void hipgraph_core_result_free(hipgraph_core_result_t* result)
{
    rocgraph_core_result_free((rocgraph_core_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_k_core_result_get_src_vertices(hipgraph_k_core_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_k_core_result_get_src_vertices(
        (rocgraph_k_core_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_k_core_result_get_dst_vertices(hipgraph_k_core_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_k_core_result_get_dst_vertices(
        (rocgraph_k_core_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_k_core_result_get_weights(hipgraph_k_core_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_k_core_result_get_weights(
        (rocgraph_k_core_result_t*)result);
}

void hipgraph_k_core_result_free(hipgraph_k_core_result_t* result)
{
    rocgraph_k_core_result_free((rocgraph_k_core_result_t*)result);
}

hipgraph_error_code_t hipgraph_core_number(const hipgraph_resource_handle_t* handle,
                                           hipgraph_graph_t*                 graph,
                                           hipgraph_k_core_degree_type_t     degree_type,
                                           hipgraph_bool_t                   do_expensive_check,
                                           hipgraph_core_result_t**          result,
                                           hipgraph_error_t**                error)
{
    rocgraph_k_core_degree_type rg_degree_type
        = hipgraph_k_core_degree_type_t2rocgraph_k_core_degree_type(degree_type);
    if(hghelper_rocgraph_k_core_degree_type_is_invalid(rg_degree_type))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_core_number((const rocgraph_handle_t*)handle,
                                                     (rocgraph_graph_t*)graph,
                                                     rg_degree_type,
                                                     rg_do_expensive_check,
                                                     (rocgraph_core_result_t**)result,
                                                     (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
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
    rocgraph_k_core_degree_type rg_degree_type
        = hipgraph_k_core_degree_type_t2rocgraph_k_core_degree_type(degree_type);
    if(hghelper_rocgraph_k_core_degree_type_is_invalid(rg_degree_type))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_k_core((const rocgraph_handle_t*)handle,
                                                (rocgraph_graph_t*)graph,
                                                k,
                                                rg_degree_type,
                                                (const rocgraph_core_result_t*)core_result,
                                                rg_do_expensive_check,
                                                (rocgraph_k_core_result_t**)result,
                                                (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}
