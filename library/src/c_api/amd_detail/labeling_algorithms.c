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
#include "hipgraph/hipgraph_c/labeling_algorithms.h"

hipgraph_type_erased_device_array_view_t*
    hipgraph_labeling_result_get_vertices(hipgraph_labeling_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_labeling_result_get_vertices(
        (rocgraph_labeling_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_labeling_result_get_labels(hipgraph_labeling_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_labeling_result_get_labels(
        (rocgraph_labeling_result_t*)result);
}

void hipgraph_labeling_result_free(hipgraph_labeling_result_t* result)
{
    rocgraph_labeling_result_free((rocgraph_labeling_result_t*)result);
}

hipgraph_error_code_t hipgraph_weakly_connected_components(const hipgraph_resource_handle_t* handle,
                                                           hipgraph_graph_t*                 graph,
                                                           hipgraph_bool_t do_expensive_check,
                                                           hipgraph_labeling_result_t** result,
                                                           hipgraph_error_t**           error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status
        = rocgraph_weakly_connected_components((const rocgraph_handle_t*)handle,
                                               (rocgraph_graph_t*)graph,
                                               rg_do_expensive_check,
                                               (rocgraph_labeling_result_t**)result,
                                               (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t
    hipgraph_strongly_connected_components(const hipgraph_resource_handle_t* handle,
                                           hipgraph_graph_t*                 graph,
                                           hipgraph_bool_t                   do_expensive_check,
                                           hipgraph_labeling_result_t**      result,
                                           hipgraph_error_t**                error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status
        = rocgraph_strongly_connected_components((const rocgraph_handle_t*)handle,
                                                 (rocgraph_graph_t*)graph,
                                                 rg_do_expensive_check,
                                                 (rocgraph_labeling_result_t**)result,
                                                 (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}
