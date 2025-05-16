// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*
 * Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 */

/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
#include "hipgraph/hipgraph_c/traversal_algorithms.h"

hipgraph_type_erased_device_array_view_t*
    hipgraph_paths_result_get_vertices(hipgraph_paths_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_paths_result_get_vertices(
        (rocgraph_paths_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_paths_result_get_distances(hipgraph_paths_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_paths_result_get_distances(
        (rocgraph_paths_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_paths_result_get_predecessors(hipgraph_paths_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_paths_result_get_predecessors(
        (rocgraph_paths_result_t*)result);
}

void hipgraph_paths_result_free(hipgraph_paths_result_t* result)
{
    rocgraph_paths_result_free((rocgraph_paths_result_t*)result);
}

hipgraph_error_code_t hipgraph_bfs(const hipgraph_resource_handle_t*         handle,
                                   hipgraph_graph_t*                         graph,
                                   hipgraph_type_erased_device_array_view_t* sources,
                                   hipgraph_bool_t                           direction_optimizing,
                                   size_t                                    depth_limit,
                                   hipgraph_bool_t                           compute_predecessors,
                                   hipgraph_bool_t                           do_expensive_check,
                                   hipgraph_paths_result_t**                 result,
                                   hipgraph_error_t**                        error)
{
    rocgraph_bool rg_compute_predecessors = hipgraph_bool_t2rocgraph_bool(compute_predecessors);
    if(hghelper_rocgraph_bool_is_invalid(rg_compute_predecessors))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_direction_optimizing = hipgraph_bool_t2rocgraph_bool(direction_optimizing);
    if(hghelper_rocgraph_bool_is_invalid(rg_direction_optimizing))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_bfs((const rocgraph_handle_t*)handle,
                                             (rocgraph_graph_t*)graph,
                                             (rocgraph_type_erased_device_array_view_t*)sources,
                                             rg_direction_optimizing,
                                             depth_limit,
                                             rg_compute_predecessors,
                                             rg_do_expensive_check,
                                             (rocgraph_paths_result_t**)result,
                                             (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t hipgraph_sssp(const hipgraph_resource_handle_t* handle,
                                    hipgraph_graph_t*                 graph,
                                    size_t                            source,
                                    double                            cutoff,
                                    hipgraph_bool_t                   compute_predecessors,
                                    hipgraph_bool_t                   do_expensive_check,
                                    hipgraph_paths_result_t**         result,
                                    hipgraph_error_t**                error)
{
    rocgraph_bool rg_compute_predecessors = hipgraph_bool_t2rocgraph_bool(compute_predecessors);
    if(hghelper_rocgraph_bool_is_invalid(rg_compute_predecessors))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_sssp((const rocgraph_handle_t*)handle,
                                              (rocgraph_graph_t*)graph,
                                              source,
                                              cutoff,
                                              rg_compute_predecessors,
                                              rg_do_expensive_check,
                                              (rocgraph_paths_result_t**)result,
                                              (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t
    hipgraph_extract_paths(const hipgraph_resource_handle_t*               handle,
                           hipgraph_graph_t*                               graph,
                           const hipgraph_type_erased_device_array_view_t* sources,
                           const hipgraph_paths_result_t*                  paths_result,
                           const hipgraph_type_erased_device_array_view_t* destinations,
                           hipgraph_extract_paths_result_t**               result,
                           hipgraph_error_t**                              error)
{
    rocgraph_status rg_status
        = rocgraph_extract_paths((const rocgraph_handle_t*)handle,
                                 (rocgraph_graph_t*)graph,
                                 (const rocgraph_type_erased_device_array_view_t*)sources,
                                 (const rocgraph_paths_result_t*)paths_result,
                                 (const rocgraph_type_erased_device_array_view_t*)destinations,
                                 (rocgraph_extract_paths_result_t**)result,
                                 (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

size_t hipgraph_extract_paths_result_get_max_path_length(hipgraph_extract_paths_result_t* result)
{
    return (size_t)rocgraph_extract_paths_result_get_max_path_length(
        (rocgraph_extract_paths_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_extract_paths_result_get_paths(hipgraph_extract_paths_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_extract_paths_result_get_paths(
        (rocgraph_extract_paths_result_t*)result);
}

void hipgraph_extract_paths_result_free(hipgraph_extract_paths_result_t* result)
{
    rocgraph_extract_paths_result_free((rocgraph_extract_paths_result_t*)result);
}
