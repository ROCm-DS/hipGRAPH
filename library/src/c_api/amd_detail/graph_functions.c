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
#include "hipgraph/hipgraph_c/graph_functions.h"

hipgraph_error_code_t
    hipgraph_create_vertex_pairs(const hipgraph_resource_handle_t*               handle,
                                 hipgraph_graph_t*                               graph,
                                 const hipgraph_type_erased_device_array_view_t* first,
                                 const hipgraph_type_erased_device_array_view_t* second,
                                 hipgraph_vertex_pairs_t**                       vertex_pairs,
                                 hipgraph_error_t**                              error)
{
    rocgraph_status rg_status
        = rocgraph_create_vertex_pairs((const rocgraph_handle_t*)handle,
                                       (rocgraph_graph_t*)graph,
                                       (const rocgraph_type_erased_device_array_view_t*)first,
                                       (const rocgraph_type_erased_device_array_view_t*)second,
                                       (rocgraph_vertex_pairs_t**)vertex_pairs,
                                       (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_vertex_pairs_get_first(hipgraph_vertex_pairs_t* vertex_pairs)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_vertex_pairs_get_first(
        (rocgraph_vertex_pairs_t*)vertex_pairs);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_vertex_pairs_get_second(hipgraph_vertex_pairs_t* vertex_pairs)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_vertex_pairs_get_second(
        (rocgraph_vertex_pairs_t*)vertex_pairs);
}

void hipgraph_vertex_pairs_free(hipgraph_vertex_pairs_t* vertex_pairs)
{
    rocgraph_vertex_pairs_free((rocgraph_vertex_pairs_t*)vertex_pairs);
}

hipgraph_error_code_t
    hipgraph_two_hop_neighbors(const hipgraph_resource_handle_t*               handle,
                               hipgraph_graph_t*                               graph,
                               const hipgraph_type_erased_device_array_view_t* start_vertices,
                               hipgraph_bool_t                                 do_expensive_check,
                               hipgraph_vertex_pairs_t**                       result,
                               hipgraph_error_t**                              error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_two_hop_neighbors(
        (const rocgraph_handle_t*)handle,
        (rocgraph_graph_t*)graph,
        (const rocgraph_type_erased_device_array_view_t*)start_vertices,
        rg_do_expensive_check,
        (rocgraph_vertex_pairs_t**)result,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_induced_subgraph_get_sources(hipgraph_induced_subgraph_result_t* induced_subgraph)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_induced_subgraph_get_sources(
        (rocgraph_induced_subgraph_result_t*)induced_subgraph);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_induced_subgraph_get_destinations(hipgraph_induced_subgraph_result_t* induced_subgraph)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_induced_subgraph_get_destinations(
        (rocgraph_induced_subgraph_result_t*)induced_subgraph);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_induced_subgraph_get_edge_weights(hipgraph_induced_subgraph_result_t* induced_subgraph)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_induced_subgraph_get_edge_weights(
        (rocgraph_induced_subgraph_result_t*)induced_subgraph);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_induced_subgraph_get_edge_ids(hipgraph_induced_subgraph_result_t* induced_subgraph)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_induced_subgraph_get_edge_ids(
        (rocgraph_induced_subgraph_result_t*)induced_subgraph);
}

hipgraph_type_erased_device_array_view_t* hipgraph_induced_subgraph_get_edge_type_ids(
    hipgraph_induced_subgraph_result_t* induced_subgraph)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_induced_subgraph_get_edge_type_ids(
        (rocgraph_induced_subgraph_result_t*)induced_subgraph);
}

hipgraph_type_erased_device_array_view_t* hipgraph_induced_subgraph_get_subgraph_offsets(
    hipgraph_induced_subgraph_result_t* induced_subgraph)
{
    return (hipgraph_type_erased_device_array_view_t*)
        rocgraph_induced_subgraph_get_subgraph_offsets(
            (rocgraph_induced_subgraph_result_t*)induced_subgraph);
}

void hipgraph_induced_subgraph_result_free(hipgraph_induced_subgraph_result_t* induced_subgraph)
{
    rocgraph_induced_subgraph_result_free((rocgraph_induced_subgraph_result_t*)induced_subgraph);
}

hipgraph_error_code_t hipgraph_extract_induced_subgraph(
    const hipgraph_resource_handle_t*               handle,
    hipgraph_graph_t*                               graph,
    const hipgraph_type_erased_device_array_view_t* subgraph_offsets,
    const hipgraph_type_erased_device_array_view_t* subgraph_vertices,
    hipgraph_bool_t                                 do_expensive_check,
    hipgraph_induced_subgraph_result_t**            result,
    hipgraph_error_t**                              error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_extract_induced_subgraph(
        (const rocgraph_handle_t*)handle,
        (rocgraph_graph_t*)graph,
        (const rocgraph_type_erased_device_array_view_t*)subgraph_offsets,
        (const rocgraph_type_erased_device_array_view_t*)subgraph_vertices,
        rg_do_expensive_check,
        (rocgraph_induced_subgraph_result_t**)result,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t
    hipgraph_allgather(const hipgraph_resource_handle_t*               handle,
                       const hipgraph_type_erased_device_array_view_t* src,
                       const hipgraph_type_erased_device_array_view_t* dst,
                       const hipgraph_type_erased_device_array_view_t* weights,
                       const hipgraph_type_erased_device_array_view_t* edge_ids,
                       const hipgraph_type_erased_device_array_view_t* edge_type_ids,
                       hipgraph_induced_subgraph_result_t**            result,
                       hipgraph_error_t**                              error)
{
    rocgraph_status rg_status
        = rocgraph_allgather((const rocgraph_handle_t*)handle,
                             (const rocgraph_type_erased_device_array_view_t*)src,
                             (const rocgraph_type_erased_device_array_view_t*)dst,
                             (const rocgraph_type_erased_device_array_view_t*)weights,
                             (const rocgraph_type_erased_device_array_view_t*)edge_ids,
                             (const rocgraph_type_erased_device_array_view_t*)edge_type_ids,
                             (rocgraph_induced_subgraph_result_t**)result,
                             (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}
#if 0
hipgraph_error_code_t
    hipgraph_count_multi_edges(const hipgraph_resource_handle_t* handle,
                               hipgraph_graph_t*                 graph,
                               hipgraph_bool_t                   do_expensive_check,
                               size_t*                           result,
                               hipgraph_error_t**                error)
{
    return rocgraph_status2hipgraph_error_code_t(rocgraph_count_multi_edges(
        (const rocgraph_handle_t*)handle,
        (rocgraph_graph_t*)graph,
        hipgraph_bool_t2rocgraph_bool(do_expensive_check),
        result,
        (rocgraph_error_t**)error));
}
#endif
hipgraph_error_code_t
    hipgraph_in_degrees(const hipgraph_resource_handle_t*               handle,
                        hipgraph_graph_t*                               graph,
                        const hipgraph_type_erased_device_array_view_t* source_vertices,
                        hipgraph_bool_t                                 do_expensive_check,
                        hipgraph_degrees_result_t**                     result,
                        hipgraph_error_t**                              error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status
        = rocgraph_in_degrees((const rocgraph_handle_t*)handle,
                              (rocgraph_graph_t*)graph,
                              (const rocgraph_type_erased_device_array_view_t*)source_vertices,
                              rg_do_expensive_check,
                              (rocgraph_degrees_result_t**)result,
                              (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t
    hipgraph_out_degrees(const hipgraph_resource_handle_t*               handle,
                         hipgraph_graph_t*                               graph,
                         const hipgraph_type_erased_device_array_view_t* source_vertices,
                         hipgraph_bool_t                                 do_expensive_check,
                         hipgraph_degrees_result_t**                     result,
                         hipgraph_error_t**                              error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status
        = rocgraph_out_degrees((const rocgraph_handle_t*)handle,
                               (rocgraph_graph_t*)graph,
                               (const rocgraph_type_erased_device_array_view_t*)source_vertices,
                               rg_do_expensive_check,
                               (rocgraph_degrees_result_t**)result,
                               (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t
    hipgraph_degrees(const hipgraph_resource_handle_t*               handle,
                     hipgraph_graph_t*                               graph,
                     const hipgraph_type_erased_device_array_view_t* source_vertices,
                     hipgraph_bool_t                                 do_expensive_check,
                     hipgraph_degrees_result_t**                     result,
                     hipgraph_error_t**                              error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status
        = rocgraph_degrees((const rocgraph_handle_t*)handle,
                           (rocgraph_graph_t*)graph,
                           (const rocgraph_type_erased_device_array_view_t*)source_vertices,
                           rg_do_expensive_check,
                           (rocgraph_degrees_result_t**)result,
                           (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_degrees_result_get_vertices(hipgraph_degrees_result_t* degrees_result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_degrees_result_get_vertices(
        (rocgraph_degrees_result_t*)degrees_result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_degrees_result_get_in_degrees(hipgraph_degrees_result_t* degrees_result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_degrees_result_get_in_degrees(
        (rocgraph_degrees_result_t*)degrees_result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_degrees_result_get_out_degrees(hipgraph_degrees_result_t* degrees_result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_degrees_result_get_out_degrees(
        (rocgraph_degrees_result_t*)degrees_result);
}

void hipgraph_degrees_result_free(hipgraph_degrees_result_t* degrees_result)
{
    rocgraph_degrees_result_free((rocgraph_degrees_result_t*)degrees_result);
}
