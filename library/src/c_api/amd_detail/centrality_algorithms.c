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
#include "hipgraph/hipgraph_c/centrality_algorithms.h"

hipgraph_type_erased_device_array_view_t*
    hipgraph_centrality_result_get_vertices(hipgraph_centrality_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_centrality_result_get_vertices(
        (rocgraph_centrality_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_centrality_result_get_values(hipgraph_centrality_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_centrality_result_get_values(
        (rocgraph_centrality_result_t*)result);
}

size_t hipgraph_centrality_result_get_num_iterations(hipgraph_centrality_result_t* result)
{
    return (size_t)rocgraph_centrality_result_get_num_iterations(
        (rocgraph_centrality_result_t*)result);
}

hipgraph_bool_t hipgraph_centrality_result_converged(hipgraph_centrality_result_t* result)
{
    return (hipgraph_bool_t)rocgraph_centrality_result_converged(
        (rocgraph_centrality_result_t*)result);
}

void hipgraph_centrality_result_free(hipgraph_centrality_result_t* result)
{
    rocgraph_centrality_result_free((rocgraph_centrality_result_t*)result);
}

hipgraph_error_code_t hipgraph_pagerank(
    const hipgraph_resource_handle_t*               handle,
    hipgraph_graph_t*                               graph,
    const hipgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices,
    const hipgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
    const hipgraph_type_erased_device_array_view_t* initial_guess_vertices,
    const hipgraph_type_erased_device_array_view_t* initial_guess_values,
    double                                          alpha,
    double                                          epsilon,
    size_t                                          max_iterations,
    hipgraph_bool_t                                 do_expensive_check,
    hipgraph_centrality_result_t**                  result,
    hipgraph_error_t**                              error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_pagerank(
        (const rocgraph_handle_t*)handle,
        (rocgraph_graph_t*)graph,
        (const rocgraph_type_erased_device_array_view_t*)precomputed_vertex_out_weight_vertices,
        (const rocgraph_type_erased_device_array_view_t*)precomputed_vertex_out_weight_sums,
        (const rocgraph_type_erased_device_array_view_t*)initial_guess_vertices,
        (const rocgraph_type_erased_device_array_view_t*)initial_guess_values,
        alpha,
        epsilon,
        max_iterations,
        rg_do_expensive_check,
        (rocgraph_centrality_result_t**)result,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t hipgraph_pagerank_allow_nonconvergence(
    const hipgraph_resource_handle_t*               handle,
    hipgraph_graph_t*                               graph,
    const hipgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices,
    const hipgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
    const hipgraph_type_erased_device_array_view_t* initial_guess_vertices,
    const hipgraph_type_erased_device_array_view_t* initial_guess_values,
    double                                          alpha,
    double                                          epsilon,
    size_t                                          max_iterations,
    hipgraph_bool_t                                 do_expensive_check,
    hipgraph_centrality_result_t**                  result,
    hipgraph_error_t**                              error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_pagerank_allow_nonconvergence(
        (const rocgraph_handle_t*)handle,
        (rocgraph_graph_t*)graph,
        (const rocgraph_type_erased_device_array_view_t*)precomputed_vertex_out_weight_vertices,
        (const rocgraph_type_erased_device_array_view_t*)precomputed_vertex_out_weight_sums,
        (const rocgraph_type_erased_device_array_view_t*)initial_guess_vertices,
        (const rocgraph_type_erased_device_array_view_t*)initial_guess_values,
        alpha,
        epsilon,
        max_iterations,
        rg_do_expensive_check,
        (rocgraph_centrality_result_t**)result,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t hipgraph_personalized_pagerank(
    const hipgraph_resource_handle_t*               handle,
    hipgraph_graph_t*                               graph,
    const hipgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices,
    const hipgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
    const hipgraph_type_erased_device_array_view_t* initial_guess_vertices,
    const hipgraph_type_erased_device_array_view_t* initial_guess_values,
    const hipgraph_type_erased_device_array_view_t* personalization_vertices,
    const hipgraph_type_erased_device_array_view_t* personalization_values,
    double                                          alpha,
    double                                          epsilon,
    size_t                                          max_iterations,
    hipgraph_bool_t                                 do_expensive_check,
    hipgraph_centrality_result_t**                  result,
    hipgraph_error_t**                              error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_personalized_pagerank(
        (const rocgraph_handle_t*)handle,
        (rocgraph_graph_t*)graph,
        (const rocgraph_type_erased_device_array_view_t*)precomputed_vertex_out_weight_vertices,
        (const rocgraph_type_erased_device_array_view_t*)precomputed_vertex_out_weight_sums,
        (const rocgraph_type_erased_device_array_view_t*)initial_guess_vertices,
        (const rocgraph_type_erased_device_array_view_t*)initial_guess_values,
        (const rocgraph_type_erased_device_array_view_t*)personalization_vertices,
        (const rocgraph_type_erased_device_array_view_t*)personalization_values,
        alpha,
        epsilon,
        max_iterations,
        rg_do_expensive_check,
        (rocgraph_centrality_result_t**)result,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t hipgraph_personalized_pagerank_allow_nonconvergence(
    const hipgraph_resource_handle_t*               handle,
    hipgraph_graph_t*                               graph,
    const hipgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices,
    const hipgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
    const hipgraph_type_erased_device_array_view_t* initial_guess_vertices,
    const hipgraph_type_erased_device_array_view_t* initial_guess_values,
    const hipgraph_type_erased_device_array_view_t* personalization_vertices,
    const hipgraph_type_erased_device_array_view_t* personalization_values,
    double                                          alpha,
    double                                          epsilon,
    size_t                                          max_iterations,
    hipgraph_bool_t                                 do_expensive_check,
    hipgraph_centrality_result_t**                  result,
    hipgraph_error_t**                              error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_personalized_pagerank_allow_nonconvergence(
        (const rocgraph_handle_t*)handle,
        (rocgraph_graph_t*)graph,
        (const rocgraph_type_erased_device_array_view_t*)precomputed_vertex_out_weight_vertices,
        (const rocgraph_type_erased_device_array_view_t*)precomputed_vertex_out_weight_sums,
        (const rocgraph_type_erased_device_array_view_t*)initial_guess_vertices,
        (const rocgraph_type_erased_device_array_view_t*)initial_guess_values,
        (const rocgraph_type_erased_device_array_view_t*)personalization_vertices,
        (const rocgraph_type_erased_device_array_view_t*)personalization_values,
        alpha,
        epsilon,
        max_iterations,
        rg_do_expensive_check,
        (rocgraph_centrality_result_t**)result,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t hipgraph_eigenvector_centrality(const hipgraph_resource_handle_t* handle,
                                                      hipgraph_graph_t*                 graph,
                                                      double                            epsilon,
                                                      size_t          max_iterations,
                                                      hipgraph_bool_t do_expensive_check,
                                                      hipgraph_centrality_result_t** result,
                                                      hipgraph_error_t**             error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status
        = rocgraph_eigenvector_centrality((const rocgraph_handle_t*)handle,
                                          (rocgraph_graph_t*)graph,
                                          epsilon,
                                          max_iterations,
                                          rg_do_expensive_check,
                                          (rocgraph_centrality_result_t**)result,
                                          (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t
    hipgraph_katz_centrality(const hipgraph_resource_handle_t*               handle,
                             hipgraph_graph_t*                               graph,
                             const hipgraph_type_erased_device_array_view_t* betas,
                             double                                          alpha,
                             double                                          beta,
                             double                                          epsilon,
                             size_t                                          max_iterations,
                             hipgraph_bool_t                                 do_expensive_check,
                             hipgraph_centrality_result_t**                  result,
                             hipgraph_error_t**                              error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status
        = rocgraph_katz_centrality((const rocgraph_handle_t*)handle,
                                   (rocgraph_graph_t*)graph,
                                   (const rocgraph_type_erased_device_array_view_t*)betas,
                                   alpha,
                                   beta,
                                   epsilon,
                                   max_iterations,
                                   rg_do_expensive_check,
                                   (rocgraph_centrality_result_t**)result,
                                   (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t
    hipgraph_betweenness_centrality(const hipgraph_resource_handle_t*               handle,
                                    hipgraph_graph_t*                               graph,
                                    const hipgraph_type_erased_device_array_view_t* vertex_list,
                                    hipgraph_bool_t                                 normalized,
                                    hipgraph_bool_t                include_endpoints,
                                    hipgraph_bool_t                do_expensive_check,
                                    hipgraph_centrality_result_t** result,
                                    hipgraph_error_t**             error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_include_endpoints = hipgraph_bool_t2rocgraph_bool(include_endpoints);
    if(hghelper_rocgraph_bool_is_invalid(rg_include_endpoints))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_normalized = hipgraph_bool_t2rocgraph_bool(normalized);
    if(hghelper_rocgraph_bool_is_invalid(rg_normalized))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_betweenness_centrality(
        (const rocgraph_handle_t*)handle,
        (rocgraph_graph_t*)graph,
        (const rocgraph_type_erased_device_array_view_t*)vertex_list,
        rg_normalized,
        rg_include_endpoints,
        rg_do_expensive_check,
        (rocgraph_centrality_result_t**)result,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_edge_centrality_result_get_src_vertices(hipgraph_edge_centrality_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)
        rocgraph_edge_centrality_result_get_src_vertices(
            (rocgraph_edge_centrality_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_edge_centrality_result_get_dst_vertices(hipgraph_edge_centrality_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)
        rocgraph_edge_centrality_result_get_dst_vertices(
            (rocgraph_edge_centrality_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_edge_centrality_result_get_edge_ids(hipgraph_edge_centrality_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_edge_centrality_result_get_edge_ids(
        (rocgraph_edge_centrality_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_edge_centrality_result_get_values(hipgraph_edge_centrality_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_edge_centrality_result_get_values(
        (rocgraph_edge_centrality_result_t*)result);
}

void hipgraph_edge_centrality_result_free(hipgraph_edge_centrality_result_t* result)
{
    rocgraph_edge_centrality_result_free((rocgraph_edge_centrality_result_t*)result);
}

hipgraph_error_code_t hipgraph_edge_betweenness_centrality(
    const hipgraph_resource_handle_t*               handle,
    hipgraph_graph_t*                               graph,
    const hipgraph_type_erased_device_array_view_t* vertex_list,
    hipgraph_bool_t                                 normalized,
    hipgraph_bool_t                                 do_expensive_check,
    hipgraph_edge_centrality_result_t**             result,
    hipgraph_error_t**                              error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_normalized = hipgraph_bool_t2rocgraph_bool(normalized);
    if(hghelper_rocgraph_bool_is_invalid(rg_normalized))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_edge_betweenness_centrality(
        (const rocgraph_handle_t*)handle,
        (rocgraph_graph_t*)graph,
        (const rocgraph_type_erased_device_array_view_t*)vertex_list,
        rg_normalized,
        rg_do_expensive_check,
        (rocgraph_edge_centrality_result_t**)result,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_hits_result_get_vertices(hipgraph_hits_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_hits_result_get_vertices(
        (rocgraph_hits_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_hits_result_get_hubs(hipgraph_hits_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_hits_result_get_hubs(
        (rocgraph_hits_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_hits_result_get_authorities(hipgraph_hits_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_hits_result_get_authorities(
        (rocgraph_hits_result_t*)result);
}

double hipgraph_hits_result_get_hub_score_differences(hipgraph_hits_result_t* result)
{
    return (double)rocgraph_hits_result_get_hub_score_differences((rocgraph_hits_result_t*)result);
}

size_t hipgraph_hits_result_get_number_of_iterations(hipgraph_hits_result_t* result)
{
    return (size_t)rocgraph_hits_result_get_number_of_iterations((rocgraph_hits_result_t*)result);
}

void hipgraph_hits_result_free(hipgraph_hits_result_t* result)
{
    rocgraph_hits_result_free((rocgraph_hits_result_t*)result);
}

hipgraph_error_code_t
    hipgraph_hits(const hipgraph_resource_handle_t*               handle,
                  hipgraph_graph_t*                               graph,
                  double                                          epsilon,
                  size_t                                          max_iterations,
                  const hipgraph_type_erased_device_array_view_t* initial_hubs_guess_vertices,
                  const hipgraph_type_erased_device_array_view_t* initial_hubs_guess_values,
                  hipgraph_bool_t                                 normalize,
                  hipgraph_bool_t                                 do_expensive_check,
                  hipgraph_hits_result_t**                        result,
                  hipgraph_error_t**                              error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_normalize = hipgraph_bool_t2rocgraph_bool(normalize);
    if(hghelper_rocgraph_bool_is_invalid(rg_normalize))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_hits(
        (const rocgraph_handle_t*)handle,
        (rocgraph_graph_t*)graph,
        epsilon,
        max_iterations,
        (const rocgraph_type_erased_device_array_view_t*)initial_hubs_guess_vertices,
        (const rocgraph_type_erased_device_array_view_t*)initial_hubs_guess_values,
        rg_normalize,
        rg_do_expensive_check,
        (rocgraph_hits_result_t**)result,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}
