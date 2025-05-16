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
#include "hipgraph/hipgraph_c/community_algorithms.h"

hipgraph_error_code_t hipgraph_triangle_count(const hipgraph_resource_handle_t* handle,
                                              hipgraph_graph_t*                 graph,
                                              const hipgraph_type_erased_device_array_view_t* start,
                                              hipgraph_bool_t                    do_expensive_check,
                                              hipgraph_triangle_count_result_t** result,
                                              hipgraph_error_t**                 error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status
        = rocgraph_triangle_count((const rocgraph_handle_t*)handle,
                                  (rocgraph_graph_t*)graph,
                                  (const rocgraph_type_erased_device_array_view_t*)start,
                                  rg_do_expensive_check,
                                  (rocgraph_triangle_count_result_t**)result,
                                  (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_triangle_count_result_get_vertices(hipgraph_triangle_count_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_triangle_count_result_get_vertices(
        (rocgraph_triangle_count_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_triangle_count_result_get_counts(hipgraph_triangle_count_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_triangle_count_result_get_counts(
        (rocgraph_triangle_count_result_t*)result);
}

void hipgraph_triangle_count_result_free(hipgraph_triangle_count_result_t* result)
{
    rocgraph_triangle_count_result_free((rocgraph_triangle_count_result_t*)result);
}

hipgraph_error_code_t hipgraph_louvain(const hipgraph_resource_handle_t* handle,
                                       hipgraph_graph_t*                 graph,
                                       size_t                            max_level,
                                       double                            threshold,
                                       double                            resolution,
                                       hipgraph_bool_t                   do_expensive_check,
                                       hipgraph_hierarchical_clustering_result_t** result,
                                       hipgraph_error_t**                          error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status
        = rocgraph_louvain((const rocgraph_handle_t*)handle,
                           (rocgraph_graph_t*)graph,
                           max_level,
                           threshold,
                           resolution,
                           rg_do_expensive_check,
                           (rocgraph_hierarchical_clustering_result_t**)result,
                           (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t hipgraph_leiden(const hipgraph_resource_handle_t* handle,
                                      hipgraph_rng_state_t*             rng_state,
                                      hipgraph_graph_t*                 graph,
                                      size_t                            max_level,
                                      double                            resolution,
                                      double                            theta,
                                      hipgraph_bool_t                   do_expensive_check,
                                      hipgraph_hierarchical_clustering_result_t** result,
                                      hipgraph_error_t**                          error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_leiden((const rocgraph_handle_t*)handle,
                                                (rocgraph_rng_state_t*)rng_state,
                                                (rocgraph_graph_t*)graph,
                                                max_level,
                                                resolution,
                                                theta,
                                                rg_do_expensive_check,
                                                (rocgraph_hierarchical_clustering_result_t**)result,
                                                (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_type_erased_device_array_view_t* hipgraph_hierarchical_clustering_result_get_vertices(
    hipgraph_hierarchical_clustering_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)
        rocgraph_hierarchical_clustering_result_get_vertices(
            (rocgraph_hierarchical_clustering_result_t*)result);
}

hipgraph_type_erased_device_array_view_t* hipgraph_hierarchical_clustering_result_get_clusters(
    hipgraph_hierarchical_clustering_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)
        rocgraph_hierarchical_clustering_result_get_clusters(
            (rocgraph_hierarchical_clustering_result_t*)result);
}

double hipgraph_hierarchical_clustering_result_get_modularity(
    hipgraph_hierarchical_clustering_result_t* result)
{
    return (double)rocgraph_hierarchical_clustering_result_get_modularity(
        (rocgraph_hierarchical_clustering_result_t*)result);
}

void hipgraph_hierarchical_clustering_result_free(hipgraph_hierarchical_clustering_result_t* result)
{
    rocgraph_hierarchical_clustering_result_free(
        (rocgraph_hierarchical_clustering_result_t*)result);
}

hipgraph_error_code_t hipgraph_ecg(const hipgraph_resource_handle_t*           handle,
                                   hipgraph_rng_state_t*                       rng_state,
                                   hipgraph_graph_t*                           graph,
                                   double                                      min_weight,
                                   size_t                                      ensemble_size,
                                   size_t                                      max_level,
                                   double                                      threshold,
                                   double                                      resolution,
                                   hipgraph_bool_t                             do_expensive_check,
                                   hipgraph_hierarchical_clustering_result_t** result,
                                   hipgraph_error_t**                          error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_ecg((const rocgraph_handle_t*)handle,
                                             (rocgraph_rng_state_t*)rng_state,
                                             (rocgraph_graph_t*)graph,
                                             min_weight,
                                             ensemble_size,
                                             max_level,
                                             threshold,
                                             resolution,
                                             rg_do_expensive_check,
                                             (rocgraph_hierarchical_clustering_result_t**)result,
                                             (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t
    hipgraph_extract_ego(const hipgraph_resource_handle_t*               handle,
                         hipgraph_graph_t*                               graph,
                         const hipgraph_type_erased_device_array_view_t* source_vertices,
                         size_t                                          radius,
                         hipgraph_bool_t                                 do_expensive_check,
                         hipgraph_induced_subgraph_result_t**            result,
                         hipgraph_error_t**                              error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status
        = rocgraph_extract_ego((const rocgraph_handle_t*)handle,
                               (rocgraph_graph_t*)graph,
                               (const rocgraph_type_erased_device_array_view_t*)source_vertices,
                               radius,
                               rg_do_expensive_check,
                               (rocgraph_induced_subgraph_result_t**)result,
                               (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t hipgraph_k_truss_subgraph(const hipgraph_resource_handle_t* handle,
                                                hipgraph_graph_t*                 graph,
                                                size_t                            k,
                                                hipgraph_bool_t do_expensive_check,
                                                hipgraph_induced_subgraph_result_t** result,
                                                hipgraph_error_t**                   error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status
        = rocgraph_k_truss_subgraph((const rocgraph_handle_t*)handle,
                                    (rocgraph_graph_t*)graph,
                                    k,
                                    rg_do_expensive_check,
                                    (rocgraph_induced_subgraph_result_t**)result,
                                    (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t hipgraph_balanced_cut_clustering(const hipgraph_resource_handle_t* handle,
                                                       hipgraph_graph_t*                 graph,
                                                       size_t                            n_clusters,
                                                       size_t          n_eigenvectors,
                                                       double          evs_tolerance,
                                                       int32_t         evs_max_iterations,
                                                       double          k_means_tolerance,
                                                       int32_t         k_means_max_iterations,
                                                       hipgraph_bool_t do_expensive_check,
                                                       hipgraph_clustering_result_t** result,
                                                       hipgraph_error_t**             error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status
        = rocgraph_balanced_cut_clustering((const rocgraph_handle_t*)handle,
                                           (rocgraph_graph_t*)graph,
                                           n_clusters,
                                           n_eigenvectors,
                                           evs_tolerance,
                                           evs_max_iterations,
                                           k_means_tolerance,
                                           k_means_max_iterations,
                                           rg_do_expensive_check,
                                           (rocgraph_clustering_result_t**)result,
                                           (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t
    hipgraph_spectral_modularity_maximization(const hipgraph_resource_handle_t* handle,
                                              hipgraph_graph_t*                 graph,
                                              size_t                            n_clusters,
                                              size_t                            n_eigenvectors,
                                              double                            evs_tolerance,
                                              int32_t                           evs_max_iterations,
                                              double                            k_means_tolerance,
                                              int32_t                        k_means_max_iterations,
                                              hipgraph_bool_t                do_expensive_check,
                                              hipgraph_clustering_result_t** result,
                                              hipgraph_error_t**             error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status
        = rocgraph_spectral_modularity_maximization((const rocgraph_handle_t*)handle,
                                                    (rocgraph_graph_t*)graph,
                                                    n_clusters,
                                                    n_eigenvectors,
                                                    evs_tolerance,
                                                    evs_max_iterations,
                                                    k_means_tolerance,
                                                    k_means_max_iterations,
                                                    rg_do_expensive_check,
                                                    (rocgraph_clustering_result_t**)result,
                                                    (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t
    hipgraph_analyze_clustering_modularity(const hipgraph_resource_handle_t* handle,
                                           hipgraph_graph_t*                 graph,
                                           size_t                            n_clusters,
                                           const hipgraph_type_erased_device_array_view_t* vertices,
                                           const hipgraph_type_erased_device_array_view_t* clusters,
                                           double*                                         score,
                                           hipgraph_error_t**                              error)
{
    rocgraph_status rg_status = rocgraph_analyze_clustering_modularity(
        (const rocgraph_handle_t*)handle,
        (rocgraph_graph_t*)graph,
        n_clusters,
        (const rocgraph_type_erased_device_array_view_t*)vertices,
        (const rocgraph_type_erased_device_array_view_t*)clusters,
        score,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t
    hipgraph_analyze_clustering_edge_cut(const hipgraph_resource_handle_t*               handle,
                                         hipgraph_graph_t*                               graph,
                                         size_t                                          n_clusters,
                                         const hipgraph_type_erased_device_array_view_t* vertices,
                                         const hipgraph_type_erased_device_array_view_t* clusters,
                                         double*                                         score,
                                         hipgraph_error_t**                              error)
{
    rocgraph_status rg_status = rocgraph_analyze_clustering_edge_cut(
        (const rocgraph_handle_t*)handle,
        (rocgraph_graph_t*)graph,
        n_clusters,
        (const rocgraph_type_erased_device_array_view_t*)vertices,
        (const rocgraph_type_erased_device_array_view_t*)clusters,
        score,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t
    hipgraph_analyze_clustering_ratio_cut(const hipgraph_resource_handle_t* handle,
                                          hipgraph_graph_t*                 graph,
                                          size_t                            n_clusters,
                                          const hipgraph_type_erased_device_array_view_t* vertices,
                                          const hipgraph_type_erased_device_array_view_t* clusters,
                                          double*                                         score,
                                          hipgraph_error_t**                              error)
{
    rocgraph_status rg_status = rocgraph_analyze_clustering_ratio_cut(
        (const rocgraph_handle_t*)handle,
        (rocgraph_graph_t*)graph,
        n_clusters,
        (const rocgraph_type_erased_device_array_view_t*)vertices,
        (const rocgraph_type_erased_device_array_view_t*)clusters,
        score,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_clustering_result_get_vertices(hipgraph_clustering_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_clustering_result_get_vertices(
        (rocgraph_clustering_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_clustering_result_get_clusters(hipgraph_clustering_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_clustering_result_get_clusters(
        (rocgraph_clustering_result_t*)result);
}

void hipgraph_clustering_result_free(hipgraph_clustering_result_t* result)
{
    rocgraph_clustering_result_free((rocgraph_clustering_result_t*)result);
}
