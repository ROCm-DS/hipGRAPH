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

#include <cugraph_c/community_algorithms.h>
#include "hipgraph/hipgraph_c/community_algorithms.h"

hipgraph_error_code_t hipgraph_triangle_count(const hipgraph_resource_handle_t* handle,
                                              hipgraph_graph_t*                 graph,
                                              const hipgraph_type_erased_device_array_view_t* start,
                                              hipgraph_bool_t                    do_expensive_check,
                                              hipgraph_triangle_count_result_t** result,
                                              hipgraph_error_t**                 error)
{
    cugraph_error_code_t err;
    err = cugraph_triangle_count((const cugraph_resource_handle_t*)handle,
                                 (cugraph_graph_t*)graph,
                                 (const cugraph_type_erased_device_array_view_t*)start,
                                 (bool_t)do_expensive_check,
                                 (cugraph_triangle_count_result_t**)result,
                                 (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_triangle_count_result_get_vertices(hipgraph_triangle_count_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_triangle_count_result_get_vertices(
        (cugraph_triangle_count_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_triangle_count_result_get_counts(hipgraph_triangle_count_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_triangle_count_result_get_counts(
        (cugraph_triangle_count_result_t*)result);
}

void hipgraph_triangle_count_result_free(hipgraph_triangle_count_result_t* result)
{
    cugraph_triangle_count_result_free((cugraph_triangle_count_result_t*)result);
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
    cugraph_error_code_t err;
    err = cugraph_louvain((const cugraph_resource_handle_t*)handle,
                          (cugraph_graph_t*)graph,
                          max_level,
                          threshold,
                          resolution,
                          (bool_t)do_expensive_check,
                          (cugraph_hierarchical_clustering_result_t**)result,
                          (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
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
    cugraph_error_code_t err;
    err = cugraph_leiden((const cugraph_resource_handle_t*)handle,
                         (cugraph_rng_state_t*)rng_state,
                         (cugraph_graph_t*)graph,
                         max_level,
                         resolution,
                         theta,
                         (bool_t)do_expensive_check,
                         (cugraph_hierarchical_clustering_result_t**)result,
                         (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
}

// FIXME: These don't appear to provide access to the levels of
// the dendogram. So I (Jason Riedy) don't entirely see why
// they exist.
//
// FIXME: And why is the result not const?

hipgraph_type_erased_device_array_view_t* hipgraph_hierarchical_clustering_result_get_vertices(
    hipgraph_hierarchical_clustering_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)
        cugraph_hierarchical_clustering_result_get_vertices(
            (cugraph_hierarchical_clustering_result_t*)result);
}

hipgraph_type_erased_device_array_view_t* hipgraph_hierarchical_clustering_result_get_clusters(
    hipgraph_hierarchical_clustering_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)
        cugraph_hierarchical_clustering_result_get_clusters(
            (cugraph_hierarchical_clustering_result_t*)result);
}

// FIXME: Again, no dendogram levels to use for statistical analysis. Nor any support of other models.
double hipgraph_hierarchical_clustering_result_get_modularity(
    hipgraph_hierarchical_clustering_result_t* result)
{
    return cugraph_hierarchical_clustering_result_get_modularity(
        (cugraph_hierarchical_clustering_result_t*)result);
}

void hipgraph_hierarchical_clustering_result_free(hipgraph_hierarchical_clustering_result_t* result)
{
    cugraph_hierarchical_clustering_result_free((cugraph_hierarchical_clustering_result_t*)result);
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
    cugraph_error_code_t err;
    err = cugraph_ecg((const cugraph_resource_handle_t*)handle,
                      (cugraph_rng_state_t*)rng_state,
                      (cugraph_graph_t*)graph,
                      min_weight,
                      ensemble_size,
                      max_level,
                      threshold,
                      resolution,
                      (bool_t)do_expensive_check,
                      (cugraph_hierarchical_clustering_result_t**)result,
                      (cugraph_error_t**)error);
    return err;
}

// An ego graph is literally just the subgraph from "radius"
// hops of the source. Nothing to do with ego.
hipgraph_error_code_t
    hipgraph_extract_ego(const hipgraph_resource_handle_t*               handle,
                         hipgraph_graph_t*                               graph,
                         const hipgraph_type_erased_device_array_view_t* source_vertices,
                         size_t                                          radius,
                         hipgraph_bool_t                                 do_expensive_check,
                         hipgraph_induced_subgraph_result_t**            result,
                         hipgraph_error_t**                              error)
{
    cugraph_error_code_t err;
    err = cugraph_extract_ego((const cugraph_resource_handle_t*)handle,
                              (cugraph_graph_t*)graph,
                              (const cugraph_type_erased_device_array_view_t*)source_vertices,
                              radius,
                              (bool_t)do_expensive_check,
                              (cugraph_induced_subgraph_result_t**)result,
                              (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
}

hipgraph_error_code_t hipgraph_k_truss_subgraph(const hipgraph_resource_handle_t* handle,
                                                hipgraph_graph_t*                 graph,
                                                size_t                            k,
                                                hipgraph_bool_t do_expensive_check,
                                                hipgraph_induced_subgraph_result_t** result,
                                                hipgraph_error_t**                   error)
{
    cugraph_error_code_t err;
    err = cugraph_k_truss_subgraph((const cugraph_resource_handle_t*)handle,
                                   (cugraph_graph_t*)graph,
                                   k,
                                   (bool_t)do_expensive_check,
                                   (cugraph_induced_subgraph_result_t**)result,
                                   (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
}

// FIXME: This and the spectral methods will rely on either
// cuSparse or cuSolver, plus the measures defined in raft.
hipgraph_error_code_t hipgraph_balanced_cut_clustering(const hipgraph_resource_handle_t* handle,
                                                       hipgraph_graph_t*                 graph,
                                                       size_t                            n_clusters,
                                                       size_t          n_eigenvectors,
                                                       double          evs_tolerance,
                                                       int             evs_max_iterations,
                                                       double          k_means_tolerance,
                                                       int             k_means_max_iterations,
                                                       hipgraph_bool_t do_expensive_check,
                                                       hipgraph_clustering_result_t** result,
                                                       hipgraph_error_t**             error)
{
    cugraph_error_code_t err;
    err = cugraph_balanced_cut_clustering((const cugraph_resource_handle_t*)handle,
                                          (cugraph_graph_t*)graph,
                                          n_clusters,
                                          n_eigenvectors,
                                          evs_tolerance,
                                          evs_max_iterations,
                                          k_means_tolerance,
                                          k_means_max_iterations,
                                          (bool_t)do_expensive_check,
                                          (cugraph_clustering_result_t**)result,
                                          (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
}

hipgraph_error_code_t
    hipgraph_spectral_modularity_maximization(const hipgraph_resource_handle_t* handle,
                                              hipgraph_graph_t*                 graph,
                                              size_t                            n_clusters,
                                              size_t                            n_eigenvectors,
                                              double                            evs_tolerance,
                                              int                               evs_max_iterations,
                                              double                            k_means_tolerance,
                                              int                            k_means_max_iterations,
                                              hipgraph_bool_t                do_expensive_check,
                                              hipgraph_clustering_result_t** result,
                                              hipgraph_error_t**             error)
{
    cugraph_error_code_t err;
    err = cugraph_spectral_modularity_maximization((const cugraph_resource_handle_t*)handle,
                                                   (cugraph_graph_t*)graph,
                                                   n_clusters,
                                                   n_eigenvectors,
                                                   evs_tolerance,
                                                   evs_max_iterations,
                                                   k_means_tolerance,
                                                   k_means_max_iterations,
                                                   (bool_t)do_expensive_check,
                                                   (cugraph_clustering_result_t**)result,
                                                   (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
}

// FIXME: Leaving the following piece intact because it makes no sense.
/**
 * @brief   Compute modularity of the specified clustering
 *
 * NOTE: This currently wraps the legacy spectral modularity implementation and is only
 * available in Single GPU implementation.
 */
hipgraph_error_code_t
    hipgraph_analyze_clustering_modularity(const hipgraph_resource_handle_t* handle,
                                           hipgraph_graph_t*                 graph,
                                           size_t                            n_clusters,
                                           const hipgraph_type_erased_device_array_view_t* vertices,
                                           const hipgraph_type_erased_device_array_view_t* clusters,
                                           double*                                         score,
                                           hipgraph_error_t**                              error)
{
    cugraph_error_code_t err;
    err = cugraph_analyze_clustering_modularity(
        (const cugraph_resource_handle_t*)handle,
        (cugraph_graph_t*)graph,
        n_clusters,
        (const cugraph_type_erased_device_array_view_t*)vertices,
        (const cugraph_type_erased_device_array_view_t*)clusters,
        score,
        (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
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
    cugraph_error_code_t err;
    err = cugraph_analyze_clustering_edge_cut(
        (const cugraph_resource_handle_t*)handle,
        (cugraph_graph_t*)graph,
        n_clusters,
        (const cugraph_type_erased_device_array_view_t*)vertices,
        (const cugraph_type_erased_device_array_view_t*)clusters,
        score,
        (cugraph_error_t**)error);
    return err;
}

// Term used before being defined. Is this a conductance instance?
hipgraph_error_code_t
    hipgraph_analyze_clustering_ratio_cut(const hipgraph_resource_handle_t* handle,
                                          hipgraph_graph_t*                 graph,
                                          size_t                            n_clusters,
                                          const hipgraph_type_erased_device_array_view_t* vertices,
                                          const hipgraph_type_erased_device_array_view_t* clusters,
                                          double*                                         score,
                                          hipgraph_error_t**                              error)
{
    cugraph_error_code_t err;
    err = cugraph_analyze_clustering_ratio_cut(
        (const cugraph_resource_handle_t*)handle,
        (cugraph_graph_t*)graph,
        n_clusters,
        (const cugraph_type_erased_device_array_view_t*)vertices,
        (const cugraph_type_erased_device_array_view_t*)clusters,
        score,
        (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_clustering_result_get_vertices(hipgraph_clustering_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_clustering_result_get_vertices(
        (cugraph_clustering_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_clustering_result_get_clusters(hipgraph_clustering_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_clustering_result_get_clusters(
        (cugraph_clustering_result_t*)result);
}

void hipgraph_clustering_result_free(hipgraph_clustering_result_t* result)
{
    cugraph_clustering_result_free((cugraph_clustering_result_t*)result);
}
