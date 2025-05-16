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

#include <cugraph_c/centrality_algorithms.h>
#include "hipgraph/hipgraph_c/centrality_algorithms.h"

/*
 hipgraph_centrality_result_get_vertices
*/
hipgraph_type_erased_device_array_view_t*
    hipgraph_centrality_result_get_vertices(hipgraph_centrality_result_t* result)
{

    cugraph_type_erased_device_array_view_t* d_arrview;

    d_arrview = cugraph_centrality_result_get_vertices((cugraph_centrality_result_t*)result);

    return (hipgraph_type_erased_device_array_view_t*)d_arrview;
}

/*
 hipgraph_centrality_result_get_values
*/
hipgraph_type_erased_device_array_view_t*
    hipgraph_centrality_result_get_values(hipgraph_centrality_result_t* result)
{

    cugraph_type_erased_device_array_view_t* d_arrview;

    d_arrview = cugraph_centrality_result_get_values((cugraph_centrality_result_t*)result);

    return (hipgraph_type_erased_device_array_view_t*)d_arrview;
}

/*
 hipgraph_centrality_result_get_num_iterations
*/
size_t hipgraph_centrality_result_get_num_iterations(hipgraph_centrality_result_t* result)
{
    return cugraph_centrality_result_get_num_iterations((cugraph_centrality_result_t*)result);
}

/*
 hipgraph_centrality_result_converged
*/
hipgraph_bool_t hipgraph_centrality_result_converged(hipgraph_centrality_result_t* result)
{
    return (hipgraph_bool_t)cugraph_centrality_result_converged(
        (cugraph_centrality_result_t*)result);
}

/*
 hipgraph_centrality_result_free
*/
void hipgraph_centrality_result_free(hipgraph_centrality_result_t* result)
{
    return cugraph_centrality_result_free((cugraph_centrality_result_t*)result);
}

/*
 hipgraph_pagerank
*/
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

    cugraph_error_code_t out;

    out = cugraph_pagerank(
        (const cugraph_resource_handle_t*)handle,
        (cugraph_graph_t*)graph,
        (const cugraph_type_erased_device_array_view_t*)precomputed_vertex_out_weight_vertices,
        (const cugraph_type_erased_device_array_view_t*)precomputed_vertex_out_weight_sums,
        (const cugraph_type_erased_device_array_view_t*)initial_guess_vertices,
        (const cugraph_type_erased_device_array_view_t*)initial_guess_values,
        alpha,
        epsilon,
        max_iterations,
        (bool_t)do_expensive_check,
        (cugraph_centrality_result_t**)result,
        (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_pagerank_allow_nonconvergence
*/
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

    cugraph_error_code_t out;

    out = cugraph_pagerank_allow_nonconvergence(
        (const cugraph_resource_handle_t*)handle,
        (cugraph_graph_t*)graph,
        (const cugraph_type_erased_device_array_view_t*)precomputed_vertex_out_weight_vertices,
        (const cugraph_type_erased_device_array_view_t*)precomputed_vertex_out_weight_sums,
        (const cugraph_type_erased_device_array_view_t*)initial_guess_vertices,
        (const cugraph_type_erased_device_array_view_t*)initial_guess_values,
        alpha,
        epsilon,
        max_iterations,
        (bool_t)do_expensive_check,
        (cugraph_centrality_result_t**)result,
        (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_personalized_pagerank
*/
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

    cugraph_error_code_t out;

    out = cugraph_personalized_pagerank(
        (const cugraph_resource_handle_t*)handle,
        (cugraph_graph_t*)graph,
        (const cugraph_type_erased_device_array_view_t*)precomputed_vertex_out_weight_vertices,
        (const cugraph_type_erased_device_array_view_t*)precomputed_vertex_out_weight_sums,
        (const cugraph_type_erased_device_array_view_t*)initial_guess_vertices,
        (const cugraph_type_erased_device_array_view_t*)initial_guess_values,
        (const cugraph_type_erased_device_array_view_t*)personalization_vertices,
        (const cugraph_type_erased_device_array_view_t*)personalization_values,
        alpha,
        epsilon,
        max_iterations,
        (bool_t)do_expensive_check,
        (cugraph_centrality_result_t**)result,
        (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_personalized_pagerank_allow_nonconvergence
*/
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

    cugraph_error_code_t out;

    out = cugraph_personalized_pagerank_allow_nonconvergence(
        (const cugraph_resource_handle_t*)handle,
        (cugraph_graph_t*)graph,
        (const cugraph_type_erased_device_array_view_t*)precomputed_vertex_out_weight_vertices,
        (const cugraph_type_erased_device_array_view_t*)precomputed_vertex_out_weight_sums,
        (const cugraph_type_erased_device_array_view_t*)initial_guess_vertices,
        (const cugraph_type_erased_device_array_view_t*)initial_guess_values,
        (const cugraph_type_erased_device_array_view_t*)personalization_vertices,
        (const cugraph_type_erased_device_array_view_t*)personalization_values,
        alpha,
        epsilon,
        max_iterations,
        (bool_t)do_expensive_check,
        (cugraph_centrality_result_t**)result,
        (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_eigenvector_centrality
*/
hipgraph_error_code_t hipgraph_eigenvector_centrality(const hipgraph_resource_handle_t* handle,
                                                      hipgraph_graph_t*                 graph,
                                                      double                            epsilon,
                                                      size_t          max_iterations,
                                                      hipgraph_bool_t do_expensive_check,
                                                      hipgraph_centrality_result_t** result,
                                                      hipgraph_error_t**             error)
{

    cugraph_error_code_t out;

    out = cugraph_eigenvector_centrality((const cugraph_resource_handle_t*)handle,
                                         (cugraph_graph_t*)graph,
                                         epsilon,
                                         max_iterations,
                                         (bool_t)do_expensive_check,
                                         (cugraph_centrality_result_t**)result,
                                         (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_katz_centrality
*/
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

    cugraph_error_code_t out;

    out = cugraph_katz_centrality((const cugraph_resource_handle_t*)handle,
                                  (cugraph_graph_t*)graph,
                                  (const cugraph_type_erased_device_array_view_t*)betas,
                                  alpha,
                                  beta,
                                  epsilon,
                                  max_iterations,
                                  (bool_t)do_expensive_check,
                                  (cugraph_centrality_result_t**)result,
                                  (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_betweenness_centrality
*/
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

    cugraph_error_code_t out;

    out = cugraph_betweenness_centrality(
        (const cugraph_resource_handle_t*)handle,
        (cugraph_graph_t*)graph,
        (const cugraph_type_erased_device_array_view_t*)vertex_list,
        (bool_t)normalized,
        (bool_t)include_endpoints,
        (bool_t)do_expensive_check,
        (cugraph_centrality_result_t**)result,
        (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_edge_centrality_result_get_src_vertices
*/
hipgraph_type_erased_device_array_view_t*
    hipgraph_edge_centrality_result_get_src_vertices(hipgraph_edge_centrality_result_t* result)
{

    cugraph_type_erased_device_array_view_t* d_arrview;

    d_arrview = cugraph_edge_centrality_result_get_src_vertices(
        (cugraph_edge_centrality_result_t*)result);

    return (hipgraph_type_erased_device_array_view_t*)d_arrview;
}

/*
 hipgraph_edge_centrality_result_get_dst_vertices
*/
hipgraph_type_erased_device_array_view_t*
    hipgraph_edge_centrality_result_get_dst_vertices(hipgraph_edge_centrality_result_t* result)
{

    cugraph_type_erased_device_array_view_t* d_arrview;

    d_arrview = cugraph_edge_centrality_result_get_dst_vertices(
        (cugraph_edge_centrality_result_t*)result);

    return (hipgraph_type_erased_device_array_view_t*)d_arrview;
}

/*
 hipgraph_edge_centrality_result_get_edge_ids
*/
hipgraph_type_erased_device_array_view_t*
    hipgraph_edge_centrality_result_get_edge_ids(hipgraph_edge_centrality_result_t* result)
{

    cugraph_type_erased_device_array_view_t* d_arrview;

    d_arrview
        = cugraph_edge_centrality_result_get_edge_ids((cugraph_edge_centrality_result_t*)result);

    return (hipgraph_type_erased_device_array_view_t*)d_arrview;
}

/*
 hipgraph_edge_centrality_result_get_values
*/
hipgraph_type_erased_device_array_view_t*
    hipgraph_edge_centrality_result_get_values(hipgraph_edge_centrality_result_t* result)
{

    cugraph_type_erased_device_array_view_t* d_arrview;

    d_arrview
        = cugraph_edge_centrality_result_get_values((cugraph_edge_centrality_result_t*)result);

    return (hipgraph_type_erased_device_array_view_t*)d_arrview;
}

/*
 hipgraph_edge_centrality_result_free
*/
void hipgraph_edge_centrality_result_free(hipgraph_edge_centrality_result_t* result)
{
    return cugraph_edge_centrality_result_free((cugraph_edge_centrality_result_t*)result);
}

/*
 hipgraph_edge_betweenness_centrality
*/
hipgraph_error_code_t hipgraph_edge_betweenness_centrality(
    const hipgraph_resource_handle_t*               handle,
    hipgraph_graph_t*                               graph,
    const hipgraph_type_erased_device_array_view_t* vertex_list,
    hipgraph_bool_t                                 normalized,
    hipgraph_bool_t                                 do_expensive_check,
    hipgraph_edge_centrality_result_t**             result,
    hipgraph_error_t**                              error)
{

    cugraph_error_code_t out;

    out = cugraph_edge_betweenness_centrality(
        (const cugraph_resource_handle_t*)handle,
        (cugraph_graph_t*)graph,
        (const cugraph_type_erased_device_array_view_t*)vertex_list,
        (bool_t)normalized,
        (bool_t)do_expensive_check,
        (cugraph_edge_centrality_result_t**)result,
        (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_hits_result_get_vertices
*/
hipgraph_type_erased_device_array_view_t*
    hipgraph_hits_result_get_vertices(hipgraph_hits_result_t* result)
{

    cugraph_type_erased_device_array_view_t* d_arrview;

    d_arrview = cugraph_hits_result_get_vertices((cugraph_hits_result_t*)result);

    return (hipgraph_type_erased_device_array_view_t*)d_arrview;
}

/*
 hipgraph_hits_result_get_hubs
*/
hipgraph_type_erased_device_array_view_t*
    hipgraph_hits_result_get_hubs(hipgraph_hits_result_t* result)
{

    cugraph_type_erased_device_array_view_t* d_arrview;

    d_arrview = cugraph_hits_result_get_hubs((cugraph_hits_result_t*)result);

    return (hipgraph_type_erased_device_array_view_t*)d_arrview;
}

/*
 hipgraph_hits_result_get_authorities
*/
hipgraph_type_erased_device_array_view_t*
    hipgraph_hits_result_get_authorities(hipgraph_hits_result_t* result)
{

    cugraph_type_erased_device_array_view_t* d_arrview;

    d_arrview = cugraph_hits_result_get_authorities((cugraph_hits_result_t*)result);

    return (hipgraph_type_erased_device_array_view_t*)d_arrview;
}

/*
 hipgraph_hits_result_get_hub_score_differences
*/
double hipgraph_hits_result_get_hub_score_differences(hipgraph_hits_result_t* result)
{
    return cugraph_hits_result_get_hub_score_differences((cugraph_hits_result_t*)result);
}

/*
 hipgraph_hits_result_get_number_of_iterations
*/
size_t hipgraph_hits_result_get_number_of_iterations(hipgraph_hits_result_t* result)
{
    return cugraph_hits_result_get_number_of_iterations((cugraph_hits_result_t*)result);
}

/*
 hipgraph_hits_result_free
*/
void hipgraph_hits_result_free(hipgraph_hits_result_t* result)
{
    cugraph_hits_result_free((cugraph_hits_result_t*)result);
}

/*
 hipgraph_hits
*/
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

    cugraph_error_code_t out;

    out = cugraph_hits((const cugraph_resource_handle_t*)handle,
                       (cugraph_graph_t*)graph,
                       epsilon,
                       max_iterations,
                       (const cugraph_type_erased_device_array_view_t*)initial_hubs_guess_vertices,
                       (const cugraph_type_erased_device_array_view_t*)initial_hubs_guess_values,
                       (bool_t)normalize,
                       (bool_t)do_expensive_check,
                       (cugraph_hits_result_t**)result,
                       (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}
