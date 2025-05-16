// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*! \file */
/* ************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
 *
 * Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
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
 * ************************************************************************ */
#pragma once

#include "hipgraph/hipgraph_c/array.h"
#include "hipgraph/hipgraph_c/error.h"
#include "hipgraph/hipgraph_c/graph.h"
#include "hipgraph/hipgraph_c/random.h"
#include "hipgraph/hipgraph_c/resource_handle.h"

/** @defgroup centrality Centrality algorithms
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief     Opaque centrality result type
 */
typedef struct
{
    /** @brief align_ result type */
    int32_t align_;
} hipgraph_centrality_result_t;

/**
 * @ingroup centrality
 * @brief   Get the vertex ids from the centrality result
 *
 * @param [in]   result   The result from a centrality algorithm
 * @return type erased array of vertex ids
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_centrality_result_get_vertices(hipgraph_centrality_result_t* result);

/**
 * @ingroup centrality
 * @brief   Get the centrality values from a centrality algorithm result
 *
 * @param [in]   result   The result from a centrality algorithm
 * @return type erased array view of centrality values
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_centrality_result_get_values(hipgraph_centrality_result_t* result);

/**
 * @ingroup centrality
 * @brief     Get the number of iterations executed from the algorithm metadata
 *
 * @param [in]   result   The result from a centrality algorithm
 * @return the number of iterations
 */
HIPGRAPH_EXPORT size_t
    hipgraph_centrality_result_get_num_iterations(hipgraph_centrality_result_t* result);

/**
 * @ingroup centrality
 * @brief     Returns true if the centrality algorithm converged
 *
 * @param [in]   result   The result from a centrality algorithm
 * @return True if the centrality algorithm converged, false otherwise
 */
HIPGRAPH_EXPORT hipgraph_bool_t
    hipgraph_centrality_result_converged(hipgraph_centrality_result_t* result);

/**
 * @ingroup centrality
 * @brief     Free centrality result
 *
 * @param [in]   result   The result from a centrality algorithm
 */
HIPGRAPH_EXPORT void hipgraph_centrality_result_free(hipgraph_centrality_result_t* result);

/**
 * @brief     Compute pagerank
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  graph       Pointer to graph
 * @param [in]  precomputed_vertex_out_weight_vertices
 *                          Optionally send in precomputed sum of vertex out weights
 *                          (a performance optimization).  This defines the vertices.
 *                          Set to NULL if no value is passed.
 * @param [in]  precomputed_vertex_out_weight_sums
 *                          Optionally send in precomputed sum of vertex out weights
 *                          (a performance optimization).  Set to NULL if
 *                          no value is passed.
 * @param [in]  initial_guess_vertices
 *                          Optionally send in an initial guess of the pagerank values
 *                          (a performance optimization).  This defines the vertices.
 *                          Set to NULL if no value is passed. If NULL, initial PageRank
 *                          values are set to 1.0 divided by the number of vertices in
 *                          the graph.
 * @param [in]  initial_guess_values
 *                          Optionally send in an initial guess of the pagerank values
 *                          (a performance optimization).  Set to NULL if
 *                          no value is passed. If NULL, initial PageRank values are set
 *                          to 1.0 divided by the number of vertices in the graph.
 * @param [in]  alpha       PageRank damping factor.
 * @param [in]  epsilon     Error tolerance to check convergence. Convergence is assumed
 *                          if the sum of the differences in PageRank values between two
 *                          consecutive iterations is less than the number of vertices
 *                          in the graph multiplied by @p epsilon.
 * @param [in]  max_iterations Maximum number of PageRank iterations.
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result      Opaque pointer to pagerank results
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_pagerank(
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
    hipgraph_error_t**                              error);

/**
 * @brief     Compute pagerank
 *
 * @deprecated This version of pagerank should be dropped in favor
 *             of the hipgraph_pagerank_allow_nonconvergence version.
 *             Eventually that version will be renamed to this version.
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  graph       Pointer to graph
 * @param [in]  precomputed_vertex_out_weight_vertices
 *                          Optionally send in precomputed sum of vertex out weights
 *                          (a performance optimization).  This defines the vertices.
 *                          Set to NULL if no value is passed.
 * @param [in]  precomputed_vertex_out_weight_sums
 *                          Optionally send in precomputed sum of vertex out weights
 *                          (a performance optimization).  Set to NULL if
 *                          no value is passed.
 * @param [in]  initial_guess_vertices
 *                          Optionally send in an initial guess of the pagerank values
 *                          (a performance optimization).  This defines the vertices.
 *                          Set to NULL if no value is passed. If NULL, initial PageRank
 *                          values are set to 1.0 divided by the number of vertices in
 *                          the graph.
 * @param [in]  initial_guess_values
 *                          Optionally send in an initial guess of the pagerank values
 *                          (a performance optimization).  Set to NULL if
 *                          no value is passed. If NULL, initial PageRank values are set
 *                          to 1.0 divided by the number of vertices in the graph.
 * @param [in]  alpha       PageRank damping factor.
 * @param [in]  epsilon     Error tolerance to check convergence. Convergence is assumed
 *                          if the sum of the differences in PageRank values between two
 *                          consecutive iterations is less than the number of vertices
 *                          in the graph multiplied by @p epsilon.
 * @param [in]  max_iterations Maximum number of PageRank iterations.
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result      Opaque pointer to pagerank results
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_pagerank_allow_nonconvergence(
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
    hipgraph_error_t**                              error);

/**
 * @brief     Compute personalized pagerank
 *
 * @deprecated This version of personalized pagerank should be dropped in favor
 *             of the hipgraph_personalized_pagerank_allow_nonconvergence version.
 *             Eventually that version will be renamed to this version.
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  graph       Pointer to graph
 * @param [in]  precomputed_vertex_out_weight_vertices
 *                          Optionally send in precomputed sum of vertex out weights
 *                          (a performance optimization).  This defines the vertices.
 *                          Set to NULL if no value is passed.
 * @param [in]  precomputed_vertex_out_weight_sums
 *                          Optionally send in precomputed sum of vertex out weights
 *                          (a performance optimization).  Set to NULL if
 *                          no value is passed.
 * @param [in]  initial_guess_vertices
 *                          Optionally send in an initial guess of the pagerank values
 *                          (a performance optimization).  This defines the vertices.
 *                          Set to NULL if no value is passed. If NULL, initial PageRank
 *                          values are set to 1.0 divided by the number of vertices in
 *                          the graph.
 * @param [in]  initial_guess_values
 *                          Optionally send in an initial guess of the pagerank values
 *                          (a performance optimization).  Set to NULL if
 *                          no value is passed. If NULL, initial PageRank values are set
 *                          to 1.0 divided by the number of vertices in the graph.
 * @param [in]  personalization_vertices Pointer to an array storing personalization vertex
 * identifiers (compute personalized PageRank).
 * @param [in]  personalization_values Pointer to an array storing personalization values for the
 * vertices in the personalization set.
 * @param [in]  alpha       PageRank damping factor.
 * @param [in]  epsilon     Error tolerance to check convergence. Convergence is assumed
 *                          if the sum of the differences in PageRank values between two
 *                          consecutive iterations is less than the number of vertices
 *                          in the graph multiplied by @p epsilon.
 * @param [in]  max_iterations Maximum number of PageRank iterations.
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result      Opaque pointer to pagerank results
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_personalized_pagerank(
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
    hipgraph_error_t**                              error);

/**
 * @brief     Compute personalized pagerank
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  graph       Pointer to graph
 * @param [in]  precomputed_vertex_out_weight_vertices
 *                          Optionally send in precomputed sum of vertex out weights
 *                          (a performance optimization).  This defines the vertices.
 *                          Set to NULL if no value is passed.
 * @param [in]  precomputed_vertex_out_weight_sums
 *                          Optionally send in precomputed sum of vertex out weights
 *                          (a performance optimization).  Set to NULL if
 *                          no value is passed.
 * @param [in]  initial_guess_vertices
 *                          Optionally send in an initial guess of the pagerank values
 *                          (a performance optimization).  This defines the vertices.
 *                          Set to NULL if no value is passed. If NULL, initial PageRank
 *                          values are set to 1.0 divided by the number of vertices in
 *                          the graph.
 * @param [in]  initial_guess_values
 *                          Optionally send in an initial guess of the pagerank values
 *                          (a performance optimization).  Set to NULL if
 *                          no value is passed. If NULL, initial PageRank values are set
 *                          to 1.0 divided by the number of vertices in the graph.
 * @param [in]  personalization_vertices Pointer to an array storing personalization vertex
 * identifiers (compute personalized PageRank).
 * @param [in]  personalization_values Pointer to an array storing personalization values for the
 * vertices in the personalization set.
 * @param [in]  alpha       PageRank damping factor.
 * @param [in]  epsilon     Error tolerance to check convergence. Convergence is assumed
 *                          if the sum of the differences in PageRank values between two
 *                          consecutive iterations is less than the number of vertices
 *                          in the graph multiplied by @p epsilon.
 * @param [in]  max_iterations Maximum number of PageRank iterations.
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result      Opaque pointer to pagerank results
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_personalized_pagerank_allow_nonconvergence(
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
    hipgraph_error_t**                              error);

/**
 * @brief     Compute eigenvector centrality
 *
 * Computed using the power method.
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  graph       Pointer to graph
 * @param [in]  epsilon     Error tolerance to check convergence. Convergence is measured
 *                          comparing the L1 norm until it is less than epsilon
 * @param [in]  max_iterations Maximum number of power iterations, will not exceed this number
 *                          of iterations even if we haven't converged
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result      Opaque pointer to eigenvector centrality results
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_eigenvector_centrality(const hipgraph_resource_handle_t* handle,
                                    hipgraph_graph_t*                 graph,
                                    double                            epsilon,
                                    size_t                            max_iterations,
                                    hipgraph_bool_t                   do_expensive_check,
                                    hipgraph_centrality_result_t**    result,
                                    hipgraph_error_t**                error);

/**
 * @brief     Compute katz centrality
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  graph       Pointer to graph
 * @param [in]  betas       Optionally send in a device array holding values to be added to
 *                          each vertex's new Katz Centrality score in every iteration.
 *                          If set to NULL then @p beta is used for all vertices.
 * @param [in]  alpha       Katz centrality attenuation factor.  This should be smaller
 *                          than the inverse of the maximum eigenvalue of this graph
 * @param [in]  beta        Constant value to be added to each vertex's new Katz
 *                          Centrality score in every iteration.  Relevant only when
 *                          @p betas is NULL
 * @param [in]  epsilon     Error tolerance to check convergence. Convergence is assumed
 *                          if the sum of the differences in Katz Centrality values between
 *                          two consecutive iterations is less than the number of vertices
 *                          in the graph multiplied by @p epsilon. (L1-norm)
 * @param [in]  max_iterations Maximum number of Katz Centrality iterations.
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result      Opaque pointer to katz centrality results
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_katz_centrality(const hipgraph_resource_handle_t*               handle,
                             hipgraph_graph_t*                               graph,
                             const hipgraph_type_erased_device_array_view_t* betas,
                             double                                          alpha,
                             double                                          beta,
                             double                                          epsilon,
                             size_t                                          max_iterations,
                             hipgraph_bool_t                                 do_expensive_check,
                             hipgraph_centrality_result_t**                  result,
                             hipgraph_error_t**                              error);

/**
 * @brief     Compute betweenness centrality
 *
 * Betweenness can be computed exactly by specifying vertex_list as NULL.  This will compute
 * betweenness centrality by doing a traversal from every source vertex.
 *
 * Approximate betweenness can be computed specifying a list of vertices that should be
 * used as seeds for the traversals.  Note that the function hipgraph_select_random_vertices can be
 * used to create a list of seeds.
 *
 * @param [in]  handle             Handle for accessing resources
 * @param [in]  graph              Pointer to graph
 * @param [in]  vertex_list        Optionally specify a device array containing a list of vertices
 *                                 to use as seeds for betweenness centrality approximation
 * @param [in]  normalized         Normalize
 * @param [in]  include_endpoints  The traditional formulation of betweenness centrality does not
 *                                 include endpoints when considering a vertex to be on a shortest
 *                                 path.  Setting this to true will consider the endpoints of a
 *                                 path to be part of the path.
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result             Opaque pointer to betweenness centrality results
 * @param [out] error              Pointer to an error object storing details of any error.  Will
 *                                 be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_betweenness_centrality(const hipgraph_resource_handle_t*               handle,
                                    hipgraph_graph_t*                               graph,
                                    const hipgraph_type_erased_device_array_view_t* vertex_list,
                                    hipgraph_bool_t                                 normalized,
                                    hipgraph_bool_t                include_endpoints,
                                    hipgraph_bool_t                do_expensive_check,
                                    hipgraph_centrality_result_t** result,
                                    hipgraph_error_t**             error);

/**
 * @brief     Opaque edge centrality result type
 */
typedef struct
{
    /** @brief align_ result type */
    int32_t align_;
} hipgraph_edge_centrality_result_t;

/**
 * @ingroup centrality
 * @brief     Get the src vertex ids from an edge centrality result
 *
 * @param [in]   result   The result from an edge centrality algorithm
 * @return type erased array of src vertex ids
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_edge_centrality_result_get_src_vertices(hipgraph_edge_centrality_result_t* result);

/**
 * @ingroup centrality
 * @brief     Get the dst vertex ids from an edge centrality result
 *
 * @param [in]   result   The result from an edge centrality algorithm
 * @return type erased array of dst vertex ids
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_edge_centrality_result_get_dst_vertices(hipgraph_edge_centrality_result_t* result);

/**
 * @ingroup centrality
 * @brief     Get the edge ids from an edge centrality result
 *
 * @param [in]   result   The result from an edge centrality algorithm
 * @return type erased array of edge ids
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_edge_centrality_result_get_edge_ids(hipgraph_edge_centrality_result_t* result);

/**
 * @ingroup centrality
 * @brief     Get the centrality values from an edge centrality algorithm result
 *
 * @param [in]   result   The result from an edge centrality algorithm
 * @return type erased array view of centrality values
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_edge_centrality_result_get_values(hipgraph_edge_centrality_result_t* result);

/**
 * @ingroup centrality
 * @brief     Free centrality result
 *
 * @param [in]   result   The result from a centrality algorithm
 */
HIPGRAPH_EXPORT void
    hipgraph_edge_centrality_result_free(hipgraph_edge_centrality_result_t* result);

/**
 * @brief     Compute edge betweenness centrality
 *
 * Edge betweenness can be computed exactly by specifying vertex_list as NULL.  This will compute
 * betweenness centrality by doing a traversal from every vertex and counting the frequency that a
 * edge appears on a shortest path.
 *
 * Approximate betweenness can be computed specifying a list of vertices that should be
 * used as seeds for the traversals.  Note that the function hipgraph_select_random_vertices can be
 * used to create a list of seeds.
 *
 * @param [in]  handle             Handle for accessing resources
 * @param [in]  graph              Pointer to graph
 * @param [in]  vertex_list        Optionally specify a device array containing a list of vertices
 *                                 to use as seeds for betweenness centrality approximation
 * @param [in]  normalized         Normalize
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result             Opaque pointer to edge betweenness centrality results
 * @param [out] error              Pointer to an error object storing details of any error.  Will
 *                                 be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_edge_betweenness_centrality(
    const hipgraph_resource_handle_t*               handle,
    hipgraph_graph_t*                               graph,
    const hipgraph_type_erased_device_array_view_t* vertex_list,
    hipgraph_bool_t                                 normalized,
    hipgraph_bool_t                                 do_expensive_check,
    hipgraph_edge_centrality_result_t**             result,
    hipgraph_error_t**                              error);

/**
 * @brief     Opaque hits result type
 */
typedef struct
{
    /** @brief align_ result type */
    int32_t align_;
} hipgraph_hits_result_t;

/**
 * @ingroup centrality
 * @brief     Get the vertex ids from the hits result
 *
 * @param [in]   result   The result from hits
 * @return type erased array of vertex ids
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_hits_result_get_vertices(hipgraph_hits_result_t* result);

/**
 * @ingroup centrality
 * @brief     Get the hubs values from the hits result
 *
 * @param [in]   result   The result from hits
 * @return type erased array of hubs values
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_hits_result_get_hubs(hipgraph_hits_result_t* result);

/**
 * @ingroup centrality
 * @brief     Get the authorities values from the hits result
 *
 * @param [in]   result   The result from hits
 * @return type erased array of authorities values
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_hits_result_get_authorities(hipgraph_hits_result_t* result);

/**
 * @ingroup centrality
 * @brief   Get the score differences between the last two iterations
 *
 * @param [in]   result   The result from hits
 * @return score differences
 */
HIPGRAPH_EXPORT double
    hipgraph_hits_result_get_hub_score_differences(hipgraph_hits_result_t* result);

/**
 * @ingroup centrality
 * @brief   Get the actual number of iterations
 *
 * @param [in]   result   The result from hits
 * @return actual number of iterations
 */
HIPGRAPH_EXPORT size_t
    hipgraph_hits_result_get_number_of_iterations(hipgraph_hits_result_t* result);

/**
 * @ingroup centrality
 * @brief     Free hits result
 *
 * @param [in]   result   The result from hits
 */
HIPGRAPH_EXPORT void hipgraph_hits_result_free(hipgraph_hits_result_t* result);

/**
 * @brief     Compute hits
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  graph       Pointer to graph
 * @param [in]  epsilon     Error tolerance to check convergence. Convergence is assumed
 *                          if the sum of the differences in Hits values between two
 *                          consecutive iterations is less than the number of vertices
 *                          in the graph multiplied by @p epsilon.
 * @param [in]  max_iterations
 *                          Maximum number of Hits iterations.
 * @param [in]  initial_hubs_guess_vertices
 *                          Pointer to optional type erased device array containing
 *                          the vertex ids for an initial hubs guess.  If set to NULL
 *                          there is no initial guess.
 * @param [in]  initial_hubs_guess_values
 *                          Pointer to optional type erased device array containing
 *                          the values for an initial hubs guess.  If set to NULL
 *                          there is no initial guess.  Note that both
 *                          @p initial_hubs_guess_vertices and @p initial_hubs_guess_values
 *                          have to be specified (or they both have to be NULL).  Otherwise
 *                          this will be treated as an error.
 * @param [in]  normalize   A flag to normalize the results (if set to `true`)
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result      Opaque pointer to hits results
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_hits(const hipgraph_resource_handle_t*               handle,
                  hipgraph_graph_t*                               graph,
                  double                                          epsilon,
                  size_t                                          max_iterations,
                  const hipgraph_type_erased_device_array_view_t* initial_hubs_guess_vertices,
                  const hipgraph_type_erased_device_array_view_t* initial_hubs_guess_values,
                  hipgraph_bool_t                                 normalize,
                  hipgraph_bool_t                                 do_expensive_check,
                  hipgraph_hits_result_t**                        result,
                  hipgraph_error_t**                              error);

#ifdef __cplusplus
}
#endif
