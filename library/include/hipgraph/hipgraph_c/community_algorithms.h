// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*! \file */
/* ************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "hipgraph_c/error.h"
#include "hipgraph_c/graph.h"
#include "hipgraph_c/graph_functions.h"
#include "hipgraph_c/random.h"
#include "hipgraph_c/resource_handle.h"

/** @defgroup community Community algorithms
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief     Opaque triangle counting result type
 */
typedef struct
{
    /** @brief align_ result type */
    int32_t align_;
} hipgraph_triangle_count_result_t;

/**
 * @brief     Triangle Counting
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [in]  start        Device array of vertices we want to count triangles for.  If NULL
 *                           the entire set of vertices in the graph is processed
 * @param [in]  do_expensive_check
 *                           A flag to run expensive checks for input arguments (if set to true)
 * @param [out] result       Output from the triangle_count call
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_triangle_count(const hipgraph_resource_handle_t*               handle,
                            hipgraph_graph_t*                               graph,
                            const hipgraph_type_erased_device_array_view_t* start,
                            hipgraph_bool_t                                 do_expensive_check,
                            hipgraph_triangle_count_result_t**              result,
                            hipgraph_error_t**                              error);

/**
 * @ingroup community
 * @brief     Get triangle counting vertices
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_triangle_count_result_get_vertices(hipgraph_triangle_count_result_t* result);

/**
 * @ingroup community
 * @brief     Get triangle counting counts
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_triangle_count_result_get_counts(hipgraph_triangle_count_result_t* result);

/**
 * @ingroup community
 * @brief     Free a triangle count result
 *
 * @param [in] result     The result from a sampling algorithm
 */
HIPGRAPH_EXPORT void hipgraph_triangle_count_result_free(hipgraph_triangle_count_result_t* result);

/**
 * @brief     Opaque hierarchical clustering output
 */
typedef struct
{
    /** @brief align_ result type */
    int32_t align_;
} hipgraph_hierarchical_clustering_result_t;

/**
 * @brief     Compute Louvain
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [in]  max_level    Maximum level in hierarchy
 * @param [in]  threshold    Threshold parameter, defines convergence at each level of hierarchy
 * @param [in]  resolution   Resolution parameter (gamma) in modularity formula.
 *                           This changes the size of the communities.  Higher resolutions
 *                           lead to more smaller communities, lower resolutions lead to
 *                           fewer larger communities.
 * @param [in]  do_expensive_check
 *                           A flag to run expensive checks for input arguments (if set to true)
 * @param [out] result       Output from the Louvain call
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_louvain(const hipgraph_resource_handle_t*           handle,
                     hipgraph_graph_t*                           graph,
                     size_t                                      max_level,
                     double                                      threshold,
                     double                                      resolution,
                     hipgraph_bool_t                             do_expensive_check,
                     hipgraph_hierarchical_clustering_result_t** result,
                     hipgraph_error_t**                          error);

/**
 * @brief     Compute Leiden
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [inout] rng_state State of the random number generator, updated with each call
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [in]  max_level    Maximum level in hierarchy
 * @param [in]  resolution   Resolution parameter (gamma) in modularity formula.
 *                           This changes the size of the communities.  Higher resolutions
 *                           lead to more smaller communities, lower resolutions lead to
 *                           fewer larger communities.
 * @param[in]  theta         (optional) The value of the parameter to scale modularity
 *                           gain in Leiden refinement phase. It is used to compute
 *                           the probability of joining a random leiden community.
 *                           Called theta in the Leiden algorithm.
 * @param [in]  do_expensive_check
 *                           A flag to run expensive checks for input arguments (if set to true)
 * @param [out] result       Output from the Leiden call
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_leiden(const hipgraph_resource_handle_t*           handle,
                    hipgraph_rng_state_t*                       rng_state,
                    hipgraph_graph_t*                           graph,
                    size_t                                      max_level,
                    double                                      resolution,
                    double                                      theta,
                    hipgraph_bool_t                             do_expensive_check,
                    hipgraph_hierarchical_clustering_result_t** result,
                    hipgraph_error_t**                          error);

/**
 * @ingroup community
 * @brief     Get hierarchical clustering vertices
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_hierarchical_clustering_result_get_vertices(
        hipgraph_hierarchical_clustering_result_t* result);

/**
 * @ingroup community
 * @brief     Get hierarchical clustering clusters
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_hierarchical_clustering_result_get_clusters(
        hipgraph_hierarchical_clustering_result_t* result);

/**
 * @ingroup community
 * @brief     Get modularity
 */
HIPGRAPH_EXPORT double hipgraph_hierarchical_clustering_result_get_modularity(
    hipgraph_hierarchical_clustering_result_t* result);

/**
 * @ingroup community
 * @brief     Free a hierarchical clustering result
 *
 * @param [in] result     The result from a sampling algorithm
 */
HIPGRAPH_EXPORT void
    hipgraph_hierarchical_clustering_result_free(hipgraph_hierarchical_clustering_result_t* result);

/**
 * @brief     Compute ECG clustering of the given graph
 *
 * ECG runs truncated Louvain on an ensemble of permutations of the input graph,
 * then uses the ensemble partitions to determine weights for the input graph.
 * The final result is found by running full Louvain on the input graph using
 * the determined weights. See https://arxiv.org/abs/1809.05578 for further
 * information.
 *
 * @param [in]  handle        Handle for accessing resources
 * @param [inout] rng_state  State of the random number generator, updated with each call
 * @param [in]  graph         Pointer to graph.  NOTE: Graph might be modified if the storage
 *                            needs to be transposed
 * @param [in]  min_weight    Minimum edge weight in final graph
 * @param [in]  ensemble_size The number of Louvain iterations to run
 * @param [in]  max_level     Maximum level in hierarchy for final Louvain
 * @param [in]  threshold     Threshold parameter, defines convergence at each level of hierarchy
 *                            for final Louvain
 * @param [in]  resolution    Resolution parameter (gamma) in modularity formula.
 *                            This changes the size of the communities.  Higher resolutions
 *                            lead to more smaller communities, lower resolutions lead to
 *                            fewer larger communities.
 * @param [in]  do_expensive_check
 *                            A flag to run expensive checks for input arguments (if set to true)
 * @param [out] result        Output from the Louvain call
 * @param [out] error         Pointer to an error object storing details of any error.  Will
 *                            be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_ecg(const hipgraph_resource_handle_t*           handle,
                 hipgraph_rng_state_t*                       rng_state,
                 hipgraph_graph_t*                           graph,
                 double                                      min_weight,
                 size_t                                      ensemble_size,
                 size_t                                      max_level,
                 double                                      threshold,
                 double                                      resolution,
                 hipgraph_bool_t                             do_expensive_check,
                 hipgraph_hierarchical_clustering_result_t** result,
                 hipgraph_error_t**                          error);

/**
 * @brief   Extract ego graphs
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  source_vertices Device array of vertices we want to extract egonets for.
 * @param [in]  radius          The number of hops to go out from each source vertex
 * @param [in]  do_expensive_check
 *                               A flag to run expensive checks for input arguments (if set to true)
 * @param [out] result           Opaque object containing the extracted subgraph
 * @param [out] error            Pointer to an error object storing details of any error.  Will
 *                               be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_extract_ego(const hipgraph_resource_handle_t*               handle,
                         hipgraph_graph_t*                               graph,
                         const hipgraph_type_erased_device_array_view_t* source_vertices,
                         size_t                                          radius,
                         hipgraph_bool_t                                 do_expensive_check,
                         hipgraph_induced_subgraph_result_t**            result,
                         hipgraph_error_t**                              error);

/**
 * @brief   Extract k truss for a graph
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  k               The order of the truss
 * @param [in]  do_expensive_check
 *                              A flag to run expensive checks for input arguments (if set to true)
 * @param [out] result          Opaque object containing the extracted subgraph
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_k_truss_subgraph(const hipgraph_resource_handle_t*    handle,
                              hipgraph_graph_t*                    graph,
                              size_t                               k,
                              hipgraph_bool_t                      do_expensive_check,
                              hipgraph_induced_subgraph_result_t** result,
                              hipgraph_error_t**                   error);

/**
 * @brief     Opaque clustering output
 */
typedef struct
{
    /** @brief align_ result type */
    int32_t align_;
} hipgraph_clustering_result_t;

/**
 * @brief   Balanced cut clustering
 *
 * NOTE: This currently wraps the legacy balanced cut clustering implementation and is only
 * available in Single GPU implementation.
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  n_clusters      The desired number of clusters
 * @param [in]  n_eigenvectors  The number of eigenvectors to use
 * @param [in]  evs_tolerance   The tolerance to use for the eigenvalue solver
 * @param [in]  evs_max_iterations The maximum number of iterations of the eigenvalue solver
 * @param [in]  k_means_tolerance  The tolerance to use for the k-means solver
 * @param [in]  k_means_max_iterations The maximum number of iterations of the k-means solver
 * @param [in]  do_expensive_check
 *                               A flag to run expensive checks for input arguments (if set to true)
 * @param [out] result           Opaque object containing the clustering result
 * @param [out] error            Pointer to an error object storing details of any error.  Will
 *                               be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_balanced_cut_clustering(const hipgraph_resource_handle_t* handle,
                                     hipgraph_graph_t*                 graph,
                                     size_t                            n_clusters,
                                     size_t                            n_eigenvectors,
                                     double                            evs_tolerance,
                                     int                               evs_max_iterations,
                                     double                            k_means_tolerance,
                                     int                               k_means_max_iterations,
                                     hipgraph_bool_t                   do_expensive_check,
                                     hipgraph_clustering_result_t**    result,
                                     hipgraph_error_t**                error);

/**
 * @brief   Spectral clustering
 *
 * NOTE: This currently wraps the legacy spectral clustering implementation and is only
 * available in Single GPU implementation.
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  n_clusters      The desired number of clusters
 * @param [in]  n_eigenvectors  The number of eigenvectors to use
 * @param [in]  evs_tolerance   The tolerance to use for the eigenvalue solver
 * @param [in]  evs_max_iterations The maximum number of iterations of the eigenvalue solver
 * @param [in]  k_means_tolerance  The tolerance to use for the k-means solver
 * @param [in]  k_means_max_iterations The maximum number of iterations of the k-means solver
 * @param [in]  do_expensive_check
 *                               A flag to run expensive checks for input arguments (if set to true)
 * @param [out] result           Opaque object containing the clustering result
 * @param [out] error            Pointer to an error object storing details of any error.  Will
 *                               be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
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
                                              hipgraph_error_t**             error);

/**
 * @brief   Compute modularity of the specified clustering
 *
 * NOTE: This currently wraps the legacy spectral modularity implementation and is only
 * available in Single GPU implementation.
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  n_clusters      The desired number of clusters
 * @param [in]  vertices        Vertex ids from the clustering result
 * @param [in]  clusters        Cluster ids from the clustering result
 * @param [out] score           The modularity score for this clustering
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_analyze_clustering_modularity(const hipgraph_resource_handle_t* handle,
                                           hipgraph_graph_t*                 graph,
                                           size_t                            n_clusters,
                                           const hipgraph_type_erased_device_array_view_t* vertices,
                                           const hipgraph_type_erased_device_array_view_t* clusters,
                                           double*                                         score,
                                           hipgraph_error_t**                              error);

/**
 * @brief   Compute edge cut of the specified clustering
 *
 * NOTE: This currently wraps the legacy spectral edge cut implementation and is only
 * available in Single GPU implementation.
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  n_clusters      The desired number of clusters
 * @param [in]  vertices        Vertex ids from the clustering result
 * @param [in]  clusters        Cluster ids from the clustering result
 * @param [out] score           The edge cut score for this clustering
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_analyze_clustering_edge_cut(const hipgraph_resource_handle_t*               handle,
                                         hipgraph_graph_t*                               graph,
                                         size_t                                          n_clusters,
                                         const hipgraph_type_erased_device_array_view_t* vertices,
                                         const hipgraph_type_erased_device_array_view_t* clusters,
                                         double*                                         score,
                                         hipgraph_error_t**                              error);

/**
 * @brief   Compute ratio cut of the specified clustering
 *
 * NOTE: This currently wraps the legacy spectral ratio cut implementation and is only
 * available in Single GPU implementation.
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  n_clusters      The desired number of clusters
 * @param [in]  vertices        Vertex ids from the clustering result
 * @param [in]  clusters        Cluster ids from the clustering result
 * @param [out] score           The ratio cut score for this clustering
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_analyze_clustering_ratio_cut(const hipgraph_resource_handle_t* handle,
                                          hipgraph_graph_t*                 graph,
                                          size_t                            n_clusters,
                                          const hipgraph_type_erased_device_array_view_t* vertices,
                                          const hipgraph_type_erased_device_array_view_t* clusters,
                                          double*                                         score,
                                          hipgraph_error_t**                              error);

/**
 * @brief     Get clustering vertices
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_clustering_result_get_vertices(hipgraph_clustering_result_t* result);

/**
 * @brief     Get clustering clusters
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_clustering_result_get_clusters(hipgraph_clustering_result_t* result);

/**
 * @brief     Free a clustering result
 *
 * @param [in] result     The result from a sampling algorithm
 */
HIPGRAPH_EXPORT void hipgraph_clustering_result_free(hipgraph_clustering_result_t* result);

#ifdef __cplusplus
}
#endif
