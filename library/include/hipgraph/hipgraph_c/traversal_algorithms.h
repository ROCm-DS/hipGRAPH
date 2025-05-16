// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*! \file */
/* ************************************************************************
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include "hipgraph/hipgraph_c/error.h"
#include "hipgraph/hipgraph_c/graph.h"
#include "hipgraph/hipgraph_c/resource_handle.h"

/** @defgroup traversal Traversal Algorithms
 *  @ingroup c_api
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief     Opaque paths result type
 *
 * Store the output of BFS or SSSP, computing predecessors and distances
 * from a seed.
 */
typedef struct
{
    /** @brief align_ result type */
    int32_t align_;
} hipgraph_paths_result_t;

/**
 * @ingroup traversal
 * @brief     Get the vertex ids from the paths result
 *
 * @param [in]   result   The result from bfs or sssp
 * @return type erased array of vertex ids
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_paths_result_get_vertices(hipgraph_paths_result_t* result);

/**
 * @ingroup traversal
 * @brief     Get the distances from the paths result
 *
 * @param [in]   result   The result from bfs or sssp
 * @return type erased array of distances
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_paths_result_get_distances(hipgraph_paths_result_t* result);

/**
 * @ingroup traversal
 * @brief     Get the predecessors from the paths result
 *
 * @param [in]   result   The result from bfs or sssp
 * @return type erased array of predecessors.  Value will be NULL if
 *         compute_predecessors was FALSE in the call to bfs or sssp that
 *         produced this result.
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_paths_result_get_predecessors(hipgraph_paths_result_t* result);

/**
 * @ingroup traversal
 * @brief     Free paths result
 *
 * @param [in]   result   The result from bfs or sssp
 */
HIPGRAPH_EXPORT void hipgraph_paths_result_free(hipgraph_paths_result_t* result);

/**
 * @brief     Perform a breadth first search from a set of seed vertices.
 *
 * This function computes the distances (minimum number of hops to reach the vertex) from the source
 * vertex. If @p predecessors is not NULL, this function calculates the predecessor of each
 * vertex (parent vertex in the breadth-first search tree) as well.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * FIXME:  Make this just [in], copy it if I need to temporarily modify internally
 * @param [inout]  sources  Array of source vertices.  NOTE: Array might be modified if
 *                           renumbering is enabled for the graph
 * @param [in]  direction_optimizing If set to true, this algorithm switches between the push based
 * breadth-first search and pull based breadth-first search depending on the size of the
 * breadth-first search frontier (currently unsupported). This option is valid only for symmetric
 * input graphs.
 * @param depth_limit Sets the maximum number of breadth-first search iterations. Any vertices
 * farther than @p depth_limit hops from @p source_vertex will be marked as unreachable.
 * @param [in] compute_predecessors A flag to indicate whether to compute the predecessors in the
 * result
 * @param [in] do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result       Opaque pointer to paths results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_bfs(const hipgraph_resource_handle_t* handle,
                 hipgraph_graph_t*                 graph,
                 // FIXME:  Make this const, copy it if I need to temporarily modify internally
                 hipgraph_type_erased_device_array_view_t* sources,
                 hipgraph_bool_t                           direction_optimizing,
                 size_t                                    depth_limit,
                 hipgraph_bool_t                           compute_predecessors,
                 hipgraph_bool_t                           do_expensive_check,
                 hipgraph_paths_result_t**                 result,
                 hipgraph_error_t**                        error);

/**
 * @brief     Perform single-source shortest-path to compute the minimum distances
 *            (and predecessors) from the source vertex.
 *
 * This function computes the distances (minimum edge weight sums) from the source
 * vertex. If @p predecessors is not NULL, this function calculates the predecessor of each
 * vertex (parent vertex in the breadth-first search tree) as well.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * @param [in]  source       Source vertex id
 * @param [in]  cutoff       Maximum edge weight sum to consider
 * @param [in]  compute_predecessors A flag to indicate whether to compute the predecessors in the
 * result
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result       Opaque pointer to paths results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_sssp(const hipgraph_resource_handle_t* handle,
                                                    hipgraph_graph_t*                 graph,
                                                    size_t                            source,
                                                    double                            cutoff,
                                                    hipgraph_bool_t           compute_predecessors,
                                                    hipgraph_bool_t           do_expensive_check,
                                                    hipgraph_paths_result_t** result,
                                                    hipgraph_error_t**        error);

/**
 * @brief     Opaque extract_paths result type
 */
typedef struct
{
    /** @brief align_ result type */
    int32_t align_;
} hipgraph_extract_paths_result_t;

/**
 * @brief     Extract BFS or SSSP paths from a hipgraph_paths_result_t
 *
 * This function extracts paths from the BFS or SSSP output.  BFS and SSSP output
 * distances and predecessors.  The path from a vertex v back to the original
 * source vertex can be extracted by recursively looking up the predecessor
 * vertex until you arrive back at the original source vertex.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [in]  sources      Array of source vertices
 * @param [in]  paths_result       Output from the BFS call
 * @param [in]  destinations Array of destination vertices.
 * @param [out] result       Opaque pointer to extract_paths results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_extract_paths(const hipgraph_resource_handle_t*               handle,
                           hipgraph_graph_t*                               graph,
                           const hipgraph_type_erased_device_array_view_t* sources,
                           const hipgraph_paths_result_t*                  paths_result,
                           const hipgraph_type_erased_device_array_view_t* destinations,
                           hipgraph_extract_paths_result_t**               result,
                           hipgraph_error_t**                              error);

/**
 * @brief     Get the max path length from extract_paths result
 *
 * @param [in]   result   The result from extract_paths
 * @return maximum path length
 */
HIPGRAPH_EXPORT size_t
    hipgraph_extract_paths_result_get_max_path_length(hipgraph_extract_paths_result_t* result);

/**
 * @ingroup traversal
 * @brief     Get the matrix (row major order) of paths
 *
 * @param [in]   result   The result from extract_paths
 * @return type erased array pointing to the matrix in device memory
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_extract_paths_result_get_paths(hipgraph_extract_paths_result_t* result);

/**
 * @ingroup traversal
 * @brief     Free extract_paths result
 *
 * @param [in]   result   The result from extract_paths
 */
HIPGRAPH_EXPORT void hipgraph_extract_paths_result_free(hipgraph_extract_paths_result_t* result);

#ifdef __cplusplus
}
#endif
