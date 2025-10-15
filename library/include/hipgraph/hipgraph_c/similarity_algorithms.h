// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
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

#pragma once

#include "array.h"
#include "error.h"
#include "graph.h"
#include "graph_functions.h"
#include "resource_handle.h"


#include "hipgraph/hipgraph-common.h"
/** @defgroup similarity Similarity algorithms
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief       Opaque similarity result type
 */
typedef struct hipgraph_similarity_result hipgraph_similarity_result_t;

/**
 * @ingroup similarity
 * @brief       Get vertex pair from the similarity result.
 *
 * @param [in]     result   The result from a similarity algorithm
 * @return vertex pairs
 */
HIPGRAPH_EXPORT hipgraph_vertex_pairs_t* hipgraph_similarity_result_get_vertex_pairs(
  hipgraph_similarity_result_t* result);

/**
 * @ingroup similarity
 * @brief       Get the similarity coefficient array
 *
 * @param [in]     result   The result from a similarity algorithm
 * @return type erased array of similarity coefficients
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_similarity_result_get_similarity(
  hipgraph_similarity_result_t* result);

/**
 * @ingroup similarity
 * @brief     Free similarity result
 *
 * @param [in]    result    The result from a similarity algorithm
 */
HIPGRAPH_EXPORT void hipgraph_similarity_result_free(hipgraph_similarity_result_t* result);

/**
 * @brief     Perform Jaccard similarity computation
 *
 * Compute the similarity for the specified vertex_pairs
 *
 * Note that Jaccard similarity must run on a symmetric graph.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * @param [in]  vertex_pairs Vertex pair for input
 * @param [in]  use_weight   If true consider the edge weight in the graph, if false use an
 *                           edge weight of 1
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result       Opaque pointer to similarity results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_jaccard_coefficients(const hipgraph_resource_handle_t* handle,
                                                  hipgraph_graph_t* graph,
                                                  const hipgraph_vertex_pairs_t* vertex_pairs,
                                                  bool use_weight,
                                                  bool do_expensive_check,
                                                  hipgraph_similarity_result_t** result,
                                                  hipgraph_error_t** error);

/**
 * @brief     Perform Sorensen similarity computation
 *
 * Compute the similarity for the specified vertex_pairs
 *
 * Note that Sorensen similarity must run on a symmetric graph.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * @param [in]  vertex_pairs Vertex pair for input
 * @param [in]  use_weight   If true consider the edge weight in the graph, if false use an
 *                           edge weight of 1
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result       Opaque pointer to similarity results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_sorensen_coefficients(const hipgraph_resource_handle_t* handle,
                                                   hipgraph_graph_t* graph,
                                                   const hipgraph_vertex_pairs_t* vertex_pairs,
                                                   bool use_weight,
                                                   bool do_expensive_check,
                                                   hipgraph_similarity_result_t** result,
                                                   hipgraph_error_t** error);

/**
 * @brief     Perform overlap similarity computation
 *
 * Compute the similarity for the specified vertex_pairs
 *
 * Note that overlap similarity must run on a symmetric graph.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * @param [in]  vertex_pairs Vertex pair for input
 * @param [in]  use_weight   If true consider the edge weight in the graph, if false use an
 *                           edge weight of 1
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result       Opaque pointer to similarity results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_overlap_coefficients(const hipgraph_resource_handle_t* handle,
                                                  hipgraph_graph_t* graph,
                                                  const hipgraph_vertex_pairs_t* vertex_pairs,
                                                  bool use_weight,
                                                  bool do_expensive_check,
                                                  hipgraph_similarity_result_t** result,
                                                  hipgraph_error_t** error);

/**
 * @brief     Perform cosine similarity computation
 *
 * Compute the similarity for the specified vertex_pairs
 *
 * Note that cosine similarity must run on a symmetric graph.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * @param [in]  vertex_pairs Vertex pair for input
 * @param [in]  use_weight   If true consider the edge weight in the graph, if false use an
 *                           edge weight of 1
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result       Opaque pointer to similarity results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_cosine_similarity_coefficients(
  const hipgraph_resource_handle_t* handle,
  hipgraph_graph_t* graph,
  const hipgraph_vertex_pairs_t* vertex_pairs,
  bool use_weight,
  bool do_expensive_check,
  hipgraph_similarity_result_t** result,
  hipgraph_error_t** error);

/**
 * @brief     Perform All-Pairs Jaccard similarity computation
 *
 * Compute the similarity for all vertex pairs derived from the two-hop neighbors
 * of an optional specified vertex list.  This function will identify the two-hop
 * neighbors of the specified vertices (all vertices in the graph if not specified)
 * and compute similarity for those vertices.
 *
 * If the topk parameter is specified then the result will only contain the top k
 * highest scoring results.
 *
 * Note that Jaccard similarity must run on a symmetric graph.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * @param [in]  vertices     Vertex list for input.  If null then compute based on
 *                           all vertices in the graph.
 * @param [in]  use_weight   If true consider the edge weight in the graph, if false use an
 *                           edge weight of 1
 * @param [in]  topk         Specify how many answers to return.  Specifying SIZE_MAX
 *                           will return all values.
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result       Opaque pointer to similarity results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_all_pairs_jaccard_coefficients(
  const hipgraph_resource_handle_t* handle,
  hipgraph_graph_t* graph,
  const hipgraph_type_erased_device_array_view_t* vertices,
  bool use_weight,
  size_t topk,
  bool do_expensive_check,
  hipgraph_similarity_result_t** result,
  hipgraph_error_t** error);

/**
 * @brief     Perform All Pairs Sorensen similarity computation
 *
 * Compute the similarity for all vertex pairs derived from the two-hop neighbors
 * of an optional specified vertex list.  This function will identify the two-hop
 * neighbors of the specified vertices (all vertices in the graph if not specified)
 * and compute similarity for those vertices.
 *
 * If the topk parameter is specified then the result will only contain the top k
 * highest scoring results.
 *
 * Note that Sorensen similarity must run on a symmetric graph.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * @param [in]  vertices     Vertex list for input.  If null then compute based on
 *                           all vertices in the graph.
 * @param [in]  use_weight   If true consider the edge weight in the graph, if false use an
 *                           edge weight of 1
 * @param [in]  topk         Specify how many answers to return.  Specifying SIZE_MAX
 *                           will return all values.
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result       Opaque pointer to similarity results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_all_pairs_sorensen_coefficients(
  const hipgraph_resource_handle_t* handle,
  hipgraph_graph_t* graph,
  const hipgraph_type_erased_device_array_view_t* vertices,
  bool use_weight,
  size_t topk,
  bool do_expensive_check,
  hipgraph_similarity_result_t** result,
  hipgraph_error_t** error);

/**
 * @brief     Perform All Pairs overlap similarity computation
 *
 * Compute the similarity for all vertex pairs derived from the two-hop neighbors
 * of an optional specified vertex list.  This function will identify the two-hop
 * neighbors of the specified vertices (all vertices in the graph if not specified)
 * and compute similarity for those vertices.
 *
 * If the topk parameter is specified then the result will only contain the top k
 * highest scoring results.
 *
 * Note that overlap similarity must run on a symmetric graph.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * @param [in]  vertices     Vertex list for input.  If null then compute based on
 *                           all vertices in the graph.
 * @param [in]  use_weight   If true consider the edge weight in the graph, if false use an
 *                           edge weight of 1
 * @param [in]  topk         Specify how many answers to return.  Specifying SIZE_MAX
 *                           will return all values.
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result       Opaque pointer to similarity results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_all_pairs_overlap_coefficients(
  const hipgraph_resource_handle_t* handle,
  hipgraph_graph_t* graph,
  const hipgraph_type_erased_device_array_view_t* vertices,
  bool use_weight,
  size_t topk,
  bool do_expensive_check,
  hipgraph_similarity_result_t** result,
  hipgraph_error_t** error);

/**
 * @brief     Perform All Pairs cosine similarity computation
 *
 * Compute the similarity for all vertex pairs derived from the two-hop neighbors
 * of an optional specified vertex list.  This function will identify the two-hop
 * neighbors of the specified vertices (all vertices in the graph if not specified)
 * and compute similarity for those vertices.
 *
 * If the topk parameter is specified then the result will only contain the top k
 * highest scoring results.
 *
 * Note that cosine similarity must run on a symmetric graph.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * @param [in]  vertices     Vertex list for input.  If null then compute based on
 *                           all vertices in the graph.
 * @param [in]  use_weight   If true consider the edge weight in the graph, if false use an
 *                           edge weight of 1
 * @param [in]  topk         Specify how many answers to return.  Specifying SIZE_MAX
 *                           will return all values.
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result       Opaque pointer to similarity results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_all_pairs_cosine_similarity_coefficients(
  const hipgraph_resource_handle_t* handle,
  hipgraph_graph_t* graph,
  const hipgraph_type_erased_device_array_view_t* vertices,
  bool use_weight,
  size_t topk,
  bool do_expensive_check,
  hipgraph_similarity_result_t** result,
  hipgraph_error_t** error);

#ifdef __cplusplus
}
#endif
