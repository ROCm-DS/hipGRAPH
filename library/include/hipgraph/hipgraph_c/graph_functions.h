// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
#include "graph.h"
#include "resource_handle.h"


#include "hipgraph/hipgraph-common.h"
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief       Opaque vertex pair type
 */
typedef struct hipgraph_vertex_pairs hipgraph_vertex_pairs_t;

/**
 * @brief       Create vertex_pairs
 *
 * Input data will be shuffled to the proper GPU and stored in the
 * output vertex_pairs.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Graph to operate on
 * @param [in]  first        Type erased array of vertex ids for the first vertex of the pair
 * @param [in]  second       Type erased array of vertex ids for the second vertex of the pair
 * @param [out] vertex_pairs Opaque pointer to vertex_pairs
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_create_vertex_pairs(
  const hipgraph_resource_handle_t* handle,
  hipgraph_graph_t* graph,
  const hipgraph_type_erased_device_array_view_t* first,
  const hipgraph_type_erased_device_array_view_t* second,
  hipgraph_vertex_pairs_t** vertex_pairs,
  hipgraph_error_t** error);

/**
 * @brief       Get the first vertex id array
 *
 * @param [in]     vertex_pairs   A vertex_pairs
 * @return type erased array of vertex ids
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_vertex_pairs_get_first(
  hipgraph_vertex_pairs_t* vertex_pairs);

/**
 * @brief       Get the second vertex id array
 *
 * @param [in]     vertex_pairs   A vertex_pairs
 * @return type erased array of vertex ids
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_vertex_pairs_get_second(
  hipgraph_vertex_pairs_t* vertex_pairs);

/**
 * @brief     Free vertex pair
 *
 * @param [in]    vertex_pairs The vertex pairs
 */
HIPGRAPH_EXPORT void hipgraph_vertex_pairs_free(hipgraph_vertex_pairs_t* vertex_pairs);

/**
 * @brief      Find all 2-hop neighbors in the graph
 *
 * @param [in]  handle         Handle for accessing resources
 * @param [in]  graph          Pointer to graph
 * @param [in]  start_vertices Optional type erased array of starting vertices
 *                             If NULL use all, if specified compute two-hop
 *                             neighbors for these starting vertices
 * @param [in]  do_expensive_check
 *                             A flag to run expensive checks for input arguments (if set to true)
 * @param [out] result         Opaque pointer to resulting vertex pairs
 * @param [out] error          Pointer to an error object storing details of any error.  Will
 *                             be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_two_hop_neighbors(
  const hipgraph_resource_handle_t* handle,
  hipgraph_graph_t* graph,
  const hipgraph_type_erased_device_array_view_t* start_vertices,
  bool do_expensive_check,
  hipgraph_vertex_pairs_t** result,
  hipgraph_error_t** error);

/**
 * @brief       Opaque induced subgraph type
 *
 * @deprecated  This API will be deleted, use hipgraph_edgelist_t
 */
typedef struct hipgraph_induced_subgraph_result hipgraph_induced_subgraph_result_t;

/**
 * @brief       Get the source vertex ids
 *
 * @deprecated  This API will be deleted, use hipgraph_edgelist_get_sources
 *
 * @param [in]     induced_subgraph   Opaque pointer to induced subgraph
 * @return type erased array view of source vertex ids
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_induced_subgraph_get_sources(
  hipgraph_induced_subgraph_result_t* induced_subgraph);

/**
 * @brief       Get the destination vertex ids
 *
 * @deprecated  This API will be deleted, use hipgraph_edgelist_get_destinations
 *
 * @param [in]     induced_subgraph   Opaque pointer to induced subgraph
 * @return type erased array view of destination vertex ids
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_induced_subgraph_get_destinations(
  hipgraph_induced_subgraph_result_t* induced_subgraph);

/**
 * @brief       Get the edge weights
 *
 * @deprecated  This API will be deleted, use hipgraph_edgelist_get_edge_weights
 *
 * @param [in]     induced_subgraph   Opaque pointer to induced subgraph
 * @return type erased array view of edge weights
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_induced_subgraph_get_edge_weights(
  hipgraph_induced_subgraph_result_t* induced_subgraph);

/**
 * @brief       Get the edge ids
 *
 * @deprecated  This API will be deleted, use hipgraph_edgelist_get_edge_ids
 *
 * @param [in]     induced_subgraph   Opaque pointer to induced subgraph
 * @return type erased array view of edge ids
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_induced_subgraph_get_edge_ids(
  hipgraph_induced_subgraph_result_t* induced_subgraph);

/**
 * @brief       Get the edge types
 *
 * @deprecated  This API will be deleted, use hipgraph_edgelist_get_edge_type_ids
 *
 * @param [in]     induced_subgraph   Opaque pointer to induced subgraph
 * @return type erased array view of edge types
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_induced_subgraph_get_edge_type_ids(
  hipgraph_induced_subgraph_result_t* induced_subgraph);

/**
 * @brief       Get the subgraph offsets
 *
 * @deprecated  This API will be deleted, use hipgraph_edgelist_get_edge_offsets
 *
 * @param [in]     induced_subgraph   Opaque pointer to induced subgraph
 * @return type erased array view of subgraph identifiers
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_induced_subgraph_get_subgraph_offsets(
  hipgraph_induced_subgraph_result_t* induced_subgraph);

/**
 * @brief     Free induced subgraph
 *
 * @deprecated  This API will be deleted, use hipgraph_edgelist_free
 *
 * @param [in]    induced subgraph   Opaque pointer to induced subgraph
 */
HIPGRAPH_EXPORT void hipgraph_induced_subgraph_result_free(hipgraph_induced_subgraph_result_t* induced_subgraph);

/**
 * @brief      Extract induced subgraph(s)
 *
 * Given a list of vertex ids, extract a list of edges that represent the subgraph
 * containing only the specified vertex ids.
 *
 * This function will do multiple subgraph extractions concurrently.  The vertex ids
 * are specified in CSR-style, with @p subgraph_vertices being a list of vertex ids
 * and @p subgraph_offsets[i] identifying the start offset for each extracted subgraph
 *
 * @param [in]  handle            Handle for accessing resources
 * @param [in]  graph             Pointer to graph
 * @param [in]  subgraph_offsets  Type erased array of subgraph offsets into
 *                                @p subgraph_vertices
 * @param [in]  subgraph_vertices Type erased array of vertices to include in
 *                                extracted subgraph.
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result            Opaque pointer to induced subgraph result
 * @param [out] error             Pointer to an error object storing details of any error.  Will
 *                                be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_extract_induced_subgraph(
  const hipgraph_resource_handle_t* handle,
  hipgraph_graph_t* graph,
  const hipgraph_type_erased_device_array_view_t* subgraph_offsets,
  const hipgraph_type_erased_device_array_view_t* subgraph_vertices,
  bool do_expensive_check,
  hipgraph_induced_subgraph_result_t** result,
  hipgraph_error_t** error);

// FIXME: Rename the return type
/**
 * @brief      Gather edgelist
 *
 * This function collects the edgelist from all ranks and stores the combine edgelist
 * in each rank
 *
 * @param [in]  handle            Handle for accessing resources.
 * @param [in]  src               Device array containing the source vertex ids.
 * @param [in]  dst               Device array containing the destination vertex ids
 * @param [in]  weights           Optional device array containing the edge weights
 * @param [in]  edge_ids          Optional device array containing the edge ids for each edge.
 * @param [in]  edge_type_ids     Optional device array containing the edge types for each edge
 * @param [out] result            Opaque pointer to gathered edgelist result
 * @param [out] error             Pointer to an error object storing details of any error.  Will
 *                                be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_allgather(const hipgraph_resource_handle_t* handle,
                                       const hipgraph_type_erased_device_array_view_t* src,
                                       const hipgraph_type_erased_device_array_view_t* dst,
                                       const hipgraph_type_erased_device_array_view_t* weights,
                                       const hipgraph_type_erased_device_array_view_t* edge_ids,
                                       const hipgraph_type_erased_device_array_view_t* edge_type_ids,
                                       hipgraph_induced_subgraph_result_t** result,
                                       hipgraph_error_t** error);

/**
 * @brief      Count multi_edges
 *
 * Count the number of multi-edges in the graph
 *
 * @param [in]  handle              Handle for accessing resources.
 * @param [in]  graph               Pointer to graph
 * @param [in]  do_expensive_check  A flag to run expensive checks for input arguments (if set to
 * true)
 * @param [out] result              Where to store the count of multi-edges
 * @param [out] error               Pointer to an error object storing details of any error.  Will
 *                                  be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_count_multi_edges(const hipgraph_resource_handle_t* handle,
                                               hipgraph_graph_t* graph,
                                               bool do_expensive_check,
                                               size_t* result,
                                               hipgraph_error_t** error);

/**
 * @brief       Opaque degree result type
 */
typedef struct hipgraph_degrees_result hipgraph_degrees_result_t;

/**
 * @brief      Compute in degrees
 *
 * Compute the in degrees for the vertices in the graph.
 *
 * @param [in]  handle              Handle for accessing resources.
 * @param [in]  graph               Pointer to graph
 * @param [in]  source_vertices     Device array of vertices we want to compute in degrees for.
 * @param [in]  do_expensive_check  A flag to run expensive checks for input arguments (if set to
 * true)
 * @param [out] result              Opaque pointer to degrees result
 * @param [out] error               Pointer to an error object storing details of any error.  Will
 *                                  be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_in_degrees(
  const hipgraph_resource_handle_t* handle,
  hipgraph_graph_t* graph,
  const hipgraph_type_erased_device_array_view_t* source_vertices,
  bool do_expensive_check,
  hipgraph_degrees_result_t** result,
  hipgraph_error_t** error);

/**
 * @brief      Compute out degrees
 *
 * Compute the out degrees for the vertices in the graph.
 *
 * @param [in]  handle              Handle for accessing resources.
 * @param [in]  graph               Pointer to graph
 * @param [in]  source_vertices     Device array of vertices we want to compute out degrees for.
 * @param [in]  do_expensive_check  A flag to run expensive checks for input arguments (if set to
 * true)
 * @param [out] result              Opaque pointer to degrees result
 * @param [out] error               Pointer to an error object storing details of any error.  Will
 *                                  be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_out_degrees(
  const hipgraph_resource_handle_t* handle,
  hipgraph_graph_t* graph,
  const hipgraph_type_erased_device_array_view_t* source_vertices,
  bool do_expensive_check,
  hipgraph_degrees_result_t** result,
  hipgraph_error_t** error);

/**
 * @brief      Compute degrees
 *
 * Compute the degrees for the vertices in the graph.
 *
 * @param [in]  handle              Handle for accessing resources.
 * @param [in]  graph               Pointer to graph
 * @param [in]  source_vertices     Device array of vertices we want to compute degrees for.
 * @param [in]  do_expensive_check  A flag to run expensive checks for input arguments (if set to
 * true)
 * @param [out] result              Opaque pointer to degrees result
 * @param [out] error               Pointer to an error object storing details of any error.  Will
 *                                  be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_degrees(const hipgraph_resource_handle_t* handle,
                                     hipgraph_graph_t* graph,
                                     const hipgraph_type_erased_device_array_view_t* source_vertices,
                                     bool do_expensive_check,
                                     hipgraph_degrees_result_t** result,
                                     hipgraph_error_t** error);

/**
 * @brief       Get the vertex ids
 *
 * @param [in]     degrees_result   Opaque pointer to degree result
 * @return type erased array view of vertex ids
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_degrees_result_get_vertices(
  hipgraph_degrees_result_t* degrees_result);

/**
 * @brief       Get the in degrees
 *
 * @param [in]     degrees_result   Opaque pointer to degree result
 * @return type erased array view of vertex ids
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_degrees_result_get_in_degrees(
  hipgraph_degrees_result_t* degrees_result);

/**
 * @brief       Get the out degrees
 *
 * If the graph is symmetric, in degrees and out degrees will be equal (and
 * will be stored in the same memory).
 *
 * @param [in]     degrees_result   Opaque pointer to degree result
 * @return type erased array view of vertex ids
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_degrees_result_get_out_degrees(
  hipgraph_degrees_result_t* degrees_result);

/**
 * @brief     Free degree result
 *
 * @param [in]    degrees_result   Opaque pointer to degree result
 */
HIPGRAPH_EXPORT void hipgraph_degrees_result_free(hipgraph_degrees_result_t* degrees_result);

/**
 * @brief       Opaque edgelist type
 *
 */
typedef struct hipgraph_edgelist hipgraph_edgelist_t;

/**
 * @brief       Get the source vertex ids
 *
 * @param [in]     edgelist   Opaque pointer to edgelist
 * @return type erased array view of source vertex ids
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_edgelist_get_sources(hipgraph_edgelist_t* edgelist);

/**
 * @brief       Get the destination vertex ids
 *
 * @param [in]     edgelist   Opaque pointer to edgelist
 * @return type erased array view of destination vertex ids
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_edgelist_get_destinations(
  hipgraph_edgelist_t* edgelist);

/**
 * @brief       Get the edge weights
 *
 * @param [in]     edgelist   Opaque pointer to edgelist
 * @return type erased array view of edge weights
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_edgelist_get_edge_weights(
  hipgraph_edgelist_t* edgelist);

/**
 * @brief       Get the edge ids
 *
 * @param [in]     edgelist   Opaque pointer to edgelist
 * @return type erased array view of edge ids
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_edgelist_get_edge_ids(
  hipgraph_edgelist_t* edgelist);

/**
 * @brief       Get the edge types
 *
 * @param [in]     edgelist   Opaque pointer to edgelist
 * @return type erased array view of edge types
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_edgelist_get_edge_type_ids(
  hipgraph_edgelist_t* edgelist);

/**
 * @brief       Get the edge offsets
 *
 * @param [in]     edgelist   Opaque pointer to edgelist
 * @return type erased array view of subgraph identifiers
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_edgelist_get_edge_offsets(
  hipgraph_edgelist_t* edgelist);

/**
 * @brief     Free edgelist
 *
 * @param [in]    edgelist   Opaque pointer to edgelist
 */
HIPGRAPH_EXPORT void hipgraph_edgelist_free(hipgraph_edgelist_t* edgelist);

/**
 * @brief       Construct the edge list from the graph view object.
 *
 * @param [in]  handle              Handle for accessing resources
 * @param [in]  graph               Graph to operate on
 * @param [in]  do_expensive_check  A flag to run expensive checks for input arguments (if set to
 * true)
 * @param [out] result              Opaque pointer to edgelist
 * @param [out] error               Pointer to an error object storing details of any error.  Will
 *                                  be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_decompress_to_edgelist(const hipgraph_resource_handle_t* handle,
                                                    hipgraph_graph_t* graph,
                                                    bool do_expensive_check,
                                                    hipgraph_edgelist_t** result,
                                                    hipgraph_error_t** error);

/**
 * @brief     Renumber arbitrary edgelist
 *
 * This function is designed to assist renumbering graph vertices in the case where the
 * the global vertex id list exceeds the GPU memory.  Renumbering is done in-place in the
 * supplied @p src and @p dst parameters.
 *
 * @param [in]     handle         Handle for accessing resources
 * @param [in]     renumber_map   Host array with the renumber map
 * @param [in/out] srcs           Device array of src vertices to renumber
 * @param [in/out] dsts           Device array of dst vertices to renumber
 * @param [out] error             Pointer to an error object storing details of any error.  Will
 *                                be populated if error code is not HIPGRAPH_SUCCESS
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_renumber_arbitrary_edgelist(
  const hipgraph_resource_handle_t* handle,
  const hipgraph_type_erased_host_array_view_t* renumber_map,
  hipgraph_type_erased_device_array_view_t* srcs,
  hipgraph_type_erased_device_array_view_t* dsts,
  hipgraph_error_t** error);

#ifdef __cplusplus
}
#endif
