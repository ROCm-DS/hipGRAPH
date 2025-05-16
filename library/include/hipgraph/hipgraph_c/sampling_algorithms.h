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

#include "hipgraph_c/error.h"
#include "hipgraph_c/graph.h"
#include "hipgraph_c/random.h"
#include "hipgraph_c/resource_handle.h"

/** @defgroup samplingC Sampling algorithms
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief     Opaque random walk result type
 */
typedef struct
{
    /** @brief align_ result type */
    int32_t align_;
} hipgraph_random_walk_result_t;

/**
 * @brief  Compute uniform random walks
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  start_vertices  Array of source vertices
 * @param [in]  max_length      Maximum length of the generated path
 * @param [in]  result          Output from the node2vec call
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_uniform_random_walks(const hipgraph_resource_handle_t*               handle,
                                  hipgraph_graph_t*                               graph,
                                  const hipgraph_type_erased_device_array_view_t* start_vertices,
                                  size_t                                          max_length,
                                  hipgraph_random_walk_result_t**                 result,
                                  hipgraph_error_t**                              error);

/**
 * @brief  Compute biased random walks
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  start_vertices  Array of source vertices
 * @param [in]  max_length      Maximum length of the generated path
 * @param [in]  result          Output from the node2vec call
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_biased_random_walks(const hipgraph_resource_handle_t*               handle,
                                 hipgraph_graph_t*                               graph,
                                 const hipgraph_type_erased_device_array_view_t* start_vertices,
                                 size_t                                          max_length,
                                 hipgraph_random_walk_result_t**                 result,
                                 hipgraph_error_t**                              error);

/**
 * @brief  Compute random walks using the node2vec framework.
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  start_vertices  Array of source vertices
 * @param [in]  max_length      Maximum length of the generated path
 * @param [in]  p               The return parameter
 * @param [in]  q               The inout parameter
 * @param [in]  result          Output from the node2vec call
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_node2vec_random_walks(const hipgraph_resource_handle_t*               handle,
                                   hipgraph_graph_t*                               graph,
                                   const hipgraph_type_erased_device_array_view_t* start_vertices,
                                   size_t                                          max_length,
                                   double                                          p,
                                   double                                          q,
                                   hipgraph_random_walk_result_t**                 result,
                                   hipgraph_error_t**                              error);

/**
 * @brief  Compute random walks using the node2vec framework.
 * @deprecated This call should be replaced with hipgraph_node2vec_random_walks
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [in]  sources      Array of source vertices
 * @param [in]  max_depth    Maximum length of the generated path
 * @param [in]  compress_result If true, return the paths as a compressed sparse row matrix,
 *                              otherwise return as a dense matrix
 * @param [in]  p            The return parameter
 * @param [in]  q            The inout parameter
 * @param [in]  result       Output from the node2vec call
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_node2vec(const hipgraph_resource_handle_t*               handle,
                      hipgraph_graph_t*                               graph,
                      const hipgraph_type_erased_device_array_view_t* sources,
                      size_t                                          max_depth,
                      hipgraph_bool_t                                 compress_result,
                      double                                          p,
                      double                                          q,
                      hipgraph_random_walk_result_t**                 result,
                      hipgraph_error_t**                              error);

/**
 * @ingroup samplingC
 * @brief     Get the max path length from random walk result
 *
 * @param [in]   result   The result from random walks
 * @return maximum path length
 */
HIPGRAPH_EXPORT size_t
    hipgraph_random_walk_result_get_max_path_length(hipgraph_random_walk_result_t* result);

// FIXME:  Should this be the same as extract_paths_result_t?  The only
//         difference at the moment is that RW results contain weights
//         and extract_paths results don't.  But that's probably wrong.
/**
 * @ingroup samplingC
 * @brief     Get the matrix (row major order) of vertices in the paths
 *
 * @param [in]   result   The result from a random walk algorithm
 * @return type erased array pointing to the path matrix in device memory
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_random_walk_result_get_paths(hipgraph_random_walk_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the matrix (row major order) of edge weights in the paths
 *
 * @param [in]   result   The result from a random walk algorithm
 * @return type erased array pointing to the path edge weights in device memory
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_random_walk_result_get_weights(hipgraph_random_walk_result_t* result);

/**
 * @ingroup samplingC
 * @brief     If the random walk result is compressed, get the path sizes
 * @deprecated This call will no longer be relevant once the new node2vec are called
 *
 * @param [in]   result   The result from a random walk algorithm
 * @return type erased array pointing to the path sizes in device memory
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_random_walk_result_get_path_sizes(hipgraph_random_walk_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Free random walks result
 *
 * @param [in]   result   The result from random walks
 */
HIPGRAPH_EXPORT void hipgraph_random_walk_result_free(hipgraph_random_walk_result_t* result);

/**
 * @brief     Opaque neighborhood sampling result type
 */
typedef struct
{
    /** @brief align_ result type */
    int32_t align_;
} hipgraph_sample_result_t;

/**
 * @brief     Opaque sampling options type
 */
typedef struct
{
    /** @brief align_ result type */
    int32_t align_;
} hipgraph_sampling_options_t;

/**
 * @brief     Enumeration for prior sources behavior
 */
typedef enum hipgraph_prior_sources_behavior_
{
    HIPGRAPH_DEFAULT = 0, /** Construct sources for hop k from destination vertices from hop k-1 */
    HIPGRAPH_CARRY_OVER, /** Construct sources for hop k from destination vertices from hop k-1
									  and sources from hop k-1 */
    HIPGRAPH_EXCLUDE /** Construct sources for hop k from destination vertices form hop k-1,
									  but exclude any vertex that has already been used as a source */
} hipgraph_prior_sources_behavior_t;

/* See resource_handle.h for a caveat about the warning. */

#if !defined(HIPGRAPH_NO_NONPREFIXED_ALIASES)
#if defined(DEFAULT) || defined(CARRY_OVER) || defined(EXCLUDE) \
    || defined(hipgraph_prior_sources_behavior_t)
#warning \
    "cuGraph -> hipGRAPH macro aliases related to prior_sources_behavior_t may shadow existing definitions."
#endif
/**
 * @brief prior sources macros
 * @{
 */
#undef DEFAULT
#define DEFAULT HIPGRAPH_DEFAULT
#undef CARRY_OVER
#define CARRY_OVER HIPGRAPH_CARRY_OVER
#undef EXCLUDE
#define EXCLUDE HIPGRAPH_EXCLUDE
#undef hipgraph_prior_sources_behavior_t
#define prior_sources_behavior_t hipgraph_prior_sources_behavior_t
/** @} */
#endif

/**
 * @enum hipgraph_compression_type_t
 * @brief Selects the type of compression to use for the output samples.
 */
typedef enum hipgraph_compression_type_
{
    HIPGRAPH_COO = 0, /** Outputs in COO format.  Default. */
    HIPGRAPH_CSR, /** Compresses in CSR format.  This means the row (src) column
								is compressed into a row pointer. */
    HIPGRAPH_CSC, /** Compresses in CSC format.  This means the col (dst) column
								is compressed into a column pointer. */
    HIPGRAPH_DCSR, /** Compresses in DCSR format.  This outputs an additional index
								that avoids empty entries in the row pointer. */
    HIPGRAPH_DCSC /** Compresses in DCSC format.  This outputs an additional index
								that avoid empty entries in the col pointer. */
} hipgraph_compression_type_t;

#if !defined(HIPGRAPH_NO_NONPREFIXED_ALIASES)
#if defined(COO) || defined(CSR) || defined(CSC) || defined(DCSR) || defined(DCSC) \
    || defined(compression_type_t)
#warning \
    "cuGraph -> hipGRAPH macro aliases related to compression_type_t may shadow existing definitions."
#endif
/**
 * @brief storage type macros
 * @{
 */
#undef COO
#define COO HIPGRAPH_COO
#undef CSR
#define CSR HIPGRAPH_CSR
#undef CSC
#define CSC HIPGRAPH_CSC
#undef DCSR
#define DCSR HIPGRAPH_DCSR
#undef DCSC
#define DCSC HIPGRAPH_DCSC
#undef compression_type_t
#define compression_type_t hipgraph_compression_type_t
/** @} */
#endif

/**
 * @ingroup samplingC
 * @brief   Create sampling options object
 *
 * All sampling options set to FALSE
 *
 * @param [out] options Opaque pointer to the sampling options
 * @param [out] error   Pointer to an error object storing details of any error.  Will
 *                      be populated if error code is not HIPGRAPH_SUCCESS
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_sampling_options_create(
    hipgraph_sampling_options_t** options, hipgraph_error_t** error);

/**
 * @ingroup samplingC
 * @brief   Set flag to renumber results
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Boolean value to assign to the option
 */
HIPGRAPH_EXPORT void hipgraph_sampling_set_renumber_results(hipgraph_sampling_options_t* options,
                                                            hipgraph_bool_t              value);

/**
 * @ingroup samplingC
 * @brief   Set whether to compress per-hop (True) or globally (False)
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Boolean value to assign to the option
 */
HIPGRAPH_EXPORT void hipgraph_sampling_set_compress_per_hop(hipgraph_sampling_options_t* options,
                                                            hipgraph_bool_t              value);

/**
 * @ingroup samplingC
 * @brief   Set flag to sample with_replacement
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Boolean value to assign to the option
 */
HIPGRAPH_EXPORT void hipgraph_sampling_set_with_replacement(hipgraph_sampling_options_t* options,
                                                            hipgraph_bool_t              value);

/**
 * @ingroup samplingC
 * @brief   Set flag to sample return_hops
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Boolean value to assign to the option
 */
HIPGRAPH_EXPORT void hipgraph_sampling_set_return_hops(hipgraph_sampling_options_t* options,
                                                       hipgraph_bool_t              value);

/**
 * @ingroup samplingC
 * @brief   Set compression type
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Enum defining the compresion type
 */
HIPGRAPH_EXPORT void hipgraph_sampling_set_compression_type(hipgraph_sampling_options_t* options,
                                                            hipgraph_compression_type_t  value);

/**
 * @ingroup samplingC
 * @brief   Set prior sources behavior
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Enum defining prior sources behavior
 */
HIPGRAPH_EXPORT void
    hipgraph_sampling_set_prior_sources_behavior(hipgraph_sampling_options_t*      options,
                                                 hipgraph_prior_sources_behavior_t value);

/**
 * @ingroup samplingC
 * @brief   Set flag to sample dedupe_sources prior to sampling
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Boolean value to assign to the option
 */
HIPGRAPH_EXPORT void hipgraph_sampling_set_dedupe_sources(hipgraph_sampling_options_t* options,
                                                          hipgraph_bool_t              value);

/**
 * @ingroup samplingC
 * @brief     Free sampling options object
 *
 * @param [in]   options   Opaque pointer to sampling object
 */
HIPGRAPH_EXPORT void hipgraph_sampling_options_free(hipgraph_sampling_options_t* options);

/**
 * @brief     Uniform Neighborhood Sampling
 *
 * Returns a sample of the neighborhood around specified start vertices.  Optionally, each
 * start vertex can be associated with a label, allowing the caller to specify multiple batches
 * of sampling requests in the same function call - which should improve GPU utilization.
 *
 * If label is NULL then all start vertices will be considered part of the same batch and the
 * return value will not have a label column.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [in]  start_vertices Device array of start vertices for the sampling
 * @param [in]  start_vertex_labels  Device array of start vertex labels for the sampling.  The
 * labels associated with each start vertex will be included in the output associated with results
 * that were derived from that start vertex.  We only support label of type INT32. If label is
 * NULL, the return data will not be labeled.
 * @param [in]  label_list Device array of the labels included in @p start_vertex_labels.  If
 * @p label_to_comm_rank is not specified this parameter is ignored.  If specified, label_list
 * must be sorted in ascending order.
 * @param [in]  label_to_comm_rank Device array identifying which comm rank the output for a
 * particular label should be shuffled in the output.  If not specifed the data is not organized in
 * output.  If specified then the all data from @p label_list[i] will be shuffled to rank.  This
 * cannot be specified unless @p start_vertex_labels is also specified label_to_comm_rank[i].
 * If not specified then the output data will not be shuffled between ranks.
 * @param [in] label_offsets tbd
 * @param [in]  fan_out       Host array defining the fan out at each step in the sampling algorithm.
 *                           We only support fanout values of type INT32
 * @param [inout] rng_state State of the random number generator, updated with each call
 * @param [in]  options
 *                           Opaque pointer defining the sampling options.
 * @param [in]  do_expensive_check
 *                           A flag to run expensive checks for input arguments (if set to true)
 * @param [in]  result       Output from the uniform_neighbor_sample call
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_uniform_neighbor_sample(
    const hipgraph_resource_handle_t*               handle,
    hipgraph_graph_t*                               graph,
    const hipgraph_type_erased_device_array_view_t* start_vertices,
    const hipgraph_type_erased_device_array_view_t* start_vertex_labels,
    const hipgraph_type_erased_device_array_view_t* label_list,
    const hipgraph_type_erased_device_array_view_t* label_to_comm_rank,
    const hipgraph_type_erased_device_array_view_t* label_offsets,
    const hipgraph_type_erased_host_array_view_t*   fan_out,
    hipgraph_rng_state_t*                           rng_state,
    const hipgraph_sampling_options_t*              options,
    hipgraph_bool_t                                 do_expensive_check,
    hipgraph_sample_result_t**                      result,
    hipgraph_error_t**                              error);

/**
 * @deprecated This call should be replaced with hipgraph_sample_result_get_majors
 * @brief     Get the source vertices from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the source vertices in device memory
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_sources(const hipgraph_sample_result_t* result);

/**
 * @deprecated This call should be replaced with hipgraph_sample_result_get_minors
 * @brief     Get the destination vertices from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the destination vertices in device memory
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_destinations(const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the major vertices from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the major vertices in device memory
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_majors(const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the minor vertices from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the minor vertices in device memory
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_minors(const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the major offsets from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the major offsets in device memory
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_major_offsets(const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the start labels from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the start labels
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_start_labels(const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the edge_id from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the edge_id
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_edge_id(const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the edge_type from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the edge_type
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_edge_type(const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the edge_weight from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the edge_weight
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_edge_weight(const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the hop from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the hop
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_hop(const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the label-hop offsets from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the label-hop offsets
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_label_hop_offsets(const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the index from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the index
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_index(const hipgraph_sample_result_t* result);

/**
 * @deprecated This call should be replaced with hipgraph_sample_get_get_label_hop_offsets
 * @brief     Get the result offsets from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the result offsets
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_offsets(const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the renumber map
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the renumber map
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_renumber_map(const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the renumber map offsets
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the renumber map offsets
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_renumber_map_offsets(const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Free a sampling result
 *
 * @param [in]   result   The result from a sampling algorithm
 */
HIPGRAPH_EXPORT void hipgraph_sample_result_free(hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Create a sampling result (testing API)
 *
 * @param [in]   handle         Handle for accessing resources
 * @param [in]   srcs           Device array view to populate srcs
 * @param [in]   dsts           Device array view to populate dsts
 * @param [in]   edge_id        Device array view to populate edge_id (can be NULL)
 * @param [in]   edge_type      Device array view to populate edge_type (can be NULL)
 * @param [in]   wgt            Device array view to populate wgt (can be NULL)
 * @param [in]   hop            Device array view to populate hop
 * @param [in]   label          Device array view to populate label (can be NULL)
 * @param [out]  result         Pointer to the location to store the
 *                              hipgraph_sample_result_t*
 * @param [out]  error          Pointer to an error object storing details of
 *                              any error.  Will be populated if error code is
 *                              not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_test_sample_result_create(const hipgraph_resource_handle_t*               handle,
                                       const hipgraph_type_erased_device_array_view_t* srcs,
                                       const hipgraph_type_erased_device_array_view_t* dsts,
                                       const hipgraph_type_erased_device_array_view_t* edge_id,
                                       const hipgraph_type_erased_device_array_view_t* edge_type,
                                       const hipgraph_type_erased_device_array_view_t* wgt,
                                       const hipgraph_type_erased_device_array_view_t* hop,
                                       const hipgraph_type_erased_device_array_view_t* label,
                                       hipgraph_sample_result_t**                      result,
                                       hipgraph_error_t**                              error);

/**
 * @ingroup samplingC
 * @brief     Create a sampling result (testing API)
 *
 * @param [in]   handle         Handle for accessing resources
 * @param [in]   srcs           Device array view to populate srcs
 * @param [in]   dsts           Device array view to populate dsts
 * @param [in]   edge_id        Device array view to populate edge_id
 * @param [in]   edge_type      Device array view to populate edge_type
 * @param [in]   weight         Device array view to populate weight
 * @param [in]   hop            Device array view to populate hop
 * @param [in]   label          Device array view to populate label
 * @param [out]  result         Pointer to the location to store the
 *                              hipgraph_sample_result_t*
 * @param [out]  error          Pointer to an error object storing details of
 *                              any error.  Will be populated if error code is
 *                              not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_test_uniform_neighborhood_sample_result_create(
    const hipgraph_resource_handle_t*               handle,
    const hipgraph_type_erased_device_array_view_t* srcs,
    const hipgraph_type_erased_device_array_view_t* dsts,
    const hipgraph_type_erased_device_array_view_t* edge_id,
    const hipgraph_type_erased_device_array_view_t* edge_type,
    const hipgraph_type_erased_device_array_view_t* weight,
    const hipgraph_type_erased_device_array_view_t* hop,
    const hipgraph_type_erased_device_array_view_t* label,
    hipgraph_sample_result_t**                      result,
    hipgraph_error_t**                              error);

/**
 * @ingroup samplingC
 * @brief Select random vertices from the graph
 *
 * @param [in]      handle        Handle for accessing resources
 * @param [in]      graph         Pointer to graph
 * @param [inout]  rng_state     State of the random number generator, updated with each call
 * @param [in]      num_vertices  Number of vertices to sample
 * @param [out]     vertices      Device array view to populate label
 * @param [out]     error         Pointer to an error object storing details of
 *                                any error.  Will be populated if error code is
 *                                not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_select_random_vertices(const hipgraph_resource_handle_t*     handle,
                                    const hipgraph_graph_t*               graph,
                                    hipgraph_rng_state_t*                 rng_state,
                                    size_t                                num_vertices,
                                    hipgraph_type_erased_device_array_t** vertices,
                                    hipgraph_error_t**                    error);

#ifdef __cplusplus
}
#endif
