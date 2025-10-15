// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include "coo.h"
#include "error.h"
#include "graph.h"
#include "properties.h"
#include "random.h"
#include "resource_handle.h"


#include "hipgraph/hipgraph-common.h"
/** @defgroup samplingC Sampling algorithms
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief     Opaque random walk result type
 */
typedef struct hipgraph_random_walk_result hipgraph_random_walk_result_t;

/**
 * @brief  Compute uniform random walks
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in,out] rng_state    State of the random number generator, updated with each call
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  start_vertices  Array of source vertices
 * @param [in]  max_length      Maximum length of the generated path
 * @param [out]  result         Output from the node2vec call
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_uniform_random_walks(
  const hipgraph_resource_handle_t* handle,
  hipgraph_rng_state_t* rng_state,
  hipgraph_graph_t* graph,
  const hipgraph_type_erased_device_array_view_t* start_vertices,
  size_t max_length,
  hipgraph_random_walk_result_t** result,
  hipgraph_error_t** error);

/**
 * @brief  Compute biased random walks
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in,out] rng_state    State of the random number generator, updated with each call
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  start_vertices  Array of source vertices
 * @param [in]  max_length      Maximum length of the generated path
 * @param [out]  result         Output from the node2vec call
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_biased_random_walks(
  const hipgraph_resource_handle_t* handle,
  hipgraph_rng_state_t* rng_state,
  hipgraph_graph_t* graph,
  const hipgraph_type_erased_device_array_view_t* start_vertices,
  size_t max_length,
  hipgraph_random_walk_result_t** result,
  hipgraph_error_t** error);

/**
 * @brief  Compute random walks using the node2vec framework.
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in,out] rng_state    State of the random number generator, updated with each call
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  start_vertices  Array of source vertices
 * @param [in]  max_length      Maximum length of the generated path
 * @param [in]  compress_result If true, return the paths as a compressed sparse row matrix,
 *                              otherwise return as a dense matrix
 * @param [in]  p               The return parameter
 * @param [in]  q               The in/out parameter
 * @param [out]  result         Output from the node2vec call
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_node2vec_random_walks(
  const hipgraph_resource_handle_t* handle,
  hipgraph_rng_state_t* rng_state,
  hipgraph_graph_t* graph,
  const hipgraph_type_erased_device_array_view_t* start_vertices,
  size_t max_length,
  double p,
  double q,
  hipgraph_random_walk_result_t** result,
  hipgraph_error_t** error);

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
 * @param [in]  q            The in/out parameter
 * @param [out]  result      Output from the node2vec call
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_node2vec(const hipgraph_resource_handle_t* handle,
                                      hipgraph_graph_t* graph,
                                      const hipgraph_type_erased_device_array_view_t* sources,
                                      size_t max_depth,
                                      bool compress_result,
                                      double p,
                                      double q,
                                      hipgraph_random_walk_result_t** result,
                                      hipgraph_error_t** error);

/**
 * @ingroup samplingC
 * @brief     Get the max path length from random walk result
 *
 * @param [in]   result   The result from random walks
 * @return maximum path length
 */
HIPGRAPH_EXPORT size_t hipgraph_random_walk_result_get_max_path_length(hipgraph_random_walk_result_t* result);

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
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_random_walk_result_get_paths(
  hipgraph_random_walk_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the matrix (row major order) of edge weights in the paths
 *
 * @param [in]   result   The result from a random walk algorithm
 * @return type erased array pointing to the path edge weights in device memory
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_random_walk_result_get_weights(
  hipgraph_random_walk_result_t* result);

/**
 * @ingroup samplingC
 * @brief     If the random walk result is compressed, get the path sizes
 * @deprecated This call will no longer be relevant once the new node2vec are called
 *
 * @param [in]   result   The result from a random walk algorithm
 * @return type erased array pointing to the path sizes in device memory
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_random_walk_result_get_path_sizes(
  hipgraph_random_walk_result_t* result);

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
typedef struct hipgraph_sample_result hipgraph_sample_result_t;

/**
 * @brief     Opaque sampling options type
 */
typedef struct hipgraph_sampling_options hipgraph_sampling_options_t;




/**
 * @brief     Enumeration for prior sources behavior
 */
typedef enum { 
  HIPGRAPH_DEFAULT = 0, /** Construct sources for hop k from destination vertices from hop k-1 */
  HIPGRAPH_CARRY_OVER,  /** Construct sources for hop k from destination vertices from hop k-1
                   and sources from hop k-1 */
  HIPGRAPH_EXCLUDE      /** Construct sources for hop k from destination vertices form hop k-1,
                   but exclude any vertex that has already been used as a source */ 
} hipgraph_prior_sources_behavior_t;

#  if defined(DEFAULT)
#    warning "Not re-defining macro DEFAULT"
#  else
#    define DEFAULT HIPGRAPH_DEFAULT
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(CARRY_OVER)
#    warning "Not re-defining macro CARRY_OVER"
#  else
#    define CARRY_OVER HIPGRAPH_CARRY_OVER
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(EXCLUDE)
#    warning "Not re-defining macro EXCLUDE"
#  else
#    define EXCLUDE HIPGRAPH_EXCLUDE
#  endif
#endif#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(DEFAULT)
#    warning "Not re-defining macro DEFAULT"
#  else
#    define DEFAULT HIPGRAPH_DEFAULT
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(CARRY_OVER)
#    warning "Not re-defining macro CARRY_OVER"
#  else
#    define CARRY_OVER HIPGRAPH_CARRY_OVER
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(EXCLUDE)
#    warning "Not re-defining macro EXCLUDE"
#  else
#    define EXCLUDE HIPGRAPH_EXCLUDE
#  endif
#endif

/**
 * @brief Selects the type of compression to use for the output samples.
 */
typedef enum { 
  HIPGRAPH_COO = 0, /** Outputs in COO format.  Default. */
  HIPGRAPH_CSR,     /** Compresses in CSR format.  This means the row (src) column
               is compressed into a row pointer. */
  HIPGRAPH_CSC,     /** Compresses in CSC format.  This means the col (dst) column
               is compressed into a column pointer. */
  HIPGRAPH_DCSR,    /** Compresses in DCSR format.  This outputs an additional index
              that avoids empty entries in the row pointer. */
  HIPGRAPH_DCSC     /** Compresses in DCSC format.  This outputs an additional index
               that avoid empty entries in the col pointer. */ 
} hipgraph_compression_type_t;

#  if defined(COO)
#    warning "Not re-defining macro COO"
#  else
#    define COO HIPGRAPH_COO
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(CSR)
#    warning "Not re-defining macro CSR"
#  else
#    define CSR HIPGRAPH_CSR
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(CSC)
#    warning "Not re-defining macro CSC"
#  else
#    define CSC HIPGRAPH_CSC
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(DCSR)
#    warning "Not re-defining macro DCSR"
#  else
#    define DCSR HIPGRAPH_DCSR
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(DCSC)
#    warning "Not re-defining macro DCSC"
#  else
#    define DCSC HIPGRAPH_DCSC
#  endif
#endif#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(COO)
#    warning "Not re-defining macro COO"
#  else
#    define COO HIPGRAPH_COO
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(CSR)
#    warning "Not re-defining macro CSR"
#  else
#    define CSR HIPGRAPH_CSR
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(CSC)
#    warning "Not re-defining macro CSC"
#  else
#    define CSC HIPGRAPH_CSC
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(DCSR)
#    warning "Not re-defining macro DCSR"
#  else
#    define DCSR HIPGRAPH_DCSR
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(DCSC)
#    warning "Not re-defining macro DCSC"
#  else
#    define DCSC HIPGRAPH_DCSC
#  endif
#endif

/**
 * @ingroup samplingC
 * @brief   Create sampling options object
 *
 * All sampling options set to false
 *
 * @param [out] options Opaque pointer to the sampling options
 * @param [out] error   Pointer to an error object storing details of any error.  Will
 *                      be populated if error code is not HIPGRAPH_SUCCESS
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_sampling_options_create(hipgraph_sampling_options_t** options,
                                                     hipgraph_error_t** error);

/**
 * @ingroup samplingC
 * @brief   Set flag to retain seeds (original sources)
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Boolean value to assign to the option
 */
HIPGRAPH_EXPORT void hipgraph_sampling_set_retain_seeds(hipgraph_sampling_options_t* options, bool value);

/**
 * @ingroup samplingC
 * @brief   Set flag to renumber results
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Boolean value to assign to the option
 */
HIPGRAPH_EXPORT void hipgraph_sampling_set_renumber_results(hipgraph_sampling_options_t* options, bool value);

/**
 * @ingroup samplingC
 * @brief   Set whether to compress per-hop (True) or globally (False)
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Boolean value to assign to the option
 */
HIPGRAPH_EXPORT void hipgraph_sampling_set_compress_per_hop(hipgraph_sampling_options_t* options, bool value);

/**
 * @ingroup samplingC
 * @brief   Set flag to sample with_replacement
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Boolean value to assign to the option
 */
HIPGRAPH_EXPORT void hipgraph_sampling_set_with_replacement(hipgraph_sampling_options_t* options, bool value);

/**
 * @ingroup samplingC
 * @brief   Set flag to sample return_hops
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Boolean value to assign to the option
 */
HIPGRAPH_EXPORT void hipgraph_sampling_set_return_hops(hipgraph_sampling_options_t* options, bool value);

/**
 * @ingroup samplingC
 * @brief   Set compression type
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Enum defining the compresion type
 */
HIPGRAPH_EXPORT void hipgraph_sampling_set_compression_type(hipgraph_sampling_options_t* options,
                                           hipgraph_compression_type_t value);

/**
 * @ingroup samplingC
 * @brief   Set prior sources behavior
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Enum defining prior sources behavior
 */
HIPGRAPH_EXPORT void hipgraph_sampling_set_prior_sources_behavior(hipgraph_sampling_options_t* options,
                                                 hipgraph_prior_sources_behavior_t value);

/**
 * @ingroup samplingC
 * @brief   Set flag to sample dedupe_sources prior to sampling
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Boolean value to assign to the option
 */
HIPGRAPH_EXPORT void hipgraph_sampling_set_dedupe_sources(hipgraph_sampling_options_t* options, bool value);

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
 * @deprecated  This API will be deleted, use hipgraph_homogeneous_uniform_neighbor_sample
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
 * output.  If specified then the all data from @p label_list[i] will be shuffled to rank @p.  This
 * cannot be specified unless @p start_vertex_labels is also specified
 * label_to_comm_rank[i].  If not specified then the output data will not be shuffled between ranks.
 * @param [in]  label_offsets Device array of the offsets for each label in the seed list.  This
 *                            parameter is only used with the retain_seeds option.
 * @param [in]  fan_out       Host array defining the fan out at each step in the sampling
 * algorithm. We only support fan_out values of type INT32
 * @param [in,out] rng_state State of the random number generator, updated with each call
 * @param [in]  sampling_options
 *                           Opaque pointer defining the sampling options.
 * @param [in]  do_expensive_check
 *                           A flag to run expensive checks for input arguments (if set to true)
 * @param [out]  result      Output from the uniform_neighbor_sample call
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_uniform_neighbor_sample(
  const hipgraph_resource_handle_t* handle,
  hipgraph_graph_t* graph,
  const hipgraph_type_erased_device_array_view_t* start_vertices,
  const hipgraph_type_erased_device_array_view_t* start_vertex_labels,
  const hipgraph_type_erased_device_array_view_t* label_list,
  const hipgraph_type_erased_device_array_view_t* label_to_comm_rank,
  const hipgraph_type_erased_device_array_view_t* label_offsets,
  const hipgraph_type_erased_host_array_view_t* fan_out,
  hipgraph_rng_state_t* rng_state,
  const hipgraph_sampling_options_t* options,
  bool do_expensive_check,
  hipgraph_sample_result_t** result,
  hipgraph_error_t** error);

/**
 * @brief     Biased Neighborhood Sampling
 *
 * @deprecated  This API will be deleted, use hipgraph_homogeneous_biased_neighbor_sample.
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
 * @param [in]  edge_biases  Device array of edge biases to use for sampling.  If NULL
 * use the edge weight as the bias.  NOTE: This is a placeholder for future capability, the
 * value for edge_biases should always be set to NULL at the moment.
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
 * output.  If specified then the all data from @p label_list[i] will be shuffled to rank @p.  This
 * cannot be specified unless @p start_vertex_labels is also specified
 * label_to_comm_rank[i].  If not specified then the output data will not be shuffled between ranks.
 * @param [in]  label_offsets Device array of the offsets for each label in the seed list.  This
 *                            parameter is only used with the retain_seeds option.
 * @param [in]  fan_out       Host array defining the fan out at each step in the sampling
 * algorithm. We only support fan_out values of type INT32
 * @param [in,out] rng_state State of the random number generator, updated with each call
 * @param [in]  sampling_options
 *                           Opaque pointer defining the sampling options.
 * @param [in]  do_expensive_check
 *                           A flag to run expensive checks for input arguments (if set to true)
 * @param [out]  result      Output from the uniform_neighbor_sample call
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_biased_neighbor_sample(
  const hipgraph_resource_handle_t* handle,
  hipgraph_graph_t* graph,
  const hipgraph_edge_property_view_t* edge_biases,
  const hipgraph_type_erased_device_array_view_t* start_vertices,
  const hipgraph_type_erased_device_array_view_t* start_vertex_labels,
  const hipgraph_type_erased_device_array_view_t* label_list,
  const hipgraph_type_erased_device_array_view_t* label_to_comm_rank,
  const hipgraph_type_erased_device_array_view_t* label_offsets,
  const hipgraph_type_erased_host_array_view_t* fan_out,
  hipgraph_rng_state_t* rng_state,
  const hipgraph_sampling_options_t* options,
  bool do_expensive_check,
  hipgraph_sample_result_t** result,
  hipgraph_error_t** error);

/**
 * @brief     Homogeneous Uniform Neighborhood Sampling
 *
 * Returns a sample of the neighborhood around specified start vertices and fan_out.
 * The neighborhood is sampled uniformly.
 * Optionally, each start vertex can be associated with a label, allowing the caller to specify
 * multiple batches of sampling requests in the same function call - which should improve GPU
 * utilization.
 *
 * If label is NULL then all start vertices will be considered part of the same batch and the
 * return value will not have a label column.
 *
 * @param [in]  handle       Handle for accessing resources
 *  * @param [in,out] rng_state State of the random number generator, updated with each call
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [in]  start_vertices Device array of start vertices for the sampling
 * @param [in]  starting_vertex_label_offsets Device array of the offsets for each label in
 * the seed list. This parameter is only used with the retain_seeds option.
 * @param [in]  fan_out       Host array defining the fan out at each step in the sampling
 * algorithm. We only support fan_out values of type INT32
 * @param [in]  sampling_options
 *                           Opaque pointer defining the sampling options.
 * @param [in]  do_expensive_check
 *                           A flag to run expensive checks for input arguments (if set to true)
 * @param [out]  result      Output from the uniform_neighbor_sample call
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_homogeneous_uniform_neighbor_sample(
  const hipgraph_resource_handle_t* handle,
  hipgraph_rng_state_t* rng_state,
  hipgraph_graph_t* graph,
  const hipgraph_type_erased_device_array_view_t* start_vertices,
  const hipgraph_type_erased_device_array_view_t* starting_vertex_label_offsets,
  const hipgraph_type_erased_host_array_view_t* fan_out,
  const hipgraph_sampling_options_t* options,
  bool do_expensive_check,
  hipgraph_sample_result_t** result,
  hipgraph_error_t** error);

/**
 * @brief     Homogeneous Biased Neighborhood Sampling
 *
 * Returns a sample of the neighborhood around specified start vertices and fan_out.
 * The neighborhood is sampled uniformly.
 * Optionally, each start vertex can be associated with a label, allowing the caller to specify
 * multiple batches of sampling requests in the same function call - which should improve GPU
 * utilization.
 *
 * If label is NULL then all start vertices will be considered part of the same batch and the
 * return value will not have a label column.
 *
 * @param [in]  handle       Handle for accessing resources
 *  * @param [in,out] rng_state State of the random number generator, updated with each call
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [in]  edge_biases  Device array of edge biases to use for sampling.  If NULL
 * use the edge weight as the bias. If set to NULL, edges will be sampled uniformly.
 * @param [in]  start_vertices Device array of start vertices for the sampling
 * @param [in]  starting_vertex_label_offsets Device array of the offsets for each label in
 * the seed list. This parameter is only used with the retain_seeds option.
 * @param [in]  fan_out       Host array defining the fan out at each step in the sampling
 * algorithm. We only support fan_out values of type INT32
 * @param [in]  sampling_options
 *                           Opaque pointer defining the sampling options.
 * @param [in]  do_expensive_check
 *                           A flag to run expensive checks for input arguments (if set to true)
 * @param [out]  result      Output from the uniform_neighbor_sample call
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_homogeneous_biased_neighbor_sample(
  const hipgraph_resource_handle_t* handle,
  hipgraph_rng_state_t* rng_state,
  hipgraph_graph_t* graph,
  const hipgraph_edge_property_view_t* edge_biases,
  const hipgraph_type_erased_device_array_view_t* start_vertices,
  const hipgraph_type_erased_device_array_view_t* starting_vertex_label_offsets,
  const hipgraph_type_erased_host_array_view_t* fan_out,
  const hipgraph_sampling_options_t* options,
  bool do_expensive_check,
  hipgraph_sample_result_t** result,
  hipgraph_error_t** error);

/**
 * @brief     Heterogeneous Uniform Neighborhood Sampling
 *
 * Returns a sample of the neighborhood around specified start vertices and fan_out.
 * The neighborhood is sampled uniformly.
 * Optionally, each start vertex can be associated with a label, allowing the caller to specify
 * multiple batches of sampling requests in the same function call - which should improve GPU
 * utilization.
 *
 * If label is NULL then all start vertices will be considered part of the same batch and the
 * return value will not have a label column.
 *
 * @param [in]  handle       Handle for accessing resources
 *  * @param [in,out] rng_state State of the random number generator, updated with each call
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [in]  start_vertices Device array of start vertices for the sampling
 * @param [in]  starting_vertex_label_offsets Device array of the offsets for each label in
 * the seed list. This parameter is only used with the retain_seeds option.
 * @param [in]  vertex_type_offsets Device array of the offsets for each vertex type in the
 * graph.
 * @param [in]  fan_out       Host array defining the fan out at each step in the sampling
 * algorithm. We only support fan_out values of type INT32
 * @param [in]  num_edge_types Number of edge types where a value of 1 translates to homogeneous
 * neighbor sample whereas a value greater than 1 translates to heterogeneous neighbor sample.
 * @param [in]  sampling_options
 *                           Opaque pointer defining the sampling options.
 * @param [in]  do_expensive_check
 *                           A flag to run expensive checks for input arguments (if set to true)
 * @param [out]  result      Output from the uniform_neighbor_sample call
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_heterogeneous_uniform_neighbor_sample(
  const hipgraph_resource_handle_t* handle,
  hipgraph_rng_state_t* rng_state,
  hipgraph_graph_t* graph,
  const hipgraph_type_erased_device_array_view_t* start_vertices,
  const hipgraph_type_erased_device_array_view_t* starting_vertex_label_offsets,
  const hipgraph_type_erased_device_array_view_t* vertex_type_offsets,
  const hipgraph_type_erased_host_array_view_t* fan_out,
  int num_edge_types,
  const hipgraph_sampling_options_t* options,
  bool do_expensive_check,
  hipgraph_sample_result_t** result,
  hipgraph_error_t** error);

/**
 * @brief     Heterogeneous Biased Neighborhood Sampling
 *
 * Returns a sample of the neighborhood around specified start vertices and fan_out.
 * The neighborhood is sampled uniformly.
 * Optionally, each start vertex can be associated with a label, allowing the caller to specify
 * multiple batches of sampling requests in the same function call - which should improve GPU
 * utilization.
 *
 * If label is NULL then all start vertices will be considered part of the same batch and the
 * return value will not have a label column.
 *
 * @param [in]  handle       Handle for accessing resources
 *  * @param [in,out] rng_state State of the random number generator, updated with each call
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [in]  edge_biases  Device array of edge biases to use for sampling.  If NULL
 * use the edge weight as the bias. If set to NULL, edges will be sampled uniformly.
 * @param [in]  start_vertices Device array of start vertices for the sampling
 * @param [in]  starting_vertex_label_offsets Device array of the offsets for each label in
 * the seed list. This parameter is only used with the retain_seeds option.
 * @param [in]  vertex_type_offsets Device array of the offsets for each vertex type in the
 * graph.
 * @param [in]  fan_out       Host array defining the fan out at each step in the sampling
 * algorithm. We only support fan_out values of type INT32
 * @param [in]  num_edge_types Number of edge types where a value of 1 translates to homogeneous
 * neighbor sample whereas a value greater than 1 translates to heterogeneous neighbor sample.
 * @param [in]  sampling_options
 *                           Opaque pointer defining the sampling options.
 * @param [in]  do_expensive_check
 *                           A flag to run expensive checks for input arguments (if set to true)
 * @param [out]  result      Output from the uniform_neighbor_sample call
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_heterogeneous_biased_neighbor_sample(
  const hipgraph_resource_handle_t* handle,
  hipgraph_rng_state_t* rng_state,
  hipgraph_graph_t* graph,
  const hipgraph_edge_property_view_t* edge_biases,
  const hipgraph_type_erased_device_array_view_t* start_vertices,
  const hipgraph_type_erased_device_array_view_t* starting_vertex_label_offsets,
  const hipgraph_type_erased_device_array_view_t* vertex_type_offsets,
  const hipgraph_type_erased_host_array_view_t* fan_out,
  int num_edge_types,
  const hipgraph_sampling_options_t* options,
  bool do_expensive_check,
  hipgraph_sample_result_t** result,
  hipgraph_error_t** error);

/**
 * @deprecated This call should be replaced with hipgraph_sample_result_get_majors
 * @brief     Get the source vertices from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the source vertices in device memory
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_sources(
  const hipgraph_sample_result_t* result);

/**
 * @deprecated This call should be replaced with hipgraph_sample_result_get_minors
 * @brief     Get the destination vertices from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the destination vertices in device memory
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_destinations(
  const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the major vertices from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the major vertices in device memory
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_majors(
  const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the minor vertices from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the minor vertices in device memory
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_minors(
  const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the major offsets from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the major offsets in device memory
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_major_offsets(
  const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the start labels from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the start labels
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_start_labels(
  const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the edge_id from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the edge_id
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_edge_id(
  const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the edge_type from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the edge_type
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_edge_type(
  const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the edge_weight from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the edge_weight
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_edge_weight(
  const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the hop from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the hop
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_hop(
  const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the label-hop offsets from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the label-hop offsets
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_label_hop_offsets(
  const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the label-type-hop offsets from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the label-type-hop offsets
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_label_type_hop_offsets(
  const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the index from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the index
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_index(
  const hipgraph_sample_result_t* result);

/**
 * @deprecated This call should be replaced with hipgraph_sample_get_get_label_hop_offsets
 * @brief     Get the result offsets from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the result offsets
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_offsets(
  const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the renumber map
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the renumber map
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_renumber_map(
  const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the renumber map offsets
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the renumber map offsets
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_renumber_map_offsets(
  const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the edge renumber map
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the renumber map
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_edge_renumber_map(
  const hipgraph_sample_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Get the edge renumber map offets
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the renumber map
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_edge_renumber_map_offsets(
  const hipgraph_sample_result_t* result);

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
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_test_sample_result_create(
  const hipgraph_resource_handle_t* handle,
  const hipgraph_type_erased_device_array_view_t* srcs,
  const hipgraph_type_erased_device_array_view_t* dsts,
  const hipgraph_type_erased_device_array_view_t* edge_id,
  const hipgraph_type_erased_device_array_view_t* edge_type,
  const hipgraph_type_erased_device_array_view_t* wgt,
  const hipgraph_type_erased_device_array_view_t* hop,
  const hipgraph_type_erased_device_array_view_t* label,
  hipgraph_sample_result_t** result,
  hipgraph_error_t** error);

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
  const hipgraph_resource_handle_t* handle,
  const hipgraph_type_erased_device_array_view_t* srcs,
  const hipgraph_type_erased_device_array_view_t* dsts,
  const hipgraph_type_erased_device_array_view_t* edge_id,
  const hipgraph_type_erased_device_array_view_t* edge_type,
  const hipgraph_type_erased_device_array_view_t* weight,
  const hipgraph_type_erased_device_array_view_t* hop,
  const hipgraph_type_erased_device_array_view_t* label,
  hipgraph_sample_result_t** result,
  hipgraph_error_t** error);

/**
 * @ingroup samplingC
 * @brief Select random vertices from the graph
 *
 * @param [in]      handle        Handle for accessing resources
 * @param [in]      graph         Pointer to graph
 * @param [in,out]  rng_state     State of the random number generator, updated with each call
 * @param [in]      num_vertices  Number of vertices to sample
 * @param [out]     vertices      Device array view to populate label
 * @param [out]     error         Pointer to an error object storing details of
 *                                any error.  Will be populated if error code is
 *                                not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_select_random_vertices(const hipgraph_resource_handle_t* handle,
                                                    const hipgraph_graph_t* graph,
                                                    hipgraph_rng_state_t* rng_state,
                                                    size_t num_vertices,
                                                    hipgraph_type_erased_device_array_t** vertices,
                                                    hipgraph_error_t** error);

/**
 * @ingroup samplingC
 * @brief Perform negative sampling
 *
 * Negative sampling generates a COO structure defining edges according to the specified parameters
 *
 * @param [in]     handle                  Handle for accessing resources
 * @param [in,out] rng_state               State of the random number generator, updated with each
 *                                         call
 * @param [in]     graph                   Pointer to graph
 * @param [in]     vertices                Vertex ids for the source biases.  If @p src_bias and
 *                                         @p dst_bias are not specified this is ignored.  If
 *                                         @p vertices is specified then vertices[i] is the vertex
 *                                         id of src_biases[i] and dst_biases[i].  If @p vertices
 *                                         is not specified then i is the vertex id if src_biases[i]
 *                                         and dst_biases[i]
 * @param [in]     src_biases              Bias for selecting source vertices.  If NULL, do uniform
 *                                         sampling, if provided probability of vertex i will be
 *                                         src_bias[i] / (sum of all source biases)
 * @param [in]     dst_biases              Bias for selecting destination vertices.  If NULL, do
 *                                         uniform sampling, if provided probability of vertex i
 *                                         will be dst_bias[i] / (sum of all destination biases)
 * @param [in]     num_samples             Number of negative samples to generate
 * @param [in]     remove_duplicates       If true, remove duplicates from sampled edges
 * @param [in]     remove_existing_edges   If true, remove sampled edges that actually exist in
 *                                         the graph
 * @param [in]     exact_number_of_samples If true, result should contain exactly @p num_samples. If
 *                                         false the code will generate @p num_samples and then do
 *                                         any filtering as specified
 * @param [in]     do_expensive_check      A flag to run expensive checks for input arguments (if
 *                                         set to true)
 * @param [out]    result                  Opaque pointer to generated coo list
 * @param [out]    error                   Pointer to an error object storing details of any error.
 *                                         Will be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_negative_sampling(
  const hipgraph_resource_handle_t* handle,
  hipgraph_rng_state_t* rng_state,
  hipgraph_graph_t* graph,
  const hipgraph_type_erased_device_array_view_t* vertices,
  const hipgraph_type_erased_device_array_view_t* src_biases,
  const hipgraph_type_erased_device_array_view_t* dst_biases,
  size_t num_samples,
  bool remove_duplicates,
  bool remove_existing_edges,
  bool exact_number_of_samples,
  bool do_expensive_check,
  hipgraph_coo_t** result,
  hipgraph_error_t** error);

#ifdef __cplusplus
}
#endif
