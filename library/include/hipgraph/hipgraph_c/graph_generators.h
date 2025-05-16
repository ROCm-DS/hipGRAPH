// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*! \file */
/* ************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include "hipgraph/hipgraph_c/graph.h"
#include "hipgraph/hipgraph_c/random.h"
#include "hipgraph/hipgraph_c/resource_handle.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @enum hipgraph_generator_distribution_t
 * @brief generator enum
 */
typedef enum hipgraph_generator_distribution_
{
    HIPGRAPH_POWER_LAW = 0,
    HIPGRAPH_UNIFORM
} hipgraph_generator_distribution_t;

/* See resource_handle.h for a caveat about the warning. */

#if !defined(HIPGRAPH_NO_NONPREFIXED_ALIASES)
#if defined(POWER_LAW) || defined(UNIFORM) || defined(generator_distribution_t)
#warning \
    "cuGraph -> hipGRAPH macro aliases related to generator_distribution_t may shadow existing definitions."
#endif
/**
 * @brief Generator macros
 * @{
 */
#undef POWER_LAW
#define POWER_LAW HIPGRAPH_POWER_LAW
#undef UNIFORM
#define UNIFORM HIPGRAPH_UNIFORM
#undef generator_distribution_t
#define generator_distribution_t hipgraph_generator_distribution_t
/** @} */
#endif

/**
 * @brief       Opaque COO definition
 */
typedef struct
{
    /** @brief align_ result type */
    int32_t align_;
} hipgraph_coo_t;

/**
 * @brief       Opaque COO list definition
 */
typedef struct
{
    /** @brief align_ result type */
    int32_t align_;
} hipgraph_coo_list_t;

/**
 * @brief       Get the source vertex ids
 *
 * @param [in]     coo   Opaque pointer to COO
 * @return type erased array view of source vertex ids
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_coo_get_sources(hipgraph_coo_t* coo);

/**
 * @brief       Get the destination vertex ids
 *
 * @param [in]     coo   Opaque pointer to COO
 * @return type erased array view of destination vertex ids
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_coo_get_destinations(hipgraph_coo_t* coo);

/**
 * @brief       Get the edge weights
 *
 * @param [in]     coo   Opaque pointer to COO
 * @return type erased array view of edge weights, NULL if no edge weights in COO
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_coo_get_edge_weights(hipgraph_coo_t* coo);

/**
 * @brief       Get the edge id
 *
 * @param [in]     coo   Opaque pointer to COO
 * @return type erased array view of edge id, NULL if no edge ids in COO
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_coo_get_edge_id(hipgraph_coo_t* coo);

/**
 * @brief       Get the edge type
 *
 * @param [in]     coo   Opaque pointer to COO
 * @return type erased array view of edge type, NULL if no edge types in COO
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_coo_get_edge_type(hipgraph_coo_t* coo);

/**
 * @brief       Get the number of coo object in the list
 *
 * @param [in]     coo_list   Opaque pointer to COO list
 * @return number of elements
 */
HIPGRAPH_EXPORT size_t hipgraph_coo_list_size(const hipgraph_coo_list_t* coo_list);

/**
 * @brief       Get a COO from the list
 *
 * @param [in]     coo_list   Opaque pointer to COO list
 * @param [in]     index      Index of desired COO from list
 * @return a hipgraph_coo_t* object from the list
 */
HIPGRAPH_EXPORT hipgraph_coo_t* hipgraph_coo_list_element(hipgraph_coo_list_t* coo_list,
                                                          size_t               index);

/**
 * @brief     Free coo object
 *
 * @param [in]    coo Opaque pointer to COO
 */
HIPGRAPH_EXPORT void hipgraph_coo_free(hipgraph_coo_t* coo);

/**
 * @brief     Free coo list
 *
 * @param [in]    coo_list Opaque pointer to list of COO objects
 */
HIPGRAPH_EXPORT void hipgraph_coo_list_free(hipgraph_coo_list_t* coo_list);

/**
 * @brief      Generate RMAT edge list
 *
 * Returns a COO containing edges generated from the RMAT generator.
 *
 * Vertex types will be int32 if scale < 32 and int64 if scale >= 32
 *
 * @param [in]     handle             Handle for accessing resources
 * @param [inout] rng_state          State of the random number generator, updated with each call
 * @param [in]     scale Scale factor to set the number of vertices in the graph. Vertex IDs have
 * values in [0, V), where V = 1 << @p scale.
 * @param [in]     num_edges          Number of edges to generate.
 * @param [in]     a                  a, b, c, d (= 1.0 - (a + b + c)) in the R-mat graph generator
 * (vist https://graph500.org for additional details). a, b, c, d should be non-negative and a + b +
 * c should be no larger than 1.0.
 * @param [in]     b                  a, b, c, d (= 1.0 - (a + b + c)) in the R-mat graph generator
 * (vist https://graph500.org for additional details). a, b, c, d should be non-negative and a + b +
 * c should be no larger than 1.0.
 * @param [in]     c                  a, b, c, d (= 1.0 - (a + b + c)) in the R-mat graph generator
 * (vist https://graph500.org for additional details). a, b, c, d should be non-negative and a + b +
 * c should be no larger than 1.0.
 * @param [in]     clip_and_flip      Flag controlling whether to generate edges only in the lower
 * triangular part (including the diagonal) of the graph adjacency matrix (if set to `true`) or not
 * (if set to `false`).
 * @param [in]     scramble_vertex_ids Flag controlling whether to scramble vertex ID bits
 * (if set to `true`) or not (if set to `false`); scrambling vertex ID bits breaks correlation
 * between vertex ID values and vertex degrees.
 * @param [out]    result             Opaque pointer to generated coo
 * @param [out]    error              Pointer to an error object storing details of any error.  Will
 *                                    be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_generate_rmat_edgelist(const hipgraph_resource_handle_t* handle,
                                    hipgraph_rng_state_t*             rng_state,
                                    size_t                            scale,
                                    size_t                            num_edges,
                                    double                            a,
                                    double                            b,
                                    double                            c,
                                    hipgraph_bool_t                   clip_and_flip,
                                    hipgraph_bool_t                   scramble_vertex_ids,
                                    hipgraph_coo_t**                  result,
                                    hipgraph_error_t**                error);

/**
 * @brief      Generate RMAT edge lists
 *
 * Returns a COO list containing edges generated from the RMAT generator.
 *
 * Vertex types will be int32 if scale < 32 and int64 if scale >= 32
 *
 * @param [in]     handle             Handle for accessing resources
 * @param [inout] rng_state          State of the random number generator, updated with each call
 * @param [in]     n_edgelists Number of edge lists (graphs) to generate
 * @param [in]     min_scale Scale factor to set the minimum number of verties in the graph.
 * @param [in]     max_scale Scale factor to set the maximum number of verties in the graph.
 * @param [in]     edge_factor Average number of edges per vertex to generate.
 * @param [in]     size_distribution Distribution of the graph sizes, impacts the scale parameter of
 * the R-MAT generator
 * @param [in]     edge_distribution Edges distribution for each graph, impacts how R-MAT parameters
 * a,b,c,d, are set.
 * @param [in]     clip_and_flip      Flag controlling whether to generate edges only in the lower
 * triangular part (including the diagonal) of the graph adjacency matrix (if set to `true`) or not
 * (if set to `false`).
 * @param [in]     scramble_vertex_ids Flag controlling whether to scramble vertex ID bits
 * (if set to `true`) or not (if set to `false`); scrambling vertex ID bits breaks correlation
 * between vertex ID values and vertex degrees.
 * @param [out]    result             Opaque pointer to generated coo list
 * @param [out]    error              Pointer to an error object storing details of any error.  Will
 *                                    be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_generate_rmat_edgelists(const hipgraph_resource_handle_t* handle,
                                     hipgraph_rng_state_t*             rng_state,
                                     size_t                            n_edgelists,
                                     size_t                            min_scale,
                                     size_t                            max_scale,
                                     size_t                            edge_factor,
                                     hipgraph_generator_distribution_t size_distribution,
                                     hipgraph_generator_distribution_t edge_distribution,
                                     hipgraph_bool_t                   clip_and_flip,
                                     hipgraph_bool_t                   scramble_vertex_ids,
                                     hipgraph_coo_list_t**             result,
                                     hipgraph_error_t**                error);

/**
 * @brief      Generate edge weights and add to an rmat edge list
 *              Updates a COO to contain random edge weights
 *
 * @param [in]     handle             Handle for accessing resources
 * @param [inout] rng_state          State of the random number generator, updated with each call
 * @param [inout] coo                Opaque pointer to the coo, weights will be added (overwriting
 * any existing weights)
 * @param [in]     dtype              The type of weight to generate (FLOAT32 or FLOAT64), ignored
 * unless include_weights is true
 * @param [in]     minimum_weight     Minimum weight value to generate
 * @param [in]     maximum_weight     Maximum weight value to generate
 * @param [out]    error              Pointer to an error object storing details of any error.  Will
 *                                    be populated if error code is not HIPGRAPH_SUCCESS
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_generate_edge_weights(const hipgraph_resource_handle_t* handle,
                                   hipgraph_rng_state_t*             rng_state,
                                   hipgraph_coo_t*                   coo,
                                   hipgraph_data_type_id_t           dtype,
                                   double                            minimum_weight,
                                   double                            maximum_weight,
                                   hipgraph_error_t**                error);

/**
 * @brief      Add edge ids to an COO
 *
 * Updates a COO to contain edge ids.  Edges will be numbered from 0 to n-1 where n is the number of
 * edges
 *
 * @param [in]     handle             Handle for accessing resources
 * @param [inout] coo                Opaque pointer to the coo, weights will be added (overwriting
 * any existing weights)
 * @param [in]     multi_gpu          Flag if the COO is being created on multiple GPUs
 * @param [out]    error              Pointer to an error object storing details of any error.  Will
 *                                    be populated if error code is not HIPGRAPH_SUCCESS
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_generate_edge_ids(const hipgraph_resource_handle_t* handle,
                               hipgraph_coo_t*                   coo,
                               hipgraph_bool_t                   multi_gpu,
                               hipgraph_error_t**                error);

/**
 * @brief      Generate random edge types, add them to an COO
 *               Updates a COO to contain edge types.  Edges types will be randomly generated.
 *
 * @param [in]     handle             Handle for accessing resources
 * @param [inout] rng_state          State of the random number generator, updated with each call
 * @param [inout] coo                Opaque pointer to the coo, weights will be added (overwriting
 * any existing weights)
 * @param [in]     min_edge_type
 * @param [in]     max_edge_type      Edge types will be randomly generated between min_edge_type
 * and max_edge_type
 * @param [out]    error              Pointer to an error object storing details of any error.  Will
 *                                    be populated if error code is not HIPGRAPH_SUCCESS
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_generate_edge_types(const hipgraph_resource_handle_t* handle,
                                 hipgraph_rng_state_t*             rng_state,
                                 hipgraph_coo_t*                   coo,
                                 int32_t                           min_edge_type,
                                 int32_t                           max_edge_type,
                                 hipgraph_error_t**                error);

#ifdef __cplusplus
}
#endif
