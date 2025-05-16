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
#include "hipgraph/hipgraph_c/resource_handle.h"

/** @defgroup core Core algorithms
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief       Opaque core number result type
 */
typedef struct
{
    /** @brief align_ result type */
    int32_t align_;
} hipgraph_core_result_t;

/**
 * @brief       Opaque k-core result type
 */
typedef struct
{
    /** @brief align_ result type */
    int32_t align_;
} hipgraph_k_core_result_t;

/**
 * @ingroup core
 * @brief       Create a core_number result (in case it was previously extracted)
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  vertices     The result from core number
 * @param [in]  core_numbers The result from core number
 * @param [out] core_result       Opaque pointer to core number results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_core_result_create(const hipgraph_resource_handle_t*         handle,
                                hipgraph_type_erased_device_array_view_t* vertices,
                                hipgraph_type_erased_device_array_view_t* core_numbers,
                                hipgraph_core_result_t**                  core_result,
                                hipgraph_error_t**                        error);

/**
 * @ingroup core
 * @brief       Get the vertex ids from the core result
 *
 * @param [in]     result   The result from core number
 * @return type erased array of vertex ids
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_core_result_get_vertices(hipgraph_core_result_t* result);

/**
 * @ingroup core
 * @brief       Get the core numbers from the core result
 *
 * @param [in]    result    The result from core number
 * @return type erased array of core numbers
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_core_result_get_core_numbers(hipgraph_core_result_t* result);

/**
 * @ingroup core
 * @brief     Free core result
 *
 * @param [in]    result    The result from core number
 */
HIPGRAPH_EXPORT void hipgraph_core_result_free(hipgraph_core_result_t* result);

/**
 * @ingroup core
 * @brief       Get the src vertex ids from the k-core result
 *
 * @param [in]     result   The result from k-core
 * @return type erased array of src vertex ids
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_k_core_result_get_src_vertices(hipgraph_k_core_result_t* result);

/**
 * @ingroup core
 * @brief       Get the dst vertex ids from the k-core result
 *
 * @param [in]     result   The result from k-core
 * @return type erased array of dst vertex ids
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_k_core_result_get_dst_vertices(hipgraph_k_core_result_t* result);

/**
 * @ingroup core
 * @brief       Get the weights from the k-core result
 *
 * Returns NULL if the graph is unweighted
 *
 * @param [in]     result   The result from k-core
 * @return type erased array of weights
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t*
    hipgraph_k_core_result_get_weights(hipgraph_k_core_result_t* result);

/**
 * @ingroup core
 * @brief     Free k-core result
 *
 * @param [in]    result    The result from k-core
 */
HIPGRAPH_EXPORT void hipgraph_k_core_result_free(hipgraph_k_core_result_t* result);

/**
 * @ingroup core
 * @brief     Enumeration for computing core number
 */
typedef enum hipgraph_k_core_degree_type_
{
    HIPGRAPH_K_CORE_DEGREE_TYPE_IN  = 0, /** Compute core_number using incoming edges */
    HIPGRAPH_K_CORE_DEGREE_TYPE_OUT = 1, /** Compute core_number using outgoing edges */
    HIPGRAPH_K_CORE_DEGREE_TYPE_INOUT
    = 2 /** Compute core_number using both incoming and outgoing edges */
} hipgraph_k_core_degree_type_t;

/* See resource_handle.h for a caveat about the warning. */

#if !defined(HIPGRAPH_NO_NONPREFIXED_ALIASES)
#if defined(K_CORE_DEGREE_TYPE_IN) || defined(K_CORE_DEGREE_TYPE_OUT) \
    || defined(K_CORE_DEGREE_TYPE_INOUT) || defined(k_core_degree_type_t)
#warning \
    "cuGraph -> hipGRAPH macro aliases related to k_core_degree_type_t may shadow existing definitions."
#endif
/**
 * @brief K_CORE macros
 *
 * @{
 */

#undef K_CORE_DEGREE_TYPE_IN
#define K_CORE_DEGREE_TYPE_IN HIPGRAPH_K_CORE_DEGREE_TYPE_IN
#undef K_CORE_DEGREE_TYPE_OUT
#define K_CORE_DEGREE_TYPE_OUT HIPGRAPH_K_CORE_DEGREE_TYPE_OUT
#undef K_CORE_DEGREE_TYPE_INOUT
#define K_CORE_DEGREE_TYPE_INOUT HIPGRAPH_K_CORE_DEGREE_TYPE_INOUT
#undef k_core_degree_type_t
#define k_core_degree_type_t hipgraph_k_core_degree_type_t
#endif
/** @} */

/**
 * @brief     Perform core number.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * @param [in]  degree_type  Compute core_number using in, out or both in and out edges
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result       Opaque pointer to core number results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_core_number(const hipgraph_resource_handle_t* handle,
                         hipgraph_graph_t*                 graph,
                         hipgraph_k_core_degree_type_t     degree_type,
                         hipgraph_bool_t                   do_expensive_check,
                         hipgraph_core_result_t**          result,
                         hipgraph_error_t**                error);

/**
 * @brief     Perform k_core using output from core_number
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * @param [in]  k            The value of k to use
 * @param [in]  degree_type  Compute core_number using in, out or both in and out edges.
 *                           Ignored if core_result is specified.
 * @param [in]  core_result  Result from calling hipgraph_core_number, if NULL then
 *                           call core_number inside this function call.
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result       Opaque pointer to k_core results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_k_core(const hipgraph_resource_handle_t* handle,
                                                      hipgraph_graph_t*                 graph,
                                                      size_t                            k,
                                                      hipgraph_k_core_degree_type_t     degree_type,
                                                      const hipgraph_core_result_t*     core_result,
                                                      hipgraph_bool_t            do_expensive_check,
                                                      hipgraph_k_core_result_t** result,
                                                      hipgraph_error_t**         error);

#ifdef __cplusplus
}
#endif
