// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include "random.h"
#include "resource_handle.h"


#include "hipgraph/hipgraph-common.h"
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief       Opaque COO definition
 */
typedef struct hipgraph_coo hipgraph_coo_t;

/**
 * @brief       Opaque COO list definition
 */
typedef struct hipgraph_coo_list hipgraph_coo_list_t;

/**
 * @brief       Get the source vertex ids
 *
 * @param [in]     coo   Opaque pointer to COO
 * @return type erased array view of source vertex ids
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_coo_get_sources(hipgraph_coo_t* coo);

/**
 * @brief       Get the destination vertex ids
 *
 * @param [in]     coo   Opaque pointer to COO
 * @return type erased array view of destination vertex ids
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_coo_get_destinations(hipgraph_coo_t* coo);

/**
 * @brief       Get the edge weights
 *
 * @param [in]     coo   Opaque pointer to COO
 * @return type erased array view of edge weights, NULL if no edge weights in COO
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_coo_get_edge_weights(hipgraph_coo_t* coo);

/**
 * @brief       Get the edge id
 *
 * @param [in]     coo   Opaque pointer to COO
 * @return type erased array view of edge id, NULL if no edge ids in COO
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_coo_get_edge_id(hipgraph_coo_t* coo);

/**
 * @brief       Get the edge type
 *
 * @param [in]     coo   Opaque pointer to COO
 * @return type erased array view of edge type, NULL if no edge types in COO
 */
HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_coo_get_edge_type(hipgraph_coo_t* coo);

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
HIPGRAPH_EXPORT hipgraph_coo_t* hipgraph_coo_list_element(hipgraph_coo_list_t* coo_list, size_t index);

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

#ifdef __cplusplus
}
#endif
