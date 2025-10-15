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

#include "error.h"
#include "types.h"

#include <stddef.h>
#include <stdint.h>


#include "hipgraph/hipgraph-common.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct hipgraph_resource_handle hipgraph_resource_handle_t;

// FIXME: Don't really want a raft handle here.  We really want to be able to
//        configure the resource handle ourselves.  But that requires a bunch
//        of logic that's currently only available in python.
/**
 * @brief     Construct a resource handle
 *
 * @param [in]  raft_handle   Handle for accessing resources
 *                            If NULL, we will create a raft handle
 *                            internally
 *
 * @return A graph resource handle
 */
HIPGRAPH_EXPORT hipgraph_resource_handle_t* hipgraph_create_resource_handle(void* raft_handle);

/**
 * @brief get comm_size from resource handle
 *
 * If the resource handle has been configured for multi-gpu, this will return
 * the comm_size for this cluster.  If the resource handle has not been configured for
 * multi-gpu this will always return 1.
 *
 * @param [in]  handle          Handle for accessing resources
 * @return comm_size
 */
HIPGRAPH_EXPORT int hipgraph_resource_handle_get_comm_size(const hipgraph_resource_handle_t* handle);

/**
 * @brief get rank from resource handle
 *
 * If the resource handle has been configured for multi-gpu, this will return
 * the rank for this worker.  If the resource handle has not been configured for
 * multi-gpu this will always return 0.
 *
 * @param [in]  handle          Handle for accessing resources
 * @return rank
 */
HIPGRAPH_EXPORT int hipgraph_resource_handle_get_rank(const hipgraph_resource_handle_t* handle);

/**
 * @brief     Free resources in the resource handle
 *
 * @param [in]  handle          Handle for accessing resources
 */
HIPGRAPH_EXPORT void hipgraph_free_resource_handle(hipgraph_resource_handle_t* handle);

#ifdef __cplusplus
}
#endif
