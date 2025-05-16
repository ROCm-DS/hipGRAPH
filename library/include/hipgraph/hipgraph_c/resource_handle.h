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
#include "hipgraph/hipgraph_c/error.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @enum hipgraph_bool_t
 * @brief resource_handle bool enum
 */
typedef enum hipgraph_bool_
{
    HIPGRAPH_FALSE = 0,
    HIPGRAPH_TRUE  = 1
} hipgraph_bool_t;

/*
  For this and later no-nonprefixed-aliases cases, the warning
  only covers existing #defines. Other definitions will be
  shadowed silently in C. It's possible to "fix" that in C++
  with static_assert, but then we'd already have a namespace.
 */

#if !defined(HIPGRAPH_NO_NONPREFIXED_ALIASES)
#if defined(TRUE) || defined(FALSE) || defined(bool_t)
#warning \
    "cuGraph -> hipGRAPH macro aliases for TRUE, FALSE, and/or bool_t may shadow existing definitions."
#endif
/**
 * @brief resource_handle macros
 * @{
 */
#undef TRUE
#define TRUE HIPGRAPH_TRUE
#undef FALSE
#define FALSE HIPGRAPH_FALSE
#undef bool_t
#define bool_t hipgraph_bool_t
/** @} */
#endif

/** @brief hipgraph_byte_t definition */
typedef int8_t hipgraph_byte_t;

#if !defined(HIPGRAPH_NO_NONPREFIXED_ALIASES)
#if defined(byte_t)
#warning "cuGraph -> hipGRAPH macro alias byte_t may shadow an existing definition."
#endif
#undef byte_t
/** @brief hipgraph_byte_t definition */
#define byte_t hipgraph_byte_t
#endif

/**
 * @deprecated - use hipgraph_data_type_id_t;
 */
typedef enum hipgraph_data_type_id_
{
    HIPGRAPH_INT32 = 0,
    HIPGRAPH_INT64,
    HIPGRAPH_FLOAT32,
    HIPGRAPH_FLOAT64,
    HIPGRAPH_SIZE_T,
    HIPGRAPH_NTYPES
} hipgraph_data_type_id_t;

#if !defined(HIPGRAPH_NO_NONPREFIXED_ALIASES)
#if defined(INT32) || defined(INT64) || defined(FLOAT32) || defined(FLOAT64) || defined(SIZE_T) \
    || defined(NTYPES) || defined(data_type_id_t)
#warning \
    "cuGraph -> hipGRAPH macro aliases related to data_type_id_t may shadow existing definitions."
#endif
/**
 * @brief hipgraph data type macros
 * @{
 */
#undef INT32
#define INT32 HIPGRAPH_INT32
#undef INT64
#define INT64 HIPGRAPH_INT64
#undef FLOAT32
#define FLOAT32 HIPGRAPH_FLOAT32
#undef FLOAT64
#define FLOAT64 HIPGRAPH_FLOAT64
#undef SIZE_T
#define SIZE_T HIPGRAPH_SIZE_T
#undef NTYPES
#define NTYPES HIPGRAPH_NTYPES
#undef data_type_id_t
#define data_type_id_t hipgraph_data_type_id_t
/** @} */
#endif

typedef struct hipgraph_resource_handle_
{
    int32_t align_;
} hipgraph_resource_handle_t;

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
HIPGRAPH_EXPORT int
    hipgraph_resource_handle_get_comm_size(const hipgraph_resource_handle_t* handle);

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
