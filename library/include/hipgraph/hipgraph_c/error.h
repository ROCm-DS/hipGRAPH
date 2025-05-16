// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*! \file */
/* ************************************************************************
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <stdint.h>
#include "hipgraph/hipgraph-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @enum hipgraph_error_code_t
 * @brief enum of error codes
 */

typedef enum hipgraph_error_code_
{
    HIPGRAPH_SUCCESS = 0,
    HIPGRAPH_UNKNOWN_ERROR,
    HIPGRAPH_INVALID_HANDLE,
    HIPGRAPH_ALLOC_ERROR,
    HIPGRAPH_INVALID_INPUT,
    HIPGRAPH_NOT_IMPLEMENTED,
    HIPGRAPH_UNSUPPORTED_TYPE_COMBINATION
} hipgraph_error_code_t;

/* See resource_handle.h for a caveat about the warning. */

#if !defined(HIPGRAPH_NO_NONPREFIXED_ALIASES)
#if defined(SUCCESS) || defined(UNKNOWN_ERROR) || defined(INVALID_HANDLE) || defined(ALLOC_ERROR)  \
    || defined(INVALID_INPUT) || defined(NOT_IMPLEMENTED) || defined(UNSUPPORTED_TYPE_COMBINATION) \
    || defined(error_code_t)
#warning \
    "cuGraph -> hipGRAPH macro aliases related to error_code_t may shadow existing definitions."
#endif

/**
 * @brief Error macros
 * @{
 */

#undef SUCCESS
#define SUCCESS HIPGRAPH_SUCCESS
#undef UNKNOWN_ERROR
#define UNKNOWN_ERROR HIPGRAPH_UNKNOWN_ERROR
#undef INVALID_HANDLE
#define INVALID_HANDLE HIPGRAPH_INVALID_HANDLE
#undef ALLOC_ERROR
#define ALLOC_ERROR HIPGRAPH_ALLOC_ERROR
#undef INVALID_INPUT
#define INVALID_INPUT HIPGRAPH_INVALID_INPUT
#undef NOT_IMPLEMENTED
#define NOT_IMPLEMENTED HIPGRAPH_NOT_IMPLEMENTED
#undef UNSUPPORTED_TYPE_COMBINATION
#define UNSUPPORTED_TYPE_COMBINATION HIPGRAPH_UNSUPPORTED_TYPE_COMBINATION
#undef error_code_t
#define error_code_t hipgraph_error_code_t
/** @} */
#endif

typedef struct hipgraph_error_
{
    int32_t align_;
} hipgraph_error_t;

/**
 * @brief     Return an error message
 *
 * @param [in]  error       The error object from some hipgraph function call
 * @return a C-style string that provides detail for the error
 */
HIPGRAPH_EXPORT const char* hipgraph_error_message(const hipgraph_error_t* error);

/**
 * @brief    Destroy an error message
 *
 * @param [in]  error       The error object from some hipgraph function call
 */
HIPGRAPH_EXPORT void hipgraph_error_free(hipgraph_error_t* error);

#ifdef __cplusplus
}
#endif
