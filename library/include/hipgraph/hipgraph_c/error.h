// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <stdint.h>


#include "hipgraph/hipgraph-common.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef enum { 
  HIPGRAPH_SUCCESS = 0,
  HIPGRAPH_UNKNOWN_ERROR,
  HIPGRAPH_INVALID_HANDLE,
  HIPGRAPH_ALLOC_ERROR,
  HIPGRAPH_INVALID_INPUT,
  HIPGRAPH_NOT_IMPLEMENTED,
  HIPGRAPH_UNSUPPORTED_TYPE_COMBINATION 
} hipgraph_error_code_t;

typedef struct hipgraph_error hipgraph_error_t;

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
