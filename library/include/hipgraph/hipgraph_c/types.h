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

#include <stdint.h>


#include "hipgraph/hipgraph-common.h"
#ifdef __cplusplus
extern "C" {
#endif





typedef enum { 
  HIPGRAPH_INT8 = 0,
  HIPGRAPH_INT16,
  HIPGRAPH_INT32,
  HIPGRAPH_INT64,
  HIPGRAPH_UINT8,
  HIPGRAPH_UINT16,
  HIPGRAPH_UINT32,
  HIPGRAPH_UINT64,
  HIPGRAPH_FLOAT32,
  HIPGRAPH_FLOAT64,
  HIPGRAPH_SIZE_T,
  HIPGRAPH_BOOL,
  HIPGRAPH_NTYPES 
} hipgraph_data_type_id_t;

#  if defined(INT8)
#    warning "Not re-defining macro INT8"
#  else
#    define INT8 HIPGRAPH_INT8
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(INT16)
#    warning "Not re-defining macro INT16"
#  else
#    define INT16 HIPGRAPH_INT16
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(INT32)
#    warning "Not re-defining macro INT32"
#  else
#    define INT32 HIPGRAPH_INT32
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(INT64)
#    warning "Not re-defining macro INT64"
#  else
#    define INT64 HIPGRAPH_INT64
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(UINT8)
#    warning "Not re-defining macro UINT8"
#  else
#    define UINT8 HIPGRAPH_UINT8
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(UINT16)
#    warning "Not re-defining macro UINT16"
#  else
#    define UINT16 HIPGRAPH_UINT16
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(UINT32)
#    warning "Not re-defining macro UINT32"
#  else
#    define UINT32 HIPGRAPH_UINT32
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(UINT64)
#    warning "Not re-defining macro UINT64"
#  else
#    define UINT64 HIPGRAPH_UINT64
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(FLOAT32)
#    warning "Not re-defining macro FLOAT32"
#  else
#    define FLOAT32 HIPGRAPH_FLOAT32
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(FLOAT64)
#    warning "Not re-defining macro FLOAT64"
#  else
#    define FLOAT64 HIPGRAPH_FLOAT64
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(SIZE_T)
#    warning "Not re-defining macro SIZE_T"
#  else
#    define SIZE_T HIPGRAPH_SIZE_T
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(BOOL)
#    warning "Not re-defining macro BOOL"
#  else
#    define BOOL HIPGRAPH_BOOL
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(NTYPES)
#    warning "Not re-defining macro NTYPES"
#  else
#    define NTYPES HIPGRAPH_NTYPES
#  endif
#endif#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(INT8)
#    warning "Not re-defining macro INT8"
#  else
#    define INT8 HIPGRAPH_INT8
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(INT16)
#    warning "Not re-defining macro INT16"
#  else
#    define INT16 HIPGRAPH_INT16
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(INT32)
#    warning "Not re-defining macro INT32"
#  else
#    define INT32 HIPGRAPH_INT32
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(INT64)
#    warning "Not re-defining macro INT64"
#  else
#    define INT64 HIPGRAPH_INT64
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(UINT8)
#    warning "Not re-defining macro UINT8"
#  else
#    define UINT8 HIPGRAPH_UINT8
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(UINT16)
#    warning "Not re-defining macro UINT16"
#  else
#    define UINT16 HIPGRAPH_UINT16
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(UINT32)
#    warning "Not re-defining macro UINT32"
#  else
#    define UINT32 HIPGRAPH_UINT32
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(UINT64)
#    warning "Not re-defining macro UINT64"
#  else
#    define UINT64 HIPGRAPH_UINT64
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(FLOAT32)
#    warning "Not re-defining macro FLOAT32"
#  else
#    define FLOAT32 HIPGRAPH_FLOAT32
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(FLOAT64)
#    warning "Not re-defining macro FLOAT64"
#  else
#    define FLOAT64 HIPGRAPH_FLOAT64
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(SIZE_T)
#    warning "Not re-defining macro SIZE_T"
#  else
#    define SIZE_T HIPGRAPH_SIZE_T
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(BOOL)
#    warning "Not re-defining macro BOOL"
#  else
#    define BOOL HIPGRAPH_BOOL
#  endif
#endif
#if defined(HIPGRAPH_NONPREFIXED_ALIASES)
#  if defined(NTYPES)
#    warning "Not re-defining macro NTYPES"
#  else
#    define NTYPES HIPGRAPH_NTYPES
#  endif
#endif

#ifdef __cplusplus
}
#endif
