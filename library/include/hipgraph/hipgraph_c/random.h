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

#include "hipgraph/hipgraph-export.h"
#include "hipgraph/hipgraph_c/resource_handle.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    int32_t align_;
} hipgraph_rng_state_t;

/**
 * @brief     Create a Random Number Generator State
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  seed        Initial value for seed.  In MG this should be different
 *                          on each GPU
 * @param [out] state       Pointer to the location to store the pointer to the RngState
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not HIPGRAPH_SUCCESS
 * @return error code
 */
HIPGRAPH_EXPORT hipgraph_error_code_t
    hipgraph_rng_state_create(const hipgraph_resource_handle_t* handle,
                              uint64_t                          seed,
                              hipgraph_rng_state_t**            state,
                              hipgraph_error_t**                error);

/**
 * @brief    Destroy a Random Number Generator State
 *
 * @param [in]  p    Pointer to the Random Number Generator State
 */
HIPGRAPH_EXPORT void hipgraph_rng_state_free(hipgraph_rng_state_t* p);

#ifdef __cplusplus
}
#endif
