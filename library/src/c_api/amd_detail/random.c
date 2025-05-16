// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*
 * Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 */

/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "common.h"
#include <rocgraph/rocgraph.h>
#include "hipgraph/hipgraph_c/random.h"

hipgraph_error_code_t hipgraph_rng_state_create(const hipgraph_resource_handle_t* handle,
                                                uint64_t                          seed,
                                                hipgraph_rng_state_t**            state,
                                                hipgraph_error_t**                error)
{
    rocgraph_status rg_status = rocgraph_rng_state_create((const rocgraph_handle_t*)handle,
                                                          seed,
                                                          (rocgraph_rng_state_t**)state,
                                                          (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

void hipgraph_rng_state_free(hipgraph_rng_state_t* p)
{
    rocgraph_rng_state_free((rocgraph_rng_state_t*)p);
}
