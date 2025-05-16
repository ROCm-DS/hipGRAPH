// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*
 * Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 */

/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include "hipgraph/hipgraph_c/resource_handle.h"

hipgraph_resource_handle_t* hipgraph_create_resource_handle(void* raft_handle)
{
    rocgraph_handle_t*    handle;
    const rocgraph_status status = rocgraph_create_handle(&handle, raft_handle);
    if(status != rocgraph_status_success)
    {
        return NULL;
    }
    else
    {
        return (hipgraph_resource_handle_t*)handle;
    }
}

int32_t hipgraph_resource_handle_get_comm_size(const hipgraph_resource_handle_t* handle)
{
    int32_t               comm_size;
    const rocgraph_status status
        = rocgraph_handle_get_comm_size((const rocgraph_handle_t*)handle, &comm_size);
    if(status != rocgraph_status_success)
    {
        return -1;
    }
    else
    {
        return comm_size;
    }
}

int32_t hipgraph_resource_handle_get_rank(const hipgraph_resource_handle_t* handle)
{
    int32_t               rank;
    const rocgraph_status status
        = rocgraph_handle_get_rank((const rocgraph_handle_t*)handle, &rank);
    if(status != rocgraph_status_success)
    {
        return -1;
    }
    else
    {
        return rank;
    }
}

void hipgraph_free_resource_handle(hipgraph_resource_handle_t* handle)
{
    const rocgraph_status status = rocgraph_destroy_handle((rocgraph_handle_t*)handle);
    if(status != rocgraph_status_success)
    {
    }
}
