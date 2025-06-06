// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*! \file */
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "common.h"
#include <cugraph_c/resource_handle.h>
#include "hipgraph/hipgraph_c/resource_handle.h"

hipgraph_resource_handle_t* hipgraph_create_resource_handle(void* raft_handle)
{
    cugraph_resource_handle_t* out;
    out = cugraph_create_resource_handle(raft_handle);
    return (hipgraph_resource_handle_t*)out;
}

int hipgraph_resource_handle_get_comm_size(const hipgraph_resource_handle_t* handle)
{
    int out;
    out = cugraph_resource_handle_get_comm_size((const cugraph_resource_handle_t*)handle);
    return out;
}

int hipgraph_resource_handle_get_rank(const hipgraph_resource_handle_t* handle)
{
    int out;
    out = cugraph_resource_handle_get_rank((const cugraph_resource_handle_t*)handle);
    return out;
}

void hipgraph_free_resource_handle(hipgraph_resource_handle_t* handle)
{
    cugraph_free_resource_handle((cugraph_resource_handle_t*)handle);
}
