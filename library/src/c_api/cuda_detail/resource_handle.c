// -*- C -*-
// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common.h"
#include <stddef.h>
#include <stdint.h>
#include <cugraph_c/error.h>
#include <hipgraph_c/error.h>
#include <cugraph_c/types.h>
#include <hipgraph_c/types.h>
#include <cugraph_c/hipgraph/hipgraph-common.h>
#include <hipgraph_c/hipgraph/hipgraph-common.h>

HIPGRAPH_EXPORT hipgraph_resource_handle_t* hipgraph_create_resource_handle(void* raft_handle)
{
    cugraph_resource_handle_t* out;
    out = cugraph_create_resource_handle(raft_handle);
    return (hipgraph_resource_handle_t*)out;
}


HIPGRAPH_EXPORT int hipgraph_resource_handle_get_comm_size(const hipgraph_resource_handle_t* handle)
{
    int out;
    out = cugraph_resource_handle_get_comm_size((const cugraph_resource_handle_t*)handle);
    return (int)out;
}


HIPGRAPH_EXPORT int hipgraph_resource_handle_get_rank(const hipgraph_resource_handle_t* handle)
{
    int out;
    out = cugraph_resource_handle_get_rank((const cugraph_resource_handle_t*)handle);
    return (int)out;
}


HIPGRAPH_EXPORT void hipgraph_free_resource_handle(hipgraph_resource_handle_t* handle)
{
    cugraph_free_resource_handle((cugraph_resource_handle_t*)handle);
}




