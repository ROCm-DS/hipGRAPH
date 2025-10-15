// -*- C -*-
// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common.h"
#include <stdint.h>
#include <cugraph_c/hipgraph/hipgraph-common.h>
#include <hipgraph_c/hipgraph/hipgraph-common.h>

HIPGRAPH_EXPORT const char* hipgraph_error_message(const hipgraph_error_t* error)
{
    const char* out;
    out = cugraph_error_message((const cugraph_error_t*)error);
    return (const char*)out;
}


HIPGRAPH_EXPORT void hipgraph_error_free(hipgraph_error_t* error)
{
    cugraph_error_free((cugraph_error_t*)error);
}




