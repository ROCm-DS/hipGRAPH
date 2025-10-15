// -*- C -*-
// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT


#include <cugraph_c/hipgraph_c/error.h>
#include <hipgraph_c/hipgraph_c/error.h>
// C enums are only guaranteed to be 16 bits wide, and their signedness is not
// specified. So this is pretty much the most sane value.
#define _hipgraph_INVALID_VAL 32767

static inline cugraph_error_code_t _hipgraph_to_cugraph_error_code_t(hipgraph_error_code_t val)
{
    switch(val) {
        case HIPGRAPH_SUCCESS: return CUGRAPH_SUCCESS;
        case HIPGRAPH_UNKNOWN_ERROR: return CUGRAPH_UNKNOWN_ERROR;
        case HIPGRAPH_INVALID_HANDLE: return CUGRAPH_INVALID_HANDLE;
        case HIPGRAPH_ALLOC_ERROR: return CUGRAPH_ALLOC_ERROR;
        case HIPGRAPH_INVALID_INPUT: return CUGRAPH_INVALID_INPUT;
        case HIPGRAPH_NOT_IMPLEMENTED: return CUGRAPH_NOT_IMPLEMENTED;
        case HIPGRAPH_UNSUPPORTED_TYPE_COMBINATION: return CUGRAPH_UNSUPPORTED_TYPE_COMBINATION;
    }
    return _hipgraph_INVALID_VAL;
}

static inline hipgraph_error_code_t _cugraph_to_hipgraph_error_code_t(cugraph_error_code_t val)
{
    switch(val) {
        case CUGRAPH_SUCCESS: return HIPGRAPH_SUCCESS;
        case CUGRAPH_UNKNOWN_ERROR: return HIPGRAPH_UNKNOWN_ERROR;
        case CUGRAPH_INVALID_HANDLE: return HIPGRAPH_INVALID_HANDLE;
        case CUGRAPH_ALLOC_ERROR: return HIPGRAPH_ALLOC_ERROR;
        case CUGRAPH_INVALID_INPUT: return HIPGRAPH_INVALID_INPUT;
        case CUGRAPH_NOT_IMPLEMENTED: return HIPGRAPH_NOT_IMPLEMENTED;
        case CUGRAPH_UNSUPPORTED_TYPE_COMBINATION: return HIPGRAPH_UNSUPPORTED_TYPE_COMBINATION;
    }
    return _hipgraph_INVALID_VAL;
}

