// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
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

#include <rocgraph/rocgraph.h>
#include "hipgraph/hipgraph_c/error.h"
#include "hipgraph/hipgraph_c/array.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline hipgraph_error_code_t rocgraph_status2hipgraph_error_code_t(rocgraph_status that);
static inline hipgraph_data_type_id_t
    rocgraph_data_type_id2hipgraph_data_type_id_t(rocgraph_data_type_id that);

// C enums are only guaranteed to be 16 bits wide, and their signedness is not
// specified. So this is pretty much the most sane value.
#define HG_INVALID_VAL 32767

hipgraph_error_code_t rocgraph_status2hipgraph_error_code_t(rocgraph_status that)
{
    switch(that)
    {
    case rocgraph_status_success:
        return HIPGRAPH_SUCCESS;
    case rocgraph_status_unknown_error:
        return HIPGRAPH_UNKNOWN_ERROR;
    case rocgraph_status_invalid_handle:
        return HIPGRAPH_INVALID_HANDLE;
    case rocgraph_status_invalid_value:
    case rocgraph_status_invalid_input:
    case rocgraph_status_invalid_pointer:
    case rocgraph_status_invalid_size:
        return HIPGRAPH_INVALID_INPUT;
    case rocgraph_status_not_implemented:
        return HIPGRAPH_NOT_IMPLEMENTED;
    case rocgraph_status_memory_error:
        return HIPGRAPH_ALLOC_ERROR;
    case rocgraph_status_unsupported_type_combination:
        return HIPGRAPH_UNSUPPORTED_TYPE_COMBINATION;
    case rocgraph_status_internal_error:
    case rocgraph_status_arch_mismatch:
    case rocgraph_status_not_initialized:
    case rocgraph_status_type_mismatch:
    case rocgraph_status_requires_sorted_storage:
    case rocgraph_status_thrown_exception:
        return HIPGRAPH_UNKNOWN_ERROR;
    case rocgraph_status_continue:
        return HIPGRAPH_UNKNOWN_ERROR;
    }
    return HIPGRAPH_UNKNOWN_ERROR;
}

hipgraph_data_type_id_t rocgraph_data_type_id2hipgraph_data_type_id_t(rocgraph_data_type_id that)
{
    switch(that)
    {
    case rocgraph_data_type_id_int32:
        return HIPGRAPH_INT32;
    case rocgraph_data_type_id_int64:
        return HIPGRAPH_INT64;
    case rocgraph_data_type_id_float32:
        return HIPGRAPH_FLOAT32;
    case rocgraph_data_type_id_float64:
        return HIPGRAPH_FLOAT64;
    case rocgraph_data_type_id_size_t:
        return HIPGRAPH_SIZE_T;
    case rocgraph_data_type_id_ntypes:
        return HIPGRAPH_NTYPES;
    }
    return HG_INVALID_VAL;
}

#undef HG_INVALID_VAL

#ifdef __cplusplus
}
#endif
