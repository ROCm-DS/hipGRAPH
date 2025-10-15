// -*- C -*-
// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT


#include <cugraph_c/hipgraph_c/types.h>
#include <hipgraph_c/hipgraph_c/types.h>
// C enums are only guaranteed to be 16 bits wide, and their signedness is not
// specified. So this is pretty much the most sane value.
#define _hipgraph_INVALID_VAL 32767

static inline cugraph_data_type_id_t _hipgraph_to_cugraph_data_type_id_t(hipgraph_data_type_id_t val)
{
    switch(val) {
        case HIPGRAPH_INT8: return INT8;
        case HIPGRAPH_INT16: return INT16;
        case HIPGRAPH_INT32: return INT32;
        case HIPGRAPH_INT64: return INT64;
        case HIPGRAPH_UINT8: return UINT8;
        case HIPGRAPH_UINT16: return UINT16;
        case HIPGRAPH_UINT32: return UINT32;
        case HIPGRAPH_UINT64: return UINT64;
        case HIPGRAPH_FLOAT32: return FLOAT32;
        case HIPGRAPH_FLOAT64: return FLOAT64;
        case HIPGRAPH_SIZE_T: return SIZE_T;
        case HIPGRAPH_BOOL: return BOOL;
        case HIPGRAPH_NTYPES: return NTYPES;
    }
    return _hipgraph_INVALID_VAL;
}

static inline hipgraph_data_type_id_t _cugraph_to_hipgraph_data_type_id_t(cugraph_data_type_id_t val)
{
    switch(val) {
        case INT8: return HIPGRAPH_INT8;
        case INT16: return HIPGRAPH_INT16;
        case INT32: return HIPGRAPH_INT32;
        case INT64: return HIPGRAPH_INT64;
        case UINT8: return HIPGRAPH_UINT8;
        case UINT16: return HIPGRAPH_UINT16;
        case UINT32: return HIPGRAPH_UINT32;
        case UINT64: return HIPGRAPH_UINT64;
        case FLOAT32: return HIPGRAPH_FLOAT32;
        case FLOAT64: return HIPGRAPH_FLOAT64;
        case SIZE_T: return HIPGRAPH_SIZE_T;
        case BOOL: return HIPGRAPH_BOOL;
        case NTYPES: return HIPGRAPH_NTYPES;
    }
    return _hipgraph_INVALID_VAL;
}

