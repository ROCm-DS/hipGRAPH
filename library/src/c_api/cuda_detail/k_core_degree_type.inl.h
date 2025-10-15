// -*- C -*-
// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT


#include <cugraph_c/hipgraph_c/core_algorithms.h>
#include <hipgraph_c/hipgraph_c/core_algorithms.h>
// C enums are only guaranteed to be 16 bits wide, and their signedness is not
// specified. So this is pretty much the most sane value.
#define _hipgraph_INVALID_VAL 32767

static inline cugraph_k_core_degree_type_t _hipgraph_to_cugraph_k_core_degree_type_t(hipgraph_k_core_degree_type_t val)
{
    switch(val) {
        case HIPGRAPH_K_CORE_DEGREE_TYPE_IN: return K_CORE_DEGREE_TYPE_IN;
        case HIPGRAPH_K_CORE_DEGREE_TYPE_OUT: return K_CORE_DEGREE_TYPE_OUT;
        case HIPGRAPH_K_CORE_DEGREE_TYPE_INOUT: return K_CORE_DEGREE_TYPE_INOUT;
    }
    return _hipgraph_INVALID_VAL;
}

static inline hipgraph_k_core_degree_type_t _cugraph_to_hipgraph_k_core_degree_type_t(cugraph_k_core_degree_type_t val)
{
    switch(val) {
        case K_CORE_DEGREE_TYPE_IN: return HIPGRAPH_K_CORE_DEGREE_TYPE_IN;
        case K_CORE_DEGREE_TYPE_OUT: return HIPGRAPH_K_CORE_DEGREE_TYPE_OUT;
        case K_CORE_DEGREE_TYPE_INOUT: return HIPGRAPH_K_CORE_DEGREE_TYPE_INOUT;
    }
    return _hipgraph_INVALID_VAL;
}

