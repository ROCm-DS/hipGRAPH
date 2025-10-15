// -*- C -*-
// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT


#include <cugraph_c/hipgraph_c/sampling_algorithms.h>
#include <hipgraph_c/hipgraph_c/sampling_algorithms.h>
// C enums are only guaranteed to be 16 bits wide, and their signedness is not
// specified. So this is pretty much the most sane value.
#define _hipgraph_INVALID_VAL 32767

static inline cugraph_compression_type_t _hipgraph_to_cugraph_compression_type_t(hipgraph_compression_type_t val)
{
    switch(val) {
        case HIPGRAPH_COO: return COO;
        case HIPGRAPH_CSR: return CSR;
        case HIPGRAPH_CSC: return CSC;
        case HIPGRAPH_DCSR: return DCSR;
        case HIPGRAPH_DCSC: return DCSC;
    }
    return _hipgraph_INVALID_VAL;
}

static inline hipgraph_compression_type_t _cugraph_to_hipgraph_compression_type_t(cugraph_compression_type_t val)
{
    switch(val) {
        case COO: return HIPGRAPH_COO;
        case CSR: return HIPGRAPH_CSR;
        case CSC: return HIPGRAPH_CSC;
        case DCSR: return HIPGRAPH_DCSR;
        case DCSC: return HIPGRAPH_DCSC;
    }
    return _hipgraph_INVALID_VAL;
}

