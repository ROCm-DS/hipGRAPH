// -*- C -*-
// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT


#include <cugraph_c/hipgraph_c/graph_generators.h>
#include <hipgraph_c/hipgraph_c/graph_generators.h>
// C enums are only guaranteed to be 16 bits wide, and their signedness is not
// specified. So this is pretty much the most sane value.
#define _hipgraph_INVALID_VAL 32767

static inline cugraph_generator_distribution_t _hipgraph_to_cugraph_generator_distribution_t(hipgraph_generator_distribution_t val)
{
    switch(val) {
        case HIPGRAPH_POWER_LAW: return POWER_LAW;
        case HIPGRAPH_UNIFORM: return UNIFORM;
    }
    return _hipgraph_INVALID_VAL;
}

static inline hipgraph_generator_distribution_t _cugraph_to_hipgraph_generator_distribution_t(cugraph_generator_distribution_t val)
{
    switch(val) {
        case POWER_LAW: return HIPGRAPH_POWER_LAW;
        case UNIFORM: return HIPGRAPH_UNIFORM;
    }
    return _hipgraph_INVALID_VAL;
}

