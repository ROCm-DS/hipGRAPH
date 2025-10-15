// -*- C -*-
// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT


#include <cugraph_c/hipgraph_c/sampling_algorithms.h>
#include <hipgraph_c/hipgraph_c/sampling_algorithms.h>
// C enums are only guaranteed to be 16 bits wide, and their signedness is not
// specified. So this is pretty much the most sane value.
#define _hipgraph_INVALID_VAL 32767

static inline cugraph_prior_sources_behavior_t _hipgraph_to_cugraph_prior_sources_behavior_t(hipgraph_prior_sources_behavior_t val)
{
    switch(val) {
        case HIPGRAPH_DEFAULT: return DEFAULT;
        case HIPGRAPH_CARRY_OVER: return CARRY_OVER;
        case HIPGRAPH_EXCLUDE: return EXCLUDE;
    }
    return _hipgraph_INVALID_VAL;
}

static inline hipgraph_prior_sources_behavior_t _cugraph_to_hipgraph_prior_sources_behavior_t(cugraph_prior_sources_behavior_t val)
{
    switch(val) {
        case DEFAULT: return HIPGRAPH_DEFAULT;
        case CARRY_OVER: return HIPGRAPH_CARRY_OVER;
        case EXCLUDE: return HIPGRAPH_EXCLUDE;
    }
    return _hipgraph_INVALID_VAL;
}

