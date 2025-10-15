// -*- C -*-
// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common.h"
#include <cugraph_c/resource_handle.h>
#include <hipgraph_c/resource_handle.h>
#include <cugraph_c/hipgraph/hipgraph-common.h>
#include <hipgraph_c/hipgraph/hipgraph-common.h>
#include "error_code.inl.h"

HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_rng_state_create(const hipgraph_resource_handle_t* handle, uint64_t seed, hipgraph_rng_state_t** state, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_rng_state_create((const cugraph_resource_handle_t*)handle, seed, (cugraph_rng_state_t**)state, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT void hipgraph_rng_state_free(hipgraph_rng_state_t* p)
{
    cugraph_rng_state_free((cugraph_rng_state_t*)p);
}




