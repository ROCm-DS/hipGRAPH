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
#include "hipgraph/hipgraph_c/array.h"
#include "hipgraph/hipgraph_c/core_algorithms.h"
#include "hipgraph/hipgraph_c/graph_generators.h"
#include "hipgraph/hipgraph_c/sampling_algorithms.h"

#include "hg_rg_helpers.h"

#ifdef __cplusplus
extern "C" {
#endif
static inline rocgraph_data_type_id
    hipgraph_data_type_id_t2rocgraph_data_type_id(hipgraph_data_type_id_t that);
static inline rocgraph_bool hipgraph_bool_t2rocgraph_bool(hipgraph_bool_t that);
static inline rocgraph_k_core_degree_type
    hipgraph_k_core_degree_type_t2rocgraph_k_core_degree_type(hipgraph_k_core_degree_type_t that);
static inline rocgraph_generator_distribution
    hipgraph_generator_distribution_t2rocgraph_generator_distribution(
        hipgraph_generator_distribution_t that);
static inline rocgraph_prior_sources_behavior
    hipgraph_prior_sources_behavior_t2rocgraph_prior_sources_behavior(
        hipgraph_prior_sources_behavior_t that);
static inline rocgraph_compression_type
    hipgraph_compression_type_t2rocgraph_compression_type(hipgraph_compression_type_t that);

// C enums are only guaranteed to be 16 bits wide, and their signedness is not
// specified. So this is pretty much the most sane value.
#define RG_INVALID_VAL 32767

rocgraph_data_type_id hipgraph_data_type_id_t2rocgraph_data_type_id(hipgraph_data_type_id_t that)
{
    switch(that)
    {
    case HIPGRAPH_INT32:
        return rocgraph_data_type_id_int32;
    case HIPGRAPH_INT64:
        return rocgraph_data_type_id_int64;
    case HIPGRAPH_FLOAT32:
        return rocgraph_data_type_id_float32;
    case HIPGRAPH_FLOAT64:
        return rocgraph_data_type_id_float64;
    case HIPGRAPH_SIZE_T:
        return rocgraph_data_type_id_size_t;
    case HIPGRAPH_NTYPES:
        return RG_INVALID_VAL;
    }
    return RG_INVALID_VAL;
}

rocgraph_bool hipgraph_bool_t2rocgraph_bool(hipgraph_bool_t that)
{
    switch(that)
    {
    case HIPGRAPH_TRUE:
        return rocgraph_bool_true;
    case HIPGRAPH_FALSE:
        return rocgraph_bool_false;
    }
    return RG_INVALID_VAL;
}

rocgraph_k_core_degree_type
    hipgraph_k_core_degree_type_t2rocgraph_k_core_degree_type(hipgraph_k_core_degree_type_t that)
{
    switch(that)
    {
    case HIPGRAPH_K_CORE_DEGREE_TYPE_IN:
        return rocgraph_k_core_degree_type_in;
    case HIPGRAPH_K_CORE_DEGREE_TYPE_OUT:
        return rocgraph_k_core_degree_type_out;
    case HIPGRAPH_K_CORE_DEGREE_TYPE_INOUT:
        return rocgraph_k_core_degree_type_inout;
    }
    return RG_INVALID_VAL;
}

rocgraph_generator_distribution hipgraph_generator_distribution_t2rocgraph_generator_distribution(
    hipgraph_generator_distribution_t that)
{
    switch(that)
    {
    case HIPGRAPH_POWER_LAW:
        return rocgraph_generator_distribution_power_law;
    case HIPGRAPH_UNIFORM:
        return rocgraph_generator_distribution_uniform;
    }
    return RG_INVALID_VAL;
}

rocgraph_compression_type
    hipgraph_compression_type_t2rocgraph_compression_type(hipgraph_compression_type_t that)
{
    switch(that)
    {
    case HIPGRAPH_COO:
        return rocgraph_compression_type_coo;
    case HIPGRAPH_CSR:
        return rocgraph_compression_type_csr;
    case HIPGRAPH_CSC:
        return rocgraph_compression_type_csc;
    case HIPGRAPH_DCSR:
        return rocgraph_compression_type_dcsr;
    case HIPGRAPH_DCSC:
        return rocgraph_compression_type_dcsc;
    }
    return RG_INVALID_VAL;
}

rocgraph_prior_sources_behavior hipgraph_prior_sources_behavior_t2rocgraph_prior_sources_behavior(
    hipgraph_prior_sources_behavior_t that)
{
    switch(that)
    {
    case HIPGRAPH_DEFAULT:
        return rocgraph_prior_sources_behavior_default;
    case HIPGRAPH_CARRY_OVER:
        return rocgraph_prior_sources_behavior_carry_over;
    case HIPGRAPH_EXCLUDE:
        return rocgraph_prior_sources_behavior_exclude;
    }
    return RG_INVALID_VAL;
}

#undef RG_INVALID_VAL

#ifdef __cplusplus
}
#endif
