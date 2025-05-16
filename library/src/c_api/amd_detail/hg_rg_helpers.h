// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 */

#include <rocgraph-types.h>

static inline int hghelper_rocgraph_bool_is_invalid(rocgraph_bool arg__)
{
    switch(arg__)
    {
    case rocgraph_bool_false:
        return 0;
    case rocgraph_bool_true:
        return 0;
    }
    return 1;
}

static inline int hghelper_rocgraph_compression_type_is_invalid(rocgraph_compression_type arg__)
{
    switch(arg__)
    {
    case rocgraph_compression_type_coo:
        return 0;
    case rocgraph_compression_type_csc:
        return 0;
    case rocgraph_compression_type_csr:
        return 0;
    case rocgraph_compression_type_dcsc:
        return 0;
    case rocgraph_compression_type_dcsr:
        return 0;
    }
    return 1;
}

static inline int hghelper_rocgraph_data_type_id_is_invalid(rocgraph_data_type_id arg__)
{
    switch(arg__)
    {
    case rocgraph_data_type_id_float32:
        return 0;
    case rocgraph_data_type_id_float64:
        return 0;
    case rocgraph_data_type_id_int32:
        return 0;
    case rocgraph_data_type_id_int64:
        return 0;
    case rocgraph_data_type_id_ntypes:
        return 0;
    case rocgraph_data_type_id_size_t:
        return 0;
    }
    return 1;
}

static inline int
    hghelper_rocgraph_generator_distribution_is_invalid(rocgraph_generator_distribution arg__)
{
    switch(arg__)
    {
    case rocgraph_generator_distribution_power_law:
        return 0;
    case rocgraph_generator_distribution_uniform:
        return 0;
    }
    return 1;
}

static inline int hghelper_rocgraph_k_core_degree_type_is_invalid(rocgraph_k_core_degree_type arg__)
{
    switch(arg__)
    {
    case rocgraph_k_core_degree_type_in:
        return 0;
    case rocgraph_k_core_degree_type_inout:
        return 0;
    case rocgraph_k_core_degree_type_out:
        return 0;
    }
    return 1;
}

static inline int
    hghelper_rocgraph_prior_sources_behavior_is_invalid(rocgraph_prior_sources_behavior arg__)
{
    switch(arg__)
    {
    case rocgraph_prior_sources_behavior_carry_over:
        return 0;
    case rocgraph_prior_sources_behavior_default:
        return 0;
    case rocgraph_prior_sources_behavior_exclude:
        return 0;
    }
    return 1;
}

static inline int hghelper_rocgraph_status_is_invalid(rocgraph_status arg__)
{
    switch(arg__)
    {
    case rocgraph_status_arch_mismatch:
        return 0;
    case rocgraph_status_continue:
        return 0;
    case rocgraph_status_internal_error:
        return 0;
    case rocgraph_status_invalid_handle:
        return 0;
    case rocgraph_status_invalid_input:
        return 0;
    case rocgraph_status_invalid_pointer:
        return 0;
    case rocgraph_status_invalid_size:
        return 0;
    case rocgraph_status_invalid_value:
        return 0;
    case rocgraph_status_memory_error:
        return 0;
    case rocgraph_status_not_implemented:
        return 0;
    case rocgraph_status_not_initialized:
        return 0;
    case rocgraph_status_requires_sorted_storage:
        return 0;
    case rocgraph_status_success:
        return 0;
    case rocgraph_status_thrown_exception:
        return 0;
    case rocgraph_status_type_mismatch:
        return 0;
    case rocgraph_status_unknown_error:
        return 0;
    case rocgraph_status_unsupported_type_combination:
        return 0;
    }
    return 1;
}
