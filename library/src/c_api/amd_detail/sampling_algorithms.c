// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*
 * Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 */

/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common.h"
#include <rocgraph/rocgraph.h>
#include "hipgraph/hipgraph_c/sampling_algorithms.h"

hipgraph_error_code_t
    hipgraph_uniform_random_walks(const hipgraph_resource_handle_t*               handle,
                                  hipgraph_graph_t*                               graph,
                                  const hipgraph_type_erased_device_array_view_t* start_vertices,
                                  size_t                                          max_length,
                                  hipgraph_random_walk_result_t**                 result,
                                  hipgraph_error_t**                              error)
{
    rocgraph_status rg_status = rocgraph_uniform_random_walks(
        (const rocgraph_handle_t*)handle,
        (rocgraph_graph_t*)graph,
        (const rocgraph_type_erased_device_array_view_t*)start_vertices,
        max_length,
        (rocgraph_random_walk_result_t**)result,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t
    hipgraph_biased_random_walks(const hipgraph_resource_handle_t*               handle,
                                 hipgraph_graph_t*                               graph,
                                 const hipgraph_type_erased_device_array_view_t* start_vertices,
                                 size_t                                          max_length,
                                 hipgraph_random_walk_result_t**                 result,
                                 hipgraph_error_t**                              error)
{
    rocgraph_status rg_status = rocgraph_biased_random_walks(
        (const rocgraph_handle_t*)handle,
        (rocgraph_graph_t*)graph,
        (const rocgraph_type_erased_device_array_view_t*)start_vertices,
        max_length,
        (rocgraph_random_walk_result_t**)result,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t
    hipgraph_node2vec_random_walks(const hipgraph_resource_handle_t*               handle,
                                   hipgraph_graph_t*                               graph,
                                   const hipgraph_type_erased_device_array_view_t* start_vertices,
                                   size_t                                          max_length,
                                   double                                          p,
                                   double                                          q,
                                   hipgraph_random_walk_result_t**                 result,
                                   hipgraph_error_t**                              error)
{
    rocgraph_status rg_status = rocgraph_node2vec_random_walks(
        (const rocgraph_handle_t*)handle,
        (rocgraph_graph_t*)graph,
        (const rocgraph_type_erased_device_array_view_t*)start_vertices,
        max_length,
        p,
        q,
        (rocgraph_random_walk_result_t**)result,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t hipgraph_node2vec(const hipgraph_resource_handle_t*               handle,
                                        hipgraph_graph_t*                               graph,
                                        const hipgraph_type_erased_device_array_view_t* sources,
                                        size_t                                          max_depth,
                                        hipgraph_bool_t                 compress_result,
                                        double                          p,
                                        double                          q,
                                        hipgraph_random_walk_result_t** result,
                                        hipgraph_error_t**              error)
{
    rocgraph_bool rg_compress_result = hipgraph_bool_t2rocgraph_bool(compress_result);
    if(hghelper_rocgraph_bool_is_invalid(rg_compress_result))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status
        = rocgraph_node2vec((const rocgraph_handle_t*)handle,
                            (rocgraph_graph_t*)graph,
                            (const rocgraph_type_erased_device_array_view_t*)sources,
                            max_depth,
                            rg_compress_result,
                            p,
                            q,
                            (rocgraph_random_walk_result_t**)result,
                            (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

size_t hipgraph_random_walk_result_get_max_path_length(hipgraph_random_walk_result_t* result)
{
    return (size_t)rocgraph_random_walk_result_get_max_path_length(
        (rocgraph_random_walk_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_random_walk_result_get_paths(hipgraph_random_walk_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_random_walk_result_get_paths(
        (rocgraph_random_walk_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_random_walk_result_get_weights(hipgraph_random_walk_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_random_walk_result_get_weights(
        (rocgraph_random_walk_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_random_walk_result_get_path_sizes(hipgraph_random_walk_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_random_walk_result_get_path_sizes(
        (rocgraph_random_walk_result_t*)result);
}

void hipgraph_random_walk_result_free(hipgraph_random_walk_result_t* result)
{
    rocgraph_random_walk_result_free((rocgraph_random_walk_result_t*)result);
}

hipgraph_error_code_t hipgraph_sampling_options_create(hipgraph_sampling_options_t** options,
                                                       hipgraph_error_t**            error)
{
    rocgraph_status rg_status = rocgraph_sampling_options_create(
        (rocgraph_sampling_options_t**)options, (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

void hipgraph_sampling_set_retain_seeds(hipgraph_sampling_options_t* options, hipgraph_bool_t value)
{
    rocgraph_sampling_set_retain_seeds((rocgraph_sampling_options_t*)options,
                                       hipgraph_bool_t2rocgraph_bool(value));
}

void hipgraph_sampling_set_renumber_results(hipgraph_sampling_options_t* options,
                                            hipgraph_bool_t              value)
{
    rocgraph_sampling_set_renumber_results((rocgraph_sampling_options_t*)options,
                                           hipgraph_bool_t2rocgraph_bool(value));
}

void hipgraph_sampling_set_compress_per_hop(hipgraph_sampling_options_t* options,
                                            hipgraph_bool_t              value)
{
    rocgraph_sampling_set_compress_per_hop((rocgraph_sampling_options_t*)options,
                                           hipgraph_bool_t2rocgraph_bool(value));
}

void hipgraph_sampling_set_with_replacement(hipgraph_sampling_options_t* options,
                                            hipgraph_bool_t              value)
{
    rocgraph_sampling_set_with_replacement((rocgraph_sampling_options_t*)options,
                                           hipgraph_bool_t2rocgraph_bool(value));
}

void hipgraph_sampling_set_return_hops(hipgraph_sampling_options_t* options, hipgraph_bool_t value)
{
    rocgraph_sampling_set_return_hops((rocgraph_sampling_options_t*)options,
                                      hipgraph_bool_t2rocgraph_bool(value));
}

void hipgraph_sampling_set_compression_type(hipgraph_sampling_options_t* options,
                                            hipgraph_compression_type_t  value)
{
    rocgraph_sampling_set_compression_type(
        (rocgraph_sampling_options_t*)options,
        hipgraph_compression_type_t2rocgraph_compression_type(value));
}

void hipgraph_sampling_set_prior_sources_behavior(hipgraph_sampling_options_t*      options,
                                                  hipgraph_prior_sources_behavior_t value)
{
    rocgraph_sampling_set_prior_sources_behavior(
        (rocgraph_sampling_options_t*)options,
        hipgraph_prior_sources_behavior_t2rocgraph_prior_sources_behavior(value));
}

void hipgraph_sampling_set_dedupe_sources(hipgraph_sampling_options_t* options,
                                          hipgraph_bool_t              value)
{
    rocgraph_sampling_set_dedupe_sources((rocgraph_sampling_options_t*)options,
                                         hipgraph_bool_t2rocgraph_bool(value));
}

void hipgraph_sampling_options_free(hipgraph_sampling_options_t* options)
{
    rocgraph_sampling_options_free((rocgraph_sampling_options_t*)options);
}

hipgraph_error_code_t hipgraph_uniform_neighbor_sample(
    const hipgraph_resource_handle_t*               handle,
    hipgraph_graph_t*                               graph,
    const hipgraph_type_erased_device_array_view_t* start_vertices,
    const hipgraph_type_erased_device_array_view_t* start_vertex_labels,
    const hipgraph_type_erased_device_array_view_t* label_list,
    const hipgraph_type_erased_device_array_view_t* label_to_comm_rank,
    const hipgraph_type_erased_device_array_view_t* label_offsets,
    const hipgraph_type_erased_host_array_view_t*   fan_out,
    hipgraph_rng_state_t*                           rng_state,
    const hipgraph_sampling_options_t*              options,
    hipgraph_bool_t                                 do_expensive_check,
    hipgraph_sample_result_t**                      result,
    hipgraph_error_t**                              error)
{
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_uniform_neighbor_sample(
        (const rocgraph_handle_t*)handle,
        (rocgraph_graph_t*)graph,
        (const rocgraph_type_erased_device_array_view_t*)start_vertices,
        (const rocgraph_type_erased_device_array_view_t*)start_vertex_labels,
        (const rocgraph_type_erased_device_array_view_t*)label_list,
        (const rocgraph_type_erased_device_array_view_t*)label_to_comm_rank,
        (const rocgraph_type_erased_device_array_view_t*)label_offsets,
        (const rocgraph_type_erased_host_array_view_t*)fan_out,
        (rocgraph_rng_state_t*)rng_state,
        (const rocgraph_sampling_options_t*)options,
        rg_do_expensive_check,
        (rocgraph_sample_result_t**)result,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_sources(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_sample_result_get_sources(
        (const rocgraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_destinations(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_sample_result_get_destinations(
        (const rocgraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_majors(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_sample_result_get_majors(
        (const rocgraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_minors(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_sample_result_get_minors(
        (const rocgraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_major_offsets(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_sample_result_get_major_offsets(
        (const rocgraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_start_labels(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_sample_result_get_start_labels(
        (const rocgraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_edge_id(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_sample_result_get_edge_id(
        (const rocgraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_edge_type(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_sample_result_get_edge_type(
        (const rocgraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_edge_weight(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_sample_result_get_edge_weight(
        (const rocgraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_hop(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_sample_result_get_hop(
        (const rocgraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_label_hop_offsets(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_sample_result_get_label_hop_offsets(
        (const rocgraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_index(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_sample_result_get_index(
        (const rocgraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_offsets(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_sample_result_get_offsets(
        (const rocgraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_renumber_map(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_sample_result_get_renumber_map(
        (const rocgraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_renumber_map_offsets(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)
        rocgraph_sample_result_get_renumber_map_offsets((const rocgraph_sample_result_t*)result);
}

void hipgraph_sample_result_free(hipgraph_sample_result_t* result)
{
    rocgraph_sample_result_free((rocgraph_sample_result_t*)result);
}

hipgraph_error_code_t
    hipgraph_test_sample_result_create(const hipgraph_resource_handle_t*               handle,
                                       const hipgraph_type_erased_device_array_view_t* srcs,
                                       const hipgraph_type_erased_device_array_view_t* dsts,
                                       const hipgraph_type_erased_device_array_view_t* edge_id,
                                       const hipgraph_type_erased_device_array_view_t* edge_type,
                                       const hipgraph_type_erased_device_array_view_t* wgt,
                                       const hipgraph_type_erased_device_array_view_t* hop,
                                       const hipgraph_type_erased_device_array_view_t* label,
                                       hipgraph_sample_result_t**                      result,
                                       hipgraph_error_t**                              error)
{
    rocgraph_status rg_status = rocgraph_test_sample_result_create(
        (const rocgraph_handle_t*)handle,
        (const rocgraph_type_erased_device_array_view_t*)srcs,
        (const rocgraph_type_erased_device_array_view_t*)dsts,
        (const rocgraph_type_erased_device_array_view_t*)edge_id,
        (const rocgraph_type_erased_device_array_view_t*)edge_type,
        (const rocgraph_type_erased_device_array_view_t*)wgt,
        (const rocgraph_type_erased_device_array_view_t*)hop,
        (const rocgraph_type_erased_device_array_view_t*)label,
        (rocgraph_sample_result_t**)result,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t hipgraph_test_uniform_neighborhood_sample_result_create(
    const hipgraph_resource_handle_t*               handle,
    const hipgraph_type_erased_device_array_view_t* srcs,
    const hipgraph_type_erased_device_array_view_t* dsts,
    const hipgraph_type_erased_device_array_view_t* edge_id,
    const hipgraph_type_erased_device_array_view_t* edge_type,
    const hipgraph_type_erased_device_array_view_t* weight,
    const hipgraph_type_erased_device_array_view_t* hop,
    const hipgraph_type_erased_device_array_view_t* label,
    hipgraph_sample_result_t**                      result,
    hipgraph_error_t**                              error)
{
    rocgraph_status rg_status = rocgraph_test_uniform_neighborhood_sample_result_create(
        (const rocgraph_handle_t*)handle,
        (const rocgraph_type_erased_device_array_view_t*)srcs,
        (const rocgraph_type_erased_device_array_view_t*)dsts,
        (const rocgraph_type_erased_device_array_view_t*)edge_id,
        (const rocgraph_type_erased_device_array_view_t*)edge_type,
        (const rocgraph_type_erased_device_array_view_t*)weight,
        (const rocgraph_type_erased_device_array_view_t*)hop,
        (const rocgraph_type_erased_device_array_view_t*)label,
        (rocgraph_sample_result_t**)result,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t
    hipgraph_select_random_vertices(const hipgraph_resource_handle_t*     handle,
                                    const hipgraph_graph_t*               graph,
                                    hipgraph_rng_state_t*                 rng_state,
                                    size_t                                num_vertices,
                                    hipgraph_type_erased_device_array_t** vertices,
                                    hipgraph_error_t**                    error)
{
    rocgraph_status rg_status
        = rocgraph_select_random_vertices((const rocgraph_handle_t*)handle,
                                          (const rocgraph_graph_t*)graph,
                                          (rocgraph_rng_state_t*)rng_state,
                                          num_vertices,
                                          (rocgraph_type_erased_device_array_t**)vertices,
                                          (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}
