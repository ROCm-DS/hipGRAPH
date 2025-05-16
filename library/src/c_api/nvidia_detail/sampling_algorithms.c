// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*! \file */
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

#include "common.h"
#include <cugraph_c/sampling_algorithms.h>
#include "hipgraph/hipgraph_c/sampling_algorithms.h"

hipgraph_error_code_t
    hipgraph_uniform_random_walks(const hipgraph_resource_handle_t*               handle,
                                  hipgraph_graph_t*                               graph,
                                  const hipgraph_type_erased_device_array_view_t* start_vertices,
                                  size_t                                          max_length,
                                  hipgraph_random_walk_result_t**                 result,
                                  hipgraph_error_t**                              error)
{
    cugraph_error_code_t err;
    err = cugraph_uniform_random_walks(
        (const cugraph_resource_handle_t*)handle,
        (cugraph_graph_t*)graph,
        (const cugraph_type_erased_device_array_view_t*)start_vertices,
        max_length,
        (cugraph_random_walk_result_t**)result,
        (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
}

hipgraph_error_code_t
    hipgraph_biased_random_walks(const hipgraph_resource_handle_t*               handle,
                                 hipgraph_graph_t*                               graph,
                                 const hipgraph_type_erased_device_array_view_t* start_vertices,
                                 size_t                                          max_length,
                                 hipgraph_random_walk_result_t**                 result,
                                 hipgraph_error_t**                              error)
{
    cugraph_error_code_t err;
    err = cugraph_biased_random_walks(
        (const cugraph_resource_handle_t*)handle,
        (cugraph_graph_t*)graph,
        (const cugraph_type_erased_device_array_view_t*)start_vertices,
        max_length,
        (cugraph_random_walk_result_t**)result,
        (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
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
    cugraph_error_code_t err;
    err = cugraph_node2vec_random_walks(
        (const cugraph_resource_handle_t*)handle,
        (cugraph_graph_t*)graph,
        (const cugraph_type_erased_device_array_view_t*)start_vertices,
        max_length,
        p,
        q,
        (cugraph_random_walk_result_t**)result,
        (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
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
    cugraph_error_code_t err;
    err = cugraph_node2vec((const cugraph_resource_handle_t*)handle,
                           (cugraph_graph_t*)graph,
                           (const cugraph_type_erased_device_array_view_t*)sources,
                           max_depth,
                           (bool_t)compress_result,
                           p,
                           q,
                           (cugraph_random_walk_result_t**)result,
                           (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
}

size_t hipgraph_random_walk_result_get_max_path_length(hipgraph_random_walk_result_t* result)
{
    return cugraph_random_walk_result_get_max_path_length((cugraph_random_walk_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_random_walk_result_get_paths(hipgraph_random_walk_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_random_walk_result_get_paths(
        (cugraph_random_walk_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_random_walk_result_get_weights(hipgraph_random_walk_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_random_walk_result_get_weights(
        (cugraph_random_walk_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_random_walk_result_get_path_sizes(hipgraph_random_walk_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_random_walk_result_get_path_sizes(
        (cugraph_random_walk_result_t*)result);
}

void hipgraph_random_walk_result_free(hipgraph_random_walk_result_t* result)
{
    cugraph_random_walk_result_free((cugraph_random_walk_result_t*)result);
}

hipgraph_error_code_t hipgraph_sampling_options_create(hipgraph_sampling_options_t** options,
                                                       hipgraph_error_t**            error)
{
    cugraph_error_code_t err;
    err = cugraph_sampling_options_create((cugraph_sampling_options_t**)options,
                                          (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
}

void hipgraph_sampling_set_renumber_results(hipgraph_sampling_options_t* options,
                                            hipgraph_bool_t              value)
{
    cugraph_sampling_set_renumber_results((cugraph_sampling_options_t*)options, (bool_t)value);
}

void hipgraph_sampling_set_compress_per_hop(hipgraph_sampling_options_t* options,
                                            hipgraph_bool_t              value)
{
    cugraph_sampling_set_compress_per_hop((cugraph_sampling_options_t*)options, (bool_t)value);
}

void hipgraph_sampling_set_with_replacement(hipgraph_sampling_options_t* options,
                                            hipgraph_bool_t              value)
{
    cugraph_sampling_set_with_replacement((cugraph_sampling_options_t*)options, (bool_t)value);
}

void hipgraph_sampling_set_return_hops(hipgraph_sampling_options_t* options, hipgraph_bool_t value)
{
    cugraph_sampling_set_return_hops((cugraph_sampling_options_t*)options, (bool_t)value);
}

void hipgraph_sampling_set_compression_type(hipgraph_sampling_options_t* options,
                                            hipgraph_compression_type_t  value)
{
    cugraph_sampling_set_compression_type((cugraph_sampling_options_t*)options,
                                          (cugraph_compression_type_t)value);
}

void hipgraph_sampling_set_prior_sources_behavior(hipgraph_sampling_options_t*      options,
                                                  hipgraph_prior_sources_behavior_t value)
{
    cugraph_sampling_set_prior_sources_behavior((cugraph_sampling_options_t*)options,
                                                (cugraph_prior_sources_behavior_t)value);
}

void hipgraph_sampling_set_dedupe_sources(hipgraph_sampling_options_t* options,
                                          hipgraph_bool_t              value)
{
    cugraph_sampling_set_dedupe_sources((cugraph_sampling_options_t*)options, (bool_t)value);
}

void hipgraph_sampling_options_free(hipgraph_sampling_options_t* options)
{
    cugraph_sampling_options_free((cugraph_sampling_options_t*)options);
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
    cugraph_error_code_t err;
    err = cugraph_uniform_neighbor_sample(
        (const cugraph_resource_handle_t*)handle,
        (cugraph_graph_t*)graph,
        (const cugraph_type_erased_device_array_view_t*)start_vertices,
        (const cugraph_type_erased_device_array_view_t*)start_vertex_labels,
        (const cugraph_type_erased_device_array_view_t*)label_list,
        (const cugraph_type_erased_device_array_view_t*)label_to_comm_rank,
        (const cugraph_type_erased_device_array_view_t*)label_offsets,
        (const cugraph_type_erased_host_array_view_t*)fan_out,
        (cugraph_rng_state_t*)rng_state,
        (const cugraph_sampling_options_t*)options,
        (bool_t)do_expensive_check,
        (cugraph_sample_result_t**)result,
        (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_sources(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_sample_result_get_sources(
        (const cugraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_destinations(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_sample_result_get_destinations(
        (const cugraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_majors(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_sample_result_get_majors(
        (const cugraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_minors(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_sample_result_get_minors(
        (const cugraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_major_offsets(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_sample_result_get_major_offsets(
        (const cugraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_start_labels(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_sample_result_get_start_labels(
        (const cugraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_edge_id(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_sample_result_get_edge_id(
        (const cugraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_edge_type(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_sample_result_get_edge_type(
        (const cugraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_edge_weight(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_sample_result_get_edge_weight(
        (const cugraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_hop(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_sample_result_get_hop(
        (const cugraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_label_hop_offsets(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_sample_result_get_label_hop_offsets(
        (const cugraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_index(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_sample_result_get_index(
        (const cugraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_offsets(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_sample_result_get_offsets(
        (const cugraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_renumber_map(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_sample_result_get_renumber_map(
        (const cugraph_sample_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_sample_result_get_renumber_map_offsets(const hipgraph_sample_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)
        cugraph_sample_result_get_renumber_map_offsets((const cugraph_sample_result_t*)result);
}

void hipgraph_sample_result_free(hipgraph_sample_result_t* result)
{
    cugraph_sample_result_free((cugraph_sample_result_t*)result);
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
    cugraph_error_code_t err;
    err = cugraph_test_sample_result_create(
        (const cugraph_resource_handle_t*)handle,
        (const cugraph_type_erased_device_array_view_t*)srcs,
        (const cugraph_type_erased_device_array_view_t*)dsts,
        (const cugraph_type_erased_device_array_view_t*)edge_id,
        (const cugraph_type_erased_device_array_view_t*)edge_type,
        (const cugraph_type_erased_device_array_view_t*)wgt,
        (const cugraph_type_erased_device_array_view_t*)hop,
        (const cugraph_type_erased_device_array_view_t*)label,
        (cugraph_sample_result_t**)result,
        (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
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
    cugraph_error_code_t err;
    err = cugraph_test_uniform_neighborhood_sample_result_create(
        (const cugraph_resource_handle_t*)handle,
        (const cugraph_type_erased_device_array_view_t*)srcs,
        (const cugraph_type_erased_device_array_view_t*)dsts,
        (const cugraph_type_erased_device_array_view_t*)edge_id,
        (const cugraph_type_erased_device_array_view_t*)edge_type,
        (const cugraph_type_erased_device_array_view_t*)weight,
        (const cugraph_type_erased_device_array_view_t*)hop,
        (const cugraph_type_erased_device_array_view_t*)label,
        (cugraph_sample_result_t**)result,
        (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
}

hipgraph_error_code_t
    hipgraph_select_random_vertices(const hipgraph_resource_handle_t*     handle,
                                    const hipgraph_graph_t*               graph,
                                    hipgraph_rng_state_t*                 rng_state,
                                    size_t                                num_vertices,
                                    hipgraph_type_erased_device_array_t** vertices,
                                    hipgraph_error_t**                    error)
{
    cugraph_error_code_t err;
    err = cugraph_select_random_vertices((const cugraph_resource_handle_t*)handle,
                                         (const cugraph_graph_t*)graph,
                                         (cugraph_rng_state_t*)rng_state,
                                         num_vertices,
                                         (cugraph_type_erased_device_array_t**)vertices,
                                         (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
}
