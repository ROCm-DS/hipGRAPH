// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0
/*
 * Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

/*
 * Copyright (C) 2022-2024, NVIDIA CORPORATION.
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

#include "mg_test_utils.h" /* RUN_MG_TEST */

#include "hipgraph_c/algorithms.h"
#include "hipgraph_c/graph.h"

#include <cmath>
#include <stdbool.h>
#include <unistd.h>

using vertex_t = int32_t;
using edge_t   = int32_t;
using weight_t = float;

hipgraph_data_type_id_t vertex_tid    = HIPGRAPH_INT32;
hipgraph_data_type_id_t edge_tid      = HIPGRAPH_INT32;
hipgraph_data_type_id_t weight_tid    = HIPGRAPH_FLOAT32;
hipgraph_data_type_id_t edge_id_tid   = HIPGRAPH_INT32;
hipgraph_data_type_id_t edge_type_tid = HIPGRAPH_INT32;

namespace
{
    using namespace hipGRAPH::testing;
    int generic_uniform_neighbor_sample_test(
        const hipgraph_resource_handle_t* p_handle,
        vertex_t*                         h_src,
        vertex_t*                         h_dst,
        weight_t*                         h_wgt,
        edge_t*                           h_edge_ids,
        int32_t*                          h_edge_types,
        size_t                            num_vertices,
        size_t                            num_edges,
        vertex_t*                         h_start,
        int*                              h_start_labels,
        size_t                            num_start_vertices,
        int*                              fan_out,
        size_t                            fan_out_size,
        hipgraph_bool_t                   with_replacement,
        hipgraph_bool_t                   return_hops,
        hipgraph_prior_sources_behavior_t prior_sources_behavior,
        hipgraph_bool_t                   dedupe_sources)
    {
        // Create graph
        int                       test_ret_value = 0;
        hipgraph_error_code_t     ret_code       = HIPGRAPH_SUCCESS;
        hipgraph_error_t*         ret_error      = nullptr;
        hipgraph_graph_t*         graph          = nullptr;
        hipgraph_sample_result_t* result         = nullptr;

        int rank = hipgraph_resource_handle_get_rank(p_handle);

        ret_code = create_mg_test_graph_new(p_handle,
                                            vertex_tid,
                                            edge_tid,
                                            h_src,
                                            h_dst,
                                            weight_tid,
                                            h_wgt,
                                            edge_type_tid,
                                            h_edge_types,
                                            edge_id_tid,
                                            h_edge_ids,
                                            num_edges,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_TRUE,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            &graph,
                                            &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "graph creation failed.";

        hipgraph_type_erased_device_array_t*      d_start             = nullptr;
        hipgraph_type_erased_device_array_view_t* d_start_view        = nullptr;
        hipgraph_type_erased_device_array_t*      d_start_labels      = nullptr;
        hipgraph_type_erased_device_array_view_t* d_start_labels_view = nullptr;
        hipgraph_type_erased_host_array_view_t*   h_fan_out_view      = nullptr;

        if(rank > 0)
            num_start_vertices = 0;

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_start_vertices, HIPGRAPH_INT32, &d_start, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "d_start create failed.";

        d_start_view = hipgraph_type_erased_device_array_view(d_start);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, d_start_view, (hipgraph_byte_t*)h_start, &ret_error);

        if(h_start_labels != nullptr)
        {
            ret_code = hipgraph_type_erased_device_array_create(
                p_handle, num_start_vertices, HIPGRAPH_INT32, &d_start_labels, &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "d_start_labels create failed.";

            d_start_labels_view = hipgraph_type_erased_device_array_view(d_start_labels);

            ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
                p_handle, d_start_labels_view, (hipgraph_byte_t*)h_start_labels, &ret_error);

            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "start_labels copy_from_host failed.";
        }

        h_fan_out_view
            = hipgraph_type_erased_host_array_view_create(fan_out, fan_out_size, HIPGRAPH_INT32);

        hipgraph_rng_state_t* rng_state;
        ret_code = hipgraph_rng_state_create(p_handle, rank, &rng_state, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "rng_state create failed.";

        hipgraph_sampling_options_t* sampling_options;

        ret_code = hipgraph_sampling_options_create(&sampling_options, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "sampling_options create failed.";

        hipgraph_sampling_set_with_replacement(sampling_options, with_replacement);
        hipgraph_sampling_set_return_hops(sampling_options, return_hops);
        hipgraph_sampling_set_prior_sources_behavior(sampling_options, prior_sources_behavior);
        hipgraph_sampling_set_dedupe_sources(sampling_options, dedupe_sources);

        ret_code = hipgraph_uniform_neighbor_sample(p_handle,
                                                    graph,
                                                    d_start_view,
                                                    d_start_labels_view,
                                                    nullptr,
                                                    nullptr,
                                                    h_fan_out_view,
                                                    rng_state,
                                                    sampling_options,
                                                    HIPGRAPH_FALSE,
                                                    &result,
                                                    &ret_error);

#ifdef NO_HIPGRAPH_OPS
        EXPECT_NE(ret_code, HIPGRAPH_SUCCESS) << "uniform_neighbor_sample should have failed";
#else
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "uniform_neighbor_sample failed. " << hipgraph_error_message(ret_error);

        hipgraph_sampling_options_free(sampling_options);

        hipgraph_type_erased_device_array_view_t* result_srcs;
        hipgraph_type_erased_device_array_view_t* result_dsts;
        hipgraph_type_erased_device_array_view_t* result_edge_id;
        hipgraph_type_erased_device_array_view_t* result_weights;
        hipgraph_type_erased_device_array_view_t* result_edge_types;
        hipgraph_type_erased_device_array_view_t* result_hops;
        hipgraph_type_erased_device_array_view_t* result_offsets = nullptr;
        hipgraph_type_erased_device_array_view_t* result_labels  = nullptr;

        result_srcs       = hipgraph_sample_result_get_sources(result);
        result_dsts       = hipgraph_sample_result_get_destinations(result);
        result_edge_id    = hipgraph_sample_result_get_edge_id(result);
        result_weights    = hipgraph_sample_result_get_edge_weight(result);
        result_edge_types = hipgraph_sample_result_get_edge_type(result);
        result_hops       = hipgraph_sample_result_get_hop(result);
        result_hops       = hipgraph_sample_result_get_hop(result);

        size_t result_offsets_size = 2;

        if(d_start_labels != nullptr)
        {
            result_offsets = hipgraph_sample_result_get_offsets(result);
            result_labels  = hipgraph_sample_result_get_start_labels(result);
            result_offsets_size
                = 1
                  + hipgraph_test_scalar_reduce(
                      p_handle, hipgraph_type_erased_device_array_view_size(result_offsets) - 1);
        }

        size_t result_size = hipgraph_test_device_gatherv_size(p_handle, result_srcs);

        vertex_t h_result_srcs[result_size];
        vertex_t h_result_dsts[result_size];
        edge_t   h_result_edge_id[result_size];
        weight_t h_result_weight[result_size];
        int32_t  h_result_edge_types[result_size];
        int32_t  h_result_hops[result_size];
        size_t   h_result_offsets[result_offsets_size];
        int      h_result_labels[result_offsets_size - 1];

        if(result_offsets_size == 2)
        {
            h_result_offsets[0] = 0;
            h_result_offsets[1] = result_size;
        }

        ret_code = hipgraph_test_device_gatherv_fill(p_handle, result_srcs, h_result_srcs);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "gatherv_fill failed.";

        ret_code = hipgraph_test_device_gatherv_fill(p_handle, result_dsts, h_result_dsts);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "gatherv_fill failed.";

        if(h_edge_ids != nullptr)
        {
            ret_code
                = hipgraph_test_device_gatherv_fill(p_handle, result_edge_id, h_result_edge_id);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "gatherv_fill failed.";
        }

        if(h_wgt != nullptr)
        {
            ret_code = hipgraph_test_device_gatherv_fill(p_handle, result_weights, h_result_weight);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "gatherv_fill failed.";
        }

        if(h_edge_types != nullptr)
        {
            ret_code = hipgraph_test_device_gatherv_fill(
                p_handle, result_edge_types, h_result_edge_types);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "gatherv_fill failed.";
        }

        if(d_start_labels != nullptr)
        {
            size_t sz = hipgraph_type_erased_device_array_view_size(result_offsets);

            ret_code = hipgraph_test_device_gatherv_fill(p_handle, result_labels, h_result_labels);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "gatherv_fill failed.";

            size_t tmp_result_offsets[sz];

            ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (hipgraph_byte_t*)tmp_result_offsets, result_offsets, &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

            // convert to size
            for(size_t i = 1; i < sz; ++i)
            {
                tmp_result_offsets[i - 1] = tmp_result_offsets[i] - tmp_result_offsets[i - 1];
            }

            hipgraph_test_host_gatherv_fill(
                p_handle, tmp_result_offsets, sz - 1, HIPGRAPH_SIZE_T, h_result_offsets + 1);

            h_result_offsets[0] = 0;
            for(size_t i = 1; i < result_offsets_size; ++i)
            {
                h_result_offsets[i] += h_result_offsets[i - 1];
            }
        }

        //  First, check that all edges are actually part of the graph
        weight_t M_w[num_vertices][num_vertices];
        edge_t   M_edge_id[num_vertices][num_vertices];
        int32_t  M_edge_type[num_vertices][num_vertices];

        for(int i = 0; i < num_vertices; ++i)
            for(int j = 0; j < num_vertices; ++j)
            {
                M_w[i][j]         = 0.0;
                M_edge_id[i][j]   = -1;
                M_edge_type[i][j] = -1;
            }

        for(int i = 0; i < num_edges; ++i)
        {
            if(h_wgt != nullptr)
                M_w[h_src[i]][h_dst[i]] = h_wgt[i];
            else
                M_w[h_src[i]][h_dst[i]] = 1.0;

            if(h_edge_ids != nullptr)
                M_edge_id[h_src[i]][h_dst[i]] = h_edge_ids[i];
            if(h_edge_types != nullptr)
                M_edge_type[h_src[i]][h_dst[i]] = h_edge_types[i];
        }

        for(int i = 0; (i < result_size) && (test_ret_value == 0); ++i)
        {
            if(h_wgt != nullptr)
            {
                EXPECT_EQ(M_w[h_result_srcs[i]][h_result_dsts[i]], h_result_weight[i])
                    << "uniform_neighbor_sample got edge that doesn't exist";
            }
            else
            {
                EXPECT_EQ(M_w[h_result_srcs[i]][h_result_dsts[i]], 1.0)
                    << "uniform_neighbor_sample got edge that doesn't exist";
            }

            if(h_edge_ids != nullptr)
                EXPECT_EQ(M_edge_id[h_result_srcs[i]][h_result_dsts[i]], h_result_edge_id[i])
                    << "uniform_neighbor_sample got edge that doesn't exist";
            if(h_edge_types != nullptr)
                EXPECT_EQ(M_edge_type[h_result_srcs[i]][h_result_dsts[i]], h_result_edge_types[i])
                    << "uniform_neighbor_sample got edge that doesn't exist";
        }

        if((return_hops) && (d_start_labels != nullptr) && (result_offsets_size > 0))
        {
            //
            // For the sampling result to make sense, all sources in hop 0 must be in the seeds,
            // all sources in hop 1 must be a result from hop 0, etc.
            //
            vertex_t  check_v1[result_size];
            vertex_t  check_v2[result_size];
            vertex_t* check_sources      = check_v1;
            vertex_t* check_destinations = check_v2;

            size_t degree[num_vertices];
            for(size_t i = 0; i < num_vertices; ++i)
                degree[i] = 0;

            for(size_t i = 0; i < num_edges; ++i)
            {
                degree[h_src[i]]++;
            }

            for(size_t label_id = 0; label_id < (result_offsets_size - 1); ++label_id)
            {
                // Skip any labels we already processed
                bool already_phipessed = false;
                for(size_t i = 0; (i < label_id) && !already_phipessed; ++i)
                    already_phipessed = (h_result_labels[label_id] == h_result_labels[i]);

                if(already_phipessed)
                    continue;

                size_t sources_size      = 0;
                size_t destinations_size = 0;

                // Fill sources with the input sources
                for(size_t i = 0; i < num_start_vertices; ++i)
                {
                    if(h_start_labels[i] == h_result_labels[label_id])
                    {
                        check_sources[sources_size] = h_start[i];
                        ++sources_size;
                    }
                }

                for(int hop = 0; hop < fan_out_size; ++hop)
                {
                    if(prior_sources_behavior == CARRY_OVER)
                    {
                        destinations_size = sources_size;
                        for(size_t i = 0; i < sources_size; ++i)
                        {
                            check_destinations[i] = check_sources[i];
                        }
                    }

                    for(size_t current_label_id = label_id;
                        current_label_id < (result_offsets_size - 1);
                        ++current_label_id)
                    {
                        if(h_result_labels[current_label_id] == h_result_labels[label_id])
                        {
                            for(size_t i = h_result_offsets[current_label_id];
                                (i < h_result_offsets[current_label_id + 1])
                                && (test_ret_value == 0);
                                ++i)
                            {
                                if(h_result_hops[i] == hop)
                                {
                                    bool found = false;
                                    for(size_t j = 0; (!found) && (j < sources_size); ++j)
                                    {
                                        found = (h_result_srcs[i] == check_sources[j]);
                                    }

                                    EXPECT_EQ(found)
                                        << "encountered source vertex that was not part of "
                                           "previous frontier";
                                }

                                if(prior_sources_behavior, CARRY_OVER)
                                {
                                    // Make sure destination isn't already in the source list
                                    bool found = false;
                                    for(size_t j = 0; (!found) && (j < destinations_size); ++j)
                                    {
                                        found = (h_result_dsts[i] == check_destinations[j]);
                                    }

                                    if(!found)
                                    {
                                        check_destinations[destinations_size] = h_result_dsts[i];
                                        ++destinations_size;
                                    }
                                }
                                else
                                {
                                    check_destinations[destinations_size] = h_result_dsts[i];
                                    ++destinations_size;
                                }
                            }
                        }
                    }

                    vertex_t* tmp      = check_sources;
                    check_sources      = check_destinations;
                    check_destinations = tmp;
                    sources_size       = destinations_size;
                    destinations_size  = 0;
                }

                if(prior_sources_behavior == EXCLUDE)
                {
                    // Make sure vertex v only appears as source in the first hop after it is encountered
                    for(size_t current_label_id = label_id;
                        current_label_id < (result_offsets_size - 1);
                        ++current_label_id)
                    {
                        if(h_result_labels[current_label_id] == h_result_labels[label_id])
                        {
                            for(size_t i = h_result_offsets[current_label_id];
                                (i < h_result_offsets[current_label_id + 1])
                                && (test_ret_value == 0);
                                ++i)
                            {
                                for(size_t j = i + 1; (j < h_result_offsets[current_label_id + 1])
                                                      && (test_ret_value == 0);
                                    ++j)
                                {
                                    if(h_result_srcs[i] == h_result_srcs[j])
                                    {
                                        EXPECT_EQ(h_result_hops[i], h_result_hops[j])
                                            << "source vertex should not have been used in "
                                               "diferent hops";
                                    }
                                }
                            }
                        }
                    }
                }

                if(dedupe_sources)
                {
                    // Make sure vertex v only appears as source once for each edge after it appears as destination
                    // Externally test this by verifying that vertex v only appears in <= hop size/degree
                    for(size_t current_label_id = label_id;
                        current_label_id < (result_offsets_size - 1);
                        ++current_label_id)
                    {
                        if(h_result_labels[current_label_id] == h_result_labels[label_id])
                        {
                            for(size_t i = h_result_offsets[current_label_id];
                                (i < h_result_offsets[current_label_id + 1])
                                && (test_ret_value == 0);
                                ++i)
                            {
                                if(h_result_hops[i] > 0)
                                {
                                    size_t num_occurrences = 1;
                                    for(size_t j = i + 1;
                                        j < h_result_offsets[current_label_id + 1];
                                        ++j)
                                    {
                                        if((h_result_srcs[j] == h_result_srcs[i])
                                           && (h_result_hops[j] == h_result_hops[i]))
                                            num_occurrences++;
                                    }

                                    if(fan_out[h_result_hops[i]] < 0)
                                    {
                                        EXPECT_LE(num_occurrences, degree[h_result_srcs[i]])
                                            << "source vertex used in too many return edges";
                                    }
                                    else
                                    {
                                        EXPECT_LT(num_occurrences, fan_out[h_result_hops[i]])
                                            << "source vertex used in too many return edges";
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        hipgraph_sample_result_free(result);
#endif

        hipgraph_mg_graph_free(graph);
        hipgraph_error_free(ret_error);
        return test_ret_value;
    }

    int test_uniform_neighbor_sample(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 8;
        size_t num_vertices = 6;
        size_t fan_out_size = 2;
        size_t num_starts   = 2;

        vertex_t src[]     = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]     = {1, 3, 4, 0, 1, 3, 5, 5};
        edge_t   idx[]     = {0, 1, 2, 3, 4, 5, 6, 7};
        vertex_t start[]   = {2, 2};
        int      fan_out[] = {1, 2};

        hipgraph_bool_t                   with_replacement       = HIPGRAPH_FALSE;
        hipgraph_bool_t                   return_hops            = HIPGRAPH_TRUE;
        hipgraph_prior_sources_behavior_t prior_sources_behavior = DEFAULT;
        hipgraph_bool_t                   dedupe_sources         = HIPGRAPH_FALSE;

        return generic_uniform_neighbor_sample_test(p_handle,
                                                    src,
                                                    dst,
                                                    nullptr,
                                                    idx,
                                                    nullptr,
                                                    num_vertices,
                                                    num_edges,
                                                    start,
                                                    nullptr,
                                                    num_starts,
                                                    fan_out,
                                                    fan_out_size,
                                                    with_replacement,
                                                    return_hops,
                                                    prior_sources_behavior,
                                                    dedupe_sources);
    }

    int test_uniform_neighbor_from_alex(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges        = 12;
        size_t num_vertices     = 5;
        size_t fan_out_size     = 2;
        size_t num_starts       = 2;
        size_t num_start_labels = 2;

        vertex_t src[]     = {0, 1, 2, 3, 4, 3, 4, 2, 0, 1, 0, 2};
        vertex_t dst[]     = {1, 2, 4, 2, 3, 4, 1, 1, 2, 3, 4, 4};
        edge_t   idx[]     = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        int32_t  typ[]     = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0};
        weight_t wgt[]     = {0.0, 0.1, 0.2, 3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.10, 0.11};
        vertex_t start[]   = {0, 4};
        int32_t  batch[]   = {0, 1};
        int      fan_out[] = {2, 2};

        hipgraph_bool_t store_transposed = HIPGRAPH_FALSE;

        int                   test_ret_value = 0;
        hipgraph_error_code_t ret_code       = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error      = nullptr;

        hipgraph_graph_t*         graph  = nullptr;
        hipgraph_sample_result_t* result = nullptr;

        hipgraph_bool_t                   with_replacement       = HIPGRAPH_FALSE;
        hipgraph_bool_t                   return_hops            = HIPGRAPH_TRUE;
        hipgraph_prior_sources_behavior_t prior_sources_behavior = DEFAULT;
        hipgraph_bool_t                   dedupe_sources         = HIPGRAPH_FALSE;
        hipgraph_bool_t                   renumber_results       = HIPGRAPH_FALSE;
        hipgraph_compression_type_t       compression            = COO;
        hipgraph_bool_t                   compress_per_hop       = HIPGRAPH_FALSE;

        hipgraph_type_erased_device_array_t*      d_start        = nullptr;
        hipgraph_type_erased_device_array_t*      d_label        = nullptr;
        hipgraph_type_erased_device_array_view_t* d_start_view   = nullptr;
        hipgraph_type_erased_device_array_view_t* d_label_view   = nullptr;
        hipgraph_type_erased_host_array_view_t*   h_fan_out_view = nullptr;

        int rank = hipgraph_resource_handle_get_rank(p_handle);

        hipgraph_rng_state_t* rng_state;
        ret_code = hipgraph_rng_state_create(p_handle, rank, &rng_state, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "rng_state create failed. " << hipgraph_error_message(ret_error);

        ret_code = create_mg_test_graph_with_properties(p_handle,
                                                        src,
                                                        dst,
                                                        idx,
                                                        typ,
                                                        wgt,
                                                        num_edges,
                                                        store_transposed,
                                                        HIPGRAPH_FALSE,
                                                        &graph,
                                                        &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "graph creation failed. " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_starts, HIPGRAPH_INT32, &d_start, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "d_start create failed.";

        d_start_view = hipgraph_type_erased_device_array_view(d_start);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, d_start_view, (hipgraph_byte_t*)start, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "start copy_from_host failed.";

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_starts, HIPGRAPH_INT32, &d_label, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "d_label create failed.";

        d_label_view = hipgraph_type_erased_device_array_view(d_label);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, d_label_view, (hipgraph_byte_t*)batch, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "start copy_from_host failed.";

        h_fan_out_view
            = hipgraph_type_erased_host_array_view_create(fan_out, fan_out_size, HIPGRAPH_INT32);

        hipgraph_sampling_options_t* sampling_options;

        ret_code = hipgraph_sampling_options_create(&sampling_options, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "sampling_options create failed.";

        hipgraph_sampling_set_with_replacement(sampling_options, with_replacement);
        hipgraph_sampling_set_return_hops(sampling_options, return_hops);
        hipgraph_sampling_set_prior_sources_behavior(sampling_options, prior_sources_behavior);
        hipgraph_sampling_set_dedupe_sources(sampling_options, dedupe_sources);
        hipgraph_sampling_set_renumber_results(sampling_options, renumber_results);
        hipgraph_sampling_set_compression_type(sampling_options, compression);
        hipgraph_sampling_set_compress_per_hop(sampling_options, compress_per_hop);

        ret_code = hipgraph_uniform_neighbor_sample(p_handle,
                                                    graph,
                                                    d_start_view,
                                                    d_label_view,
                                                    nullptr,
                                                    nullptr,
                                                    h_fan_out_view,
                                                    rng_state,
                                                    sampling_options,
                                                    HIPGRAPH_FALSE,
                                                    &result,
                                                    &ret_error);

#ifdef NO_HIPGRAPH_OPS
        EXPECT_NE(ret_code, HIPGRAPH_SUCCESS) << "uniform_neighbor_sample should have failed";
#else
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "uniform_neighbor_sample failed. " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* result_src;
        hipgraph_type_erased_device_array_view_t* result_dst;
        hipgraph_type_erased_device_array_view_t* result_index;
        hipgraph_type_erased_device_array_view_t* result_type;
        hipgraph_type_erased_device_array_view_t* result_weight;
        hipgraph_type_erased_device_array_view_t* result_labels;
        hipgraph_type_erased_device_array_view_t* result_hops;
        hipgraph_type_erased_device_array_view_t* result_offsets;

        result_src     = hipgraph_sample_result_get_sources(result);
        result_dst     = hipgraph_sample_result_get_destinations(result);
        result_index   = hipgraph_sample_result_get_edge_id(result);
        result_type    = hipgraph_sample_result_get_edge_type(result);
        result_weight  = hipgraph_sample_result_get_edge_weight(result);
        result_labels  = hipgraph_sample_result_get_start_labels(result);
        result_hops    = hipgraph_sample_result_get_hop(result);
        result_offsets = hipgraph_sample_result_get_offsets(result);

        size_t result_size  = hipgraph_type_erased_device_array_view_size(result_src);
        size_t offsets_size = hipgraph_type_erased_device_array_view_size(result_offsets);

        vertex_t h_srcs[result_size];
        vertex_t h_dsts[result_size];
        edge_t   h_index[result_size];
        int      h_type[result_size];
        weight_t h_wgt[result_size];
        int      h_labels[result_size];
        int      h_hop[result_size];
        int      h_offsets[offsets_size];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_srcs, result_src, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_dsts, result_dst, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_index, result_index, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_type, result_type, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_wgt, result_weight, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_labels, result_labels, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_offsets, result_offsets, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        for(int k = 0; k < offsets_size - 1; k += fan_out_size)
        {
            for(int h = 0; h < fan_out_size; ++h)
            {
                int hop_start = h_offsets[k + h];
                int hop_end   = h_offsets[k + h + 1];
                for(int i = hop_start; i < hop_end; ++i)
                {
                    h_hop[i] = h;
                }
            }
        }

        for(int k = 0; k < num_start_labels + 1; ++k)
        {
            h_offsets[k] = h_offsets[k * fan_out_size];
        }
        offsets_size = num_start_labels + 1;

        //  NOTE:  The C++ tester does a more thorough validation.  For our purposes
        //  here we will do a simpler validation, merely checking that all edges
        //  are actually part of the graph
        edge_t M[num_vertices][num_vertices];

        for(int i = 0; i < num_vertices; ++i)
            for(int j = 0; j < num_vertices; ++j)
                M[i][j] = -1;

        for(int i = 0; i < num_edges; ++i)
            M[src[i]][dst[i]] = idx[i];

        for(int i = 0; (i < result_size) && (test_ret_value == 0); ++i)
        {
            EXPECT_GE(M[h_srcs[i]][h_dsts[i]], 0)
                << "uniform_neighbor_sample got edge that doesn't exist";
        }
#endif

        hipgraph_sample_result_free(result);

        hipgraph_type_erased_host_array_view_free(h_fan_out_view);
        hipgraph_mg_graph_free(graph);
        hipgraph_error_free(ret_error);
        hipgraph_sampling_options_free(sampling_options);

        return test_ret_value;
    }

    int test_uniform_neighbor_sample_alex_bug(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 156;
        size_t num_vertices = 34;
        size_t fan_out_size = 2;
        size_t num_starts   = 4;
        size_t num_labels   = 3;

        vertex_t src[]
            = {1,  2,  3,  4,  5,  6,  7,  8,  10, 11, 12, 13, 17, 19, 21, 31, 2,  3,  7,  13,
               17, 19, 21, 30, 3,  7,  8,  9,  13, 27, 28, 32, 7,  12, 13, 6,  10, 6,  10, 16,
               16, 30, 32, 33, 33, 33, 32, 33, 32, 33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29,
               32, 33, 25, 27, 31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33, 0,  0,
               0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,
               1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  4,  4,  5,  5,  5,  6,  8,
               8,  8,  9,  13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22, 23, 23, 23, 23, 23,
               24, 24, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32};
        vertex_t dst[]
            = {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,
               1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  4,  4,  5,  5,  5,
               6,  8,  8,  8,  9,  13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22, 23, 23, 23,
               23, 23, 24, 24, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 1,  2,
               3,  4,  5,  6,  7,  8,  10, 11, 12, 13, 17, 19, 21, 31, 2,  3,  7,  13, 17, 19,
               21, 30, 3,  7,  8,  9,  13, 27, 28, 32, 7,  12, 13, 6,  10, 6,  10, 16, 16, 30,
               32, 33, 33, 33, 32, 33, 32, 33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29, 32, 33,
               25, 27, 31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33};
        weight_t wgt[]
            = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

        edge_t edge_ids[]
            = {0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,
               16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
               32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
               48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,
               64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
               80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
               96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
               112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
               128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
               144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155};

        vertex_t start[]                     = {0, 1, 2, 5};
        int32_t  start_labels[]              = {0, 0, 1, 2};
        int32_t  label_list[]                = {0, 1, 2};
        int32_t  label_to_output_comm_rank[] = {0, 0, 1};
        int      fan_out[]                   = {2, 3};

        size_t expected_size[] = {3, 2, 1, 1, 1, 1, 1, 1};

        hipgraph_bool_t                   with_replacement       = HIPGRAPH_FALSE;
        hipgraph_bool_t                   return_hops            = HIPGRAPH_TRUE;
        hipgraph_prior_sources_behavior_t prior_sources_behavior = CARRY_OVER;
        hipgraph_bool_t                   dedupe_sources         = HIPGRAPH_TRUE;
        hipgraph_bool_t                   renumber_results       = HIPGRAPH_FALSE;
        hipgraph_compression_type_t       compression            = COO;
        hipgraph_bool_t                   compress_per_hop       = HIPGRAPH_FALSE;

        // Create graph
        int                       test_ret_value = 0;
        hipgraph_error_code_t     ret_code       = HIPGRAPH_SUCCESS;
        hipgraph_error_t*         ret_error      = nullptr;
        hipgraph_graph_t*         graph          = nullptr;
        hipgraph_sample_result_t* result         = nullptr;

        ret_code = create_mg_test_graph_with_properties(p_handle,
                                                        src,
                                                        dst,
                                                        edge_ids,
                                                        nullptr,
                                                        wgt,
                                                        num_edges,
                                                        HIPGRAPH_FALSE,
                                                        HIPGRAPH_TRUE,
                                                        &graph,
                                                        &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "graph creation failed.";

        hipgraph_type_erased_device_array_t*      d_start                          = nullptr;
        hipgraph_type_erased_device_array_view_t* d_start_view                     = nullptr;
        hipgraph_type_erased_device_array_t*      d_start_labels                   = nullptr;
        hipgraph_type_erased_device_array_view_t* d_start_labels_view              = nullptr;
        hipgraph_type_erased_device_array_t*      d_label_list                     = nullptr;
        hipgraph_type_erased_device_array_view_t* d_label_list_view                = nullptr;
        hipgraph_type_erased_device_array_t*      d_label_to_output_comm_rank      = nullptr;
        hipgraph_type_erased_device_array_view_t* d_label_to_output_comm_rank_view = nullptr;
        hipgraph_type_erased_host_array_view_t*   h_fan_out_view                   = nullptr;

        int rank = hipgraph_resource_handle_get_rank(p_handle);

        if(rank > 0)
        {
            num_starts = 0;
        }

        hipgraph_rng_state_t* rng_state;
        ret_code = hipgraph_rng_state_create(p_handle, rank, &rng_state, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "rng_state create failed. " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_starts, HIPGRAPH_INT32, &d_start, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "d_start create failed.";

        d_start_view = hipgraph_type_erased_device_array_view(d_start);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, d_start_view, (hipgraph_byte_t*)start, &ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_starts, HIPGRAPH_INT32, &d_start_labels, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "d_start_labels create failed.";

        d_start_labels_view = hipgraph_type_erased_device_array_view(d_start_labels);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, d_start_labels_view, (hipgraph_byte_t*)start_labels, &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "start_labels copy_from_host failed.";

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_labels, HIPGRAPH_INT32, &d_label_list, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "d_label_list create failed.";

        d_label_list_view = hipgraph_type_erased_device_array_view(d_label_list);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, d_label_list_view, (hipgraph_byte_t*)label_list, &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "label_list copy_from_host failed.";

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_labels, HIPGRAPH_INT32, &d_label_to_output_comm_rank, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "d_label_to_output_comm_rank create failed.";

        d_label_to_output_comm_rank_view
            = hipgraph_type_erased_device_array_view(d_label_to_output_comm_rank);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle,
            d_label_to_output_comm_rank_view,
            (hipgraph_byte_t*)label_to_output_comm_rank,
            &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "label_to_output_comm_rank copy_from_host failed.";

        h_fan_out_view
            = hipgraph_type_erased_host_array_view_create(fan_out, fan_out_size, HIPGRAPH_INT32);

        hipgraph_sampling_options_t* sampling_options;
        ret_code = hipgraph_sampling_options_create(&sampling_options, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "sampling_options create failed.";

        hipgraph_sampling_set_with_replacement(sampling_options, with_replacement);
        hipgraph_sampling_set_return_hops(sampling_options, return_hops);
        hipgraph_sampling_set_prior_sources_behavior(sampling_options, prior_sources_behavior);
        hipgraph_sampling_set_dedupe_sources(sampling_options, dedupe_sources);
        hipgraph_sampling_set_renumber_results(sampling_options, renumber_results);
        hipgraph_sampling_set_compression_type(sampling_options, compression);
        hipgraph_sampling_set_compress_per_hop(sampling_options, compress_per_hop);

        ret_code = hipgraph_uniform_neighbor_sample(p_handle,
                                                    graph,
                                                    d_start_view,
                                                    d_start_labels_view,
                                                    d_label_list_view,
                                                    d_label_to_output_comm_rank_view,
                                                    h_fan_out_view,
                                                    rng_state,
                                                    sampling_options,
                                                    HIPGRAPH_FALSE,
                                                    &result,
                                                    &ret_error);

#ifdef NO_HIPGRAPH_OPS
        EXPECT_NE(ret_code, HIPGRAPH_SUCCESS) << "uniform_neighbor_sample should have failed";
#else
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "uniform_neighbor_sample failed. " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* result_srcs    = nullptr;
        hipgraph_type_erased_device_array_view_t* result_dsts    = nullptr;
        hipgraph_type_erased_device_array_view_t* result_edge_id = nullptr;
        hipgraph_type_erased_device_array_view_t* result_weights = nullptr;
        hipgraph_type_erased_device_array_view_t* result_hops    = nullptr;
        hipgraph_type_erased_device_array_view_t* result_offsets = nullptr;

        result_srcs    = hipgraph_sample_result_get_sources(result);
        result_dsts    = hipgraph_sample_result_get_destinations(result);
        result_edge_id = hipgraph_sample_result_get_edge_id(result);
        result_weights = hipgraph_sample_result_get_edge_weight(result);
        result_hops    = hipgraph_sample_result_get_hop(result);
        result_offsets = hipgraph_sample_result_get_offsets(result);

        size_t result_size         = hipgraph_type_erased_device_array_view_size(result_srcs);
        size_t result_offsets_size = hipgraph_type_erased_device_array_view_size(result_offsets);

        vertex_t h_srcs[result_size];
        vertex_t h_dsts[result_size];
        edge_t   h_edge_id[result_size];
        weight_t h_weight[result_size];
        int32_t  h_hops[result_size];
        size_t   h_result_offsets[result_offsets_size];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_srcs, result_srcs, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_dsts, result_dsts, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_edge_id, result_edge_id, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_weight, result_weights, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_hops, result_hops, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_result_offsets, result_offsets, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        //  NOTE:  The C++ tester does a more thorough validation.  For our purposes
        //  here we will do a simpler validation, merely checking that all edges
        //  are actually part of the graph
        weight_t M_w[num_vertices][num_vertices];
        edge_t   M_edge_id[num_vertices][num_vertices];

        for(int i = 0; i < num_vertices; ++i)
            for(int j = 0; j < num_vertices; ++j)
            {
                M_w[i][j]       = 0.0;
                M_edge_id[i][j] = -1;
            }

        for(int i = 0; i < num_edges; ++i)
        {
            M_w[src[i]][dst[i]]       = wgt[i];
            M_edge_id[src[i]][dst[i]] = edge_ids[i];
        }

        for(int i = 0; (i < result_size) && (test_ret_value == 0); ++i)
        {
            EXPECT_EQ(M_w[h_srcs[i]][h_dsts[i]], h_weight[i])
                << "uniform_neighbor_sample got edge that doesn't exist";
            EXPECT_EQ(M_edge_id[h_srcs[i]][h_dsts[i]], h_edge_id[i])
                << "uniform_neighbor_sample got edge that doesn't exist";
        }

        EXPECT_EQ(result_offsets_size, expected_size[rank]) << "incorrect number of results";

        hipgraph_sample_result_free(result);
#endif

        hipgraph_sg_graph_free(graph);
        hipgraph_error_free(ret_error);
    }

    int test_uniform_neighbor_sample_sort_by_hop(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 156;
        size_t num_vertices = 34;
        size_t fan_out_size = 2;
        size_t num_starts   = 4;
        size_t num_labels   = 3;

        vertex_t src[]
            = {1,  2,  3,  4,  5,  6,  7,  8,  10, 11, 12, 13, 17, 19, 21, 31, 2,  3,  7,  13,
               17, 19, 21, 30, 3,  7,  8,  9,  13, 27, 28, 32, 7,  12, 13, 6,  10, 6,  10, 16,
               16, 30, 32, 33, 33, 33, 32, 33, 32, 33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29,
               32, 33, 25, 27, 31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33, 0,  0,
               0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,
               1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  4,  4,  5,  5,  5,  6,  8,
               8,  8,  9,  13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22, 23, 23, 23, 23, 23,
               24, 24, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32};
        vertex_t dst[]
            = {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,
               1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  4,  4,  5,  5,  5,
               6,  8,  8,  8,  9,  13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22, 23, 23, 23,
               23, 23, 24, 24, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 1,  2,
               3,  4,  5,  6,  7,  8,  10, 11, 12, 13, 17, 19, 21, 31, 2,  3,  7,  13, 17, 19,
               21, 30, 3,  7,  8,  9,  13, 27, 28, 32, 7,  12, 13, 6,  10, 6,  10, 16, 16, 30,
               32, 33, 33, 33, 32, 33, 32, 33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29, 32, 33,
               25, 27, 31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33};
        weight_t wgt[]
            = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

        edge_t edge_ids[]
            = {0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,
               16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
               32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
               48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,
               64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
               80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
               96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
               112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
               128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
               144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155};

        vertex_t start[]                     = {0, 1, 2, 5};
        int32_t  start_labels[]              = {0, 0, 1, 2};
        int32_t  label_list[]                = {0, 1, 2};
        int32_t  label_to_output_comm_rank[] = {0, 0, 1};
        int      fan_out[]                   = {2, 3};

        size_t expected_size[] = {3, 2, 1, 1, 1, 1, 1, 1};

        hipgraph_bool_t                   with_replacement       = HIPGRAPH_FALSE;
        hipgraph_bool_t                   return_hops            = HIPGRAPH_TRUE;
        hipgraph_prior_sources_behavior_t prior_sources_behavior = CARRY_OVER;
        hipgraph_bool_t                   dedupe_sources         = HIPGRAPH_TRUE;
        hipgraph_bool_t                   renumber_results       = HIPGRAPH_FALSE;
        hipgraph_compression_type_t       compression            = COO;
        hipgraph_bool_t                   compress_per_hop       = HIPGRAPH_FALSE;

        // Create graph
        int                       test_ret_value = 0;
        hipgraph_error_code_t     ret_code       = HIPGRAPH_SUCCESS;
        hipgraph_error_t*         ret_error      = nullptr;
        hipgraph_graph_t*         graph          = nullptr;
        hipgraph_sample_result_t* result         = nullptr;

        ret_code = create_mg_test_graph_with_properties(p_handle,
                                                        src,
                                                        dst,
                                                        edge_ids,
                                                        nullptr,
                                                        wgt,
                                                        num_edges,
                                                        HIPGRAPH_FALSE,
                                                        HIPGRAPH_TRUE,
                                                        &graph,
                                                        &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "graph creation failed.";

        hipgraph_type_erased_device_array_t*      d_start                          = nullptr;
        hipgraph_type_erased_device_array_view_t* d_start_view                     = nullptr;
        hipgraph_type_erased_device_array_t*      d_start_labels                   = nullptr;
        hipgraph_type_erased_device_array_view_t* d_start_labels_view              = nullptr;
        hipgraph_type_erased_device_array_t*      d_label_list                     = nullptr;
        hipgraph_type_erased_device_array_view_t* d_label_list_view                = nullptr;
        hipgraph_type_erased_device_array_t*      d_label_to_output_comm_rank      = nullptr;
        hipgraph_type_erased_device_array_view_t* d_label_to_output_comm_rank_view = nullptr;
        hipgraph_type_erased_host_array_view_t*   h_fan_out_view                   = nullptr;

        int rank = hipgraph_resource_handle_get_rank(p_handle);

        if(rank > 0)
        {
            num_starts = 0;
        }

        hipgraph_rng_state_t* rng_state;
        ret_code = hipgraph_rng_state_create(p_handle, rank, &rng_state, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "rng_state create failed. " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_starts, HIPGRAPH_INT32, &d_start, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "d_start create failed.";

        d_start_view = hipgraph_type_erased_device_array_view(d_start);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, d_start_view, (hipgraph_byte_t*)start, &ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_starts, HIPGRAPH_INT32, &d_start_labels, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "d_start_labels create failed.";

        d_start_labels_view = hipgraph_type_erased_device_array_view(d_start_labels);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, d_start_labels_view, (hipgraph_byte_t*)start_labels, &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "start_labels copy_from_host failed.";

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_labels, HIPGRAPH_INT32, &d_label_list, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "d_label_list create failed.";

        d_label_list_view = hipgraph_type_erased_device_array_view(d_label_list);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, d_label_list_view, (hipgraph_byte_t*)label_list, &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "label_list copy_from_host failed.";

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_labels, HIPGRAPH_INT32, &d_label_to_output_comm_rank, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "d_label_to_output_comm_rank create failed.";

        d_label_to_output_comm_rank_view
            = hipgraph_type_erased_device_array_view(d_label_to_output_comm_rank);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle,
            d_label_to_output_comm_rank_view,
            (hipgraph_byte_t*)label_to_output_comm_rank,
            &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "label_to_output_comm_rank copy_from_host failed.";

        h_fan_out_view
            = hipgraph_type_erased_host_array_view_create(fan_out, fan_out_size, HIPGRAPH_INT32);

        hipgraph_sampling_options_t* sampling_options;
        ret_code = hipgraph_sampling_options_create(&sampling_options, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "sampling_options create failed.";

        hipgraph_sampling_set_with_replacement(sampling_options, with_replacement);
        hipgraph_sampling_set_return_hops(sampling_options, return_hops);
        hipgraph_sampling_set_prior_sources_behavior(sampling_options, prior_sources_behavior);
        hipgraph_sampling_set_dedupe_sources(sampling_options, dedupe_sources);
        hipgraph_sampling_set_renumber_results(sampling_options, renumber_results);
        hipgraph_sampling_set_compression_type(sampling_options, compression);
        hipgraph_sampling_set_compress_per_hop(sampling_options, compress_per_hop);

        ret_code = hipgraph_uniform_neighbor_sample(p_handle,
                                                    graph,
                                                    d_start_view,
                                                    d_start_labels_view,
                                                    d_label_list_view,
                                                    d_label_to_output_comm_rank_view,
                                                    h_fan_out_view,
                                                    rng_state,
                                                    sampling_options,
                                                    HIPGRAPH_FALSE,
                                                    &result,
                                                    &ret_error);

#ifdef NO_HIPGRAPH_OPS
        EXPECT_NE(ret_code, HIPGRAPH_SUCCESS) << "uniform_neighbor_sample should have failed";
#else
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "uniform_neighbor_sample failed. " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* result_srcs    = nullptr;
        hipgraph_type_erased_device_array_view_t* result_dsts    = nullptr;
        hipgraph_type_erased_device_array_view_t* result_edge_id = nullptr;
        hipgraph_type_erased_device_array_view_t* result_weights = nullptr;
        hipgraph_type_erased_device_array_view_t* result_hops    = nullptr;
        hipgraph_type_erased_device_array_view_t* result_offsets = nullptr;

        result_srcs    = hipgraph_sample_result_get_sources(result);
        result_dsts    = hipgraph_sample_result_get_destinations(result);
        result_edge_id = hipgraph_sample_result_get_edge_id(result);
        result_weights = hipgraph_sample_result_get_edge_weight(result);
        result_hops    = hipgraph_sample_result_get_hop(result);
        result_offsets = hipgraph_sample_result_get_offsets(result);

        size_t result_size         = hipgraph_type_erased_device_array_view_size(result_srcs);
        size_t result_offsets_size = hipgraph_type_erased_device_array_view_size(result_offsets);

        vertex_t h_srcs[result_size];
        vertex_t h_dsts[result_size];
        edge_t   h_edge_id[result_size];
        weight_t h_weight[result_size];
        int32_t  h_hops[result_size];
        size_t   h_result_offsets[result_offsets_size];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_srcs, result_srcs, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_dsts, result_dsts, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_edge_id, result_edge_id, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_weight, result_weights, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_result_offsets, result_offsets, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        for(int k = 0; k < result_offsets_size - 1; k += fan_out_size)
        {
            for(int h = 0; h < fan_out_size; ++h)
            {
                int hop_start = h_result_offsets[k + h];
                int hop_end   = h_result_offsets[k + h + 1];
                for(int i = hop_start; i < hop_end; ++i)
                {
                    h_hops[i] = h;
                }
            }
        }

        size_t num_local_labels = (result_offsets_size - 1) / fan_out_size;

        for(int k = 0; k < num_local_labels + 1; ++k)
        {
            h_result_offsets[k] = h_result_offsets[k * fan_out_size];
        }
        result_offsets_size = num_local_labels + 1;

        //  NOTE:  The C++ tester does a more thorough validation.  For our purposes
        //  here we will do a simpler validation, merely checking that all edges
        //  are actually part of the graph
        weight_t M_w[num_vertices][num_vertices];
        edge_t   M_edge_id[num_vertices][num_vertices];

        for(int i = 0; i < num_vertices; ++i)
            for(int j = 0; j < num_vertices; ++j)
            {
                M_w[i][j]       = 0.0;
                M_edge_id[i][j] = -1;
            }

        for(int i = 0; i < num_edges; ++i)
        {
            M_w[src[i]][dst[i]]       = wgt[i];
            M_edge_id[src[i]][dst[i]] = edge_ids[i];
        }

        for(int i = 0; (i < result_size) && (test_ret_value == 0); ++i)
        {
            EXPECT_EQ(M_w[h_srcs[i]][h_dsts[i]], h_weight[i])
                << "uniform_neighbor_sample got edge that doesn't exist";
            EXPECT_EQ(M_edge_id[h_srcs[i]][h_dsts[i]], h_edge_id[i])
                << "uniform_neighbor_sample got edge that doesn't exist";
        }

        EXPECT_EQ(result_offsets_size, expected_size[rank]) << "incorrect number of results";

        for(int i = 0; i < (result_offsets_size - 1) && (test_ret_value == 0); ++i)
        {
            for(int j = h_result_offsets[i];
                j < (h_result_offsets[i + 1] - 1) && (test_ret_value == 0);
                ++j)
            {
                EXPECT_LE(h_hops[j], h_hops[j + 1]) << "Results not sorted by hop id";
            }
        }

        hipgraph_sample_result_free(result);
#endif

        hipgraph_sg_graph_free(graph);
        hipgraph_error_free(ret_error);
    }

    int test_uniform_neighbor_sample_dedupe_sources(const hipgraph_resource_handle_t* p_handle)
    {
        hipgraph_data_type_id_t vertex_tid    = HIPGRAPH_INT32;
        hipgraph_data_type_id_t edge_tid      = HIPGRAPH_INT32;
        hipgraph_data_type_id_t weight_tid    = HIPGRAPH_FLOAT32;
        hipgraph_data_type_id_t edge_id_tid   = HIPGRAPH_INT32;
        hipgraph_data_type_id_t edge_type_tid = HIPGRAPH_INT32;

        size_t num_edges    = 9;
        size_t num_vertices = 6;
        size_t fan_out_size = 3;
        size_t num_starts   = 2;

        vertex_t src[]          = {0, 0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]          = {1, 3, 3, 4, 0, 1, 3, 5, 5};
        edge_t   edge_ids[]     = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        weight_t weight[]       = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
        int32_t  edge_types[]   = {8, 7, 6, 5, 4, 3, 2, 1, 0};
        vertex_t start[]        = {2, 3};
        int      start_labels[] = {6, 12};
        int      fan_out[]      = {-1, -1, -1};

        int                   test_ret_value = 0;
        hipgraph_error_code_t ret_code       = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error      = nullptr;

        hipgraph_bool_t                   with_replacement       = HIPGRAPH_FALSE;
        hipgraph_bool_t                   return_hops            = HIPGRAPH_TRUE;
        hipgraph_prior_sources_behavior_t prior_sources_behavior = DEFAULT;
        hipgraph_bool_t                   dedupe_sources         = HIPGRAPH_TRUE;

        return generic_uniform_neighbor_sample_test(p_handle,
                                                    src,
                                                    dst,
                                                    weight,
                                                    edge_ids,
                                                    edge_types,
                                                    num_vertices,
                                                    num_edges,
                                                    start,
                                                    start_labels,
                                                    num_starts,
                                                    fan_out,
                                                    fan_out_size,
                                                    with_replacement,
                                                    return_hops,
                                                    prior_sources_behavior,
                                                    dedupe_sources);
    }

    int test_uniform_neighbor_sample_unique_sources(const hipgraph_resource_handle_t* p_handle)
    {
        hipgraph_data_type_id_t vertex_tid    = HIPGRAPH_INT32;
        hipgraph_data_type_id_t edge_tid      = HIPGRAPH_INT32;
        hipgraph_data_type_id_t weight_tid    = HIPGRAPH_FLOAT32;
        hipgraph_data_type_id_t edge_id_tid   = HIPGRAPH_INT32;
        hipgraph_data_type_id_t edge_type_tid = HIPGRAPH_INT32;

        size_t num_edges    = 9;
        size_t num_vertices = 6;
        size_t fan_out_size = 3;
        size_t num_starts   = 2;

        vertex_t src[]          = {0, 0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]          = {1, 2, 3, 4, 0, 1, 3, 5, 5};
        edge_t   edge_ids[]     = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        weight_t weight[]       = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
        int32_t  edge_types[]   = {8, 7, 6, 5, 4, 3, 2, 1, 0};
        vertex_t start[]        = {2, 3};
        int      start_labels[] = {6, 12};
        int      fan_out[]      = {-1, -1, -1};

        int                   test_ret_value = 0;
        hipgraph_error_code_t ret_code       = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error      = nullptr;

        hipgraph_bool_t                   with_replacement       = HIPGRAPH_FALSE;
        hipgraph_bool_t                   return_hops            = HIPGRAPH_TRUE;
        hipgraph_prior_sources_behavior_t prior_sources_behavior = EXCLUDE;
        hipgraph_bool_t                   dedupe_sources         = HIPGRAPH_FALSE;

        return generic_uniform_neighbor_sample_test(p_handle,
                                                    src,
                                                    dst,
                                                    weight,
                                                    edge_ids,
                                                    edge_types,
                                                    num_vertices,
                                                    num_edges,
                                                    start,
                                                    start_labels,
                                                    num_starts,
                                                    fan_out,
                                                    fan_out_size,
                                                    with_replacement,
                                                    return_hops,
                                                    prior_sources_behavior,
                                                    dedupe_sources);
    }

    int test_uniform_neighbor_sample_carry_over_sources(const hipgraph_resource_handle_t* p_handle)
    {
        hipgraph_data_type_id_t vertex_tid    = HIPGRAPH_INT32;
        hipgraph_data_type_id_t edge_tid      = HIPGRAPH_INT32;
        hipgraph_data_type_id_t weight_tid    = HIPGRAPH_FLOAT32;
        hipgraph_data_type_id_t edge_id_tid   = HIPGRAPH_INT32;
        hipgraph_data_type_id_t edge_type_tid = HIPGRAPH_INT32;

        size_t num_edges    = 9;
        size_t num_vertices = 6;
        size_t fan_out_size = 3;
        size_t num_starts   = 2;

        vertex_t src[]          = {0, 0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]          = {1, 2, 3, 4, 0, 1, 3, 5, 5};
        edge_t   edge_ids[]     = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        weight_t weight[]       = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
        int32_t  edge_types[]   = {8, 7, 6, 5, 4, 3, 2, 1, 0};
        vertex_t start[]        = {2, 3};
        int      start_labels[] = {6, 12};
        int      fan_out[]      = {-1, -1, -1};

        int                   test_ret_value = 0;
        hipgraph_error_code_t ret_code       = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error      = nullptr;

        hipgraph_bool_t                   with_replacement       = HIPGRAPH_FALSE;
        hipgraph_bool_t                   return_hops            = HIPGRAPH_TRUE;
        hipgraph_prior_sources_behavior_t prior_sources_behavior = CARRY_OVER;
        hipgraph_bool_t                   dedupe_sources         = HIPGRAPH_FALSE;

        return generic_uniform_neighbor_sample_test(p_handle,
                                                    src,
                                                    dst,
                                                    weight,
                                                    edge_ids,
                                                    edge_types,
                                                    num_vertices,
                                                    num_edges,
                                                    start,
                                                    start_labels,
                                                    num_starts,
                                                    fan_out,
                                                    fan_out_size,
                                                    with_replacement,
                                                    return_hops,
                                                    prior_sources_behavior,
                                                    dedupe_sources);
    }

} // namespace
