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

#include "hipgraph_c/sampling_algorithms.h"
#include "test_utils.h" /* RUN_TEST */

#include "hipgraph_c/algorithms.h"
#include "hipgraph_c/graph.h"

#include <cmath>
#include <stdbool.h>
#include <stdlib.h>

using vertex_t = int32_t;
using edge_t   = int32_t;
using weight_t = float;

hipgraph_data_type_id_t vertex_tid    = HIPGRAPH_INT32;
hipgraph_data_type_id_t edge_tid      = HIPGRAPH_INT32;
hipgraph_data_type_id_t weight_tid    = HIPGRAPH_FLOAT32;
hipgraph_data_type_id_t edge_id_tid   = HIPGRAPH_INT32;
hipgraph_data_type_id_t edge_type_tid = HIPGRAPH_INT32;

#define NO_HIPGRAPH_OPS TRUE

namespace
{
    using namespace hipGRAPH::testing;

#ifndef NO_HIPGRAPH_OPS
    int vertex_id_compare_function(const void* a, const void* b)
    {
        if(*((vertex_t*)a) < *((vertex_t*)b))
            return -1;
        else if(*((vertex_t*)a) > *((vertex_t*)b))
            return 1;
        else
            return 0;
    }
#endif

    void generic_uniform_neighbor_sample_test(
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
        size_t                            num_start_labels,
        int*                              fan_out,
        size_t                            fan_out_size,
        hipgraph_bool_t                   with_replacement,
        hipgraph_bool_t                   return_hops,
        hipgraph_prior_sources_behavior_t prior_sources_behavior,
        hipgraph_bool_t                   dedupe_sources,
        hipgraph_bool_t                   renumber_results)
    {
        // Create graph
        hipgraph_error_code_t     ret_code  = HIPGRAPH_SUCCESS;
        hipgraph_error_t*         ret_error = nullptr;
        hipgraph_graph_t*         graph     = nullptr;
        hipgraph_sample_result_t* result    = nullptr;

        hipgraph_resource_handle_t* p_handle = nullptr;
        p_handle                             = hipgraph_create_resource_handle(nullptr);
        ASSERT_NE(p_handle, nullptr) << "resource handle creation failed.";

        create_sg_test_graph(p_handle,
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

        hipgraph_type_erased_device_array_t*      d_start             = nullptr;
        hipgraph_type_erased_device_array_view_t* d_start_view        = nullptr;
        hipgraph_type_erased_device_array_t*      d_start_labels      = nullptr;
        hipgraph_type_erased_device_array_view_t* d_start_labels_view = nullptr;
        hipgraph_type_erased_host_array_view_t*   h_fan_out_view      = nullptr;

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_start_vertices, HIPGRAPH_INT32, &d_start, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "d_start create failed: " << hipgraph_error_message(ret_error);

        d_start_view = hipgraph_type_erased_device_array_view(d_start);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, d_start_view, (hipgraph_byte_t*)h_start, &ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_start_vertices, HIPGRAPH_INT32, &d_start_labels, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "d_start_labels create failed: " << hipgraph_error_message(ret_error);

        d_start_labels_view = hipgraph_type_erased_device_array_view(d_start_labels);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, d_start_labels_view, (hipgraph_byte_t*)h_start_labels, &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "start_labels copy_from_host failed: " << hipgraph_error_message(ret_error);

        h_fan_out_view
            = hipgraph_type_erased_host_array_view_create(fan_out, fan_out_size, HIPGRAPH_INT32);

        hipgraph_rng_state_t* rng_state;
        ret_code = hipgraph_rng_state_create(p_handle, 0, &rng_state, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "rng_state create failed: " << hipgraph_error_message(ret_error);

        hipgraph_sampling_options_t* sampling_options;

        ret_code = hipgraph_sampling_options_create(&sampling_options, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "sampling_options create failed: " << hipgraph_error_message(ret_error);

        hipgraph_sampling_set_with_replacement(sampling_options, with_replacement);
        hipgraph_sampling_set_return_hops(sampling_options, return_hops);
        hipgraph_sampling_set_prior_sources_behavior(sampling_options, prior_sources_behavior);
        hipgraph_sampling_set_dedupe_sources(sampling_options, dedupe_sources);
        hipgraph_sampling_set_renumber_results(sampling_options, renumber_results);

        ret_code = hipgraph_uniform_neighbor_sample(p_handle,
                                                    graph,
                                                    d_start_view,
                                                    d_start_labels_view,
                                                    nullptr,
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
            << "uniform_neighbor_sample failed: " << hipgraph_error_message(ret_error);

        hipgraph_sampling_options_free(sampling_options);

        hipgraph_type_erased_device_array_view_t* result_srcs;
        hipgraph_type_erased_device_array_view_t* result_dsts;
        hipgraph_type_erased_device_array_view_t* result_edge_id;
        hipgraph_type_erased_device_array_view_t* result_weights;
        hipgraph_type_erased_device_array_view_t* result_edge_types;
        hipgraph_type_erased_device_array_view_t* result_hops;
        hipgraph_type_erased_device_array_view_t* result_offsets;
        hipgraph_type_erased_device_array_view_t* result_labels;
        hipgraph_type_erased_device_array_view_t* result_renumber_map;
        hipgraph_type_erased_device_array_view_t* result_renumber_map_offsets;

        result_srcs                 = hipgraph_sample_result_get_sources(result);
        result_dsts                 = hipgraph_sample_result_get_destinations(result);
        result_edge_id              = hipgraph_sample_result_get_edge_id(result);
        result_weights              = hipgraph_sample_result_get_edge_weight(result);
        result_edge_types           = hipgraph_sample_result_get_edge_type(result);
        result_hops                 = hipgraph_sample_result_get_hop(result);
        result_hops                 = hipgraph_sample_result_get_hop(result);
        result_offsets              = hipgraph_sample_result_get_offsets(result);
        result_labels               = hipgraph_sample_result_get_start_labels(result);
        result_renumber_map         = hipgraph_sample_result_get_renumber_map(result);
        result_renumber_map_offsets = hipgraph_sample_result_get_renumber_map_offsets(result);

        size_t result_size         = hipgraph_type_erased_device_array_view_size(result_srcs);
        size_t result_offsets_size = hipgraph_type_erased_device_array_view_size(result_offsets);
        size_t renumber_map_size   = 0;

        if(renumber_results)
        {
            renumber_map_size = hipgraph_type_erased_device_array_view_size(result_renumber_map);
        }

        vertex_t h_result_srcs[result_size];
        vertex_t h_result_dsts[result_size];
        edge_t   h_result_edge_id[result_size];
        weight_t h_result_weight[result_size];
        int32_t  h_result_edge_types[result_size];
        int32_t  h_result_hops[result_size];
        size_t   h_result_offsets[result_offsets_size];
        int      h_result_labels[num_start_labels];
        vertex_t h_renumber_map[renumber_map_size];
        size_t   h_renumber_map_offsets[result_offsets_size];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            handle, (hipgraph_byte_t*)h_result_srcs, result_srcs, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            handle, (hipgraph_byte_t*)h_result_dsts, result_dsts, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            handle, (hipgraph_byte_t*)h_result_edge_id, result_edge_id, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            handle, (hipgraph_byte_t*)h_result_weight, result_weights, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            handle, (hipgraph_byte_t*)h_result_edge_types, result_edge_types, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        EXPECT_EQ(result_hops, nullptr) << "hops was not empty";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            handle, (hipgraph_byte_t*)h_result_offsets, result_offsets, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            handle, (hipgraph_byte_t*)h_result_labels, result_labels, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        for(int k = 0; k < result_offsets_size - 1; k += fan_out_size)
        {
            for(int h = 0; h < fan_out_size; ++h)
            {
                int hop_start = h_result_offsets[k + h];
                int hop_end   = h_result_offsets[k + h + 1];
                for(int i = hop_start; i < hop_end; ++i)
                {
                    h_result_hops[i] = h;
                }
            }
        }

        for(int k = 0; k < num_start_labels + 1; ++k)
        {
            h_result_offsets[k] = h_result_offsets[k * fan_out_size];
        }
        result_offsets_size = num_start_labels + 1;

        if(renumber_results)
        {
            ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
                handle, (hipgraph_byte_t*)h_renumber_map, result_renumber_map, &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
                << "copy_to_host failed: " << hipgraph_error_message(ret_error);

            ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
                handle,
                (hipgraph_byte_t*)h_renumber_map_offsets,
                result_renumber_map_offsets,
                &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
                << "copy_to_host failed: " << hipgraph_error_message(ret_error);
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
            M_w[h_src[i]][h_dst[i]]         = h_wgt[i];
            M_edge_id[h_src[i]][h_dst[i]]   = h_edge_ids[i];
            M_edge_type[h_src[i]][h_dst[i]] = h_edge_types[i];
        }

        if(renumber_results)
        {
            for(int label_id = 0; label_id < (result_offsets_size - 1); ++label_id)
            {
                for(size_t i = h_result_offsets[label_id]; i < h_result_offsets[label_id + 1]; ++i)
                {
                    vertex_t src
                        = h_renumber_map[h_renumber_map_offsets[label_id] + h_result_srcs[i]];
                    vertex_t dst
                        = h_renumber_map[h_renumber_map_offsets[label_id] + h_result_dsts[i]];

                    EXPECT_EQ(M_w[src][dst], h_result_weight[i])
                        << "uniform_neighbor_sample got edge that doesn't exist";
                    EXPECT_EQ(M_edge_id[src][dst], h_result_edge_id[i])
                        << "uniform_neighbor_sample got edge that doesn't exist";
                    EXPECT_EQ(M_edge_type[src][dst], h_result_edge_types[i])
                        << "uniform_neighbor_sample got edge that doesn't exist";
                }
            }
        }
        else
        {
            for(int i = 0; i < result_size; ++i)
            {
                EXPECT_EQ(M_w[h_result_srcs[i]][h_result_dsts[i]], h_result_weight[i])
                    << "uniform_neighbor_sample got edge that doesn't exist";
                EXPECT_EQ(M_edge_id[h_result_srcs[i]][h_result_dsts[i]], h_result_edge_id[i])
                    << "uniform_neighbor_sample got edge that doesn't exist";
                EXPECT_EQ(M_edge_type[h_result_srcs[i]][h_result_dsts[i]], h_result_edge_types[i])
                    << "uniform_neighbor_sample got edge that doesn't exist";
            }
        }

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

        for(int label_id = 0; label_id < (result_offsets_size - 1); ++label_id)
        {
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

            if(renumber_results)
            {
                size_t num_vertex_ids
                    = 2 * (h_result_offsets[label_id + 1] - h_result_offsets[label_id]);
                vertex_t vertex_ids[num_vertex_ids];

                for(size_t i = 0; i < (h_result_offsets[label_id + 1] - h_result_offsets[label_id]);
                    ++i)
                {
                    vertex_ids[2 * i]     = h_result_srcs[h_result_offsets[label_id] + i];
                    vertex_ids[2 * i + 1] = h_result_dsts[h_result_offsets[label_id] + i];
                }

                qsort(vertex_ids, num_vertex_ids, sizeof(vertex_t), vertex_id_compare_function);

                vertex_t current_v = 0;
                for(size_t i = 0; i < num_vertex_ids; ++i)
                {
                    if(vertex_ids[i] == current_v)
                        ++current_v;
                    else
                        EXPECT_EQ(vertex_ids[i], (current_v - 1))
                            << "vertices are not properly renumbered";
                }
            }

            for(int hop = 0; hop < fan_out_size; ++hop)
            {
                if(prior_sources_behavior == HIPGRAPH_CARRY_OVER)
                {
                    destinations_size = sources_size;
                    for(size_t i = 0; i < sources_size; ++i)
                    {
                        check_destinations[i] = check_sources[i];
                    }
                }

                for(size_t i = h_result_offsets[label_id]; i < h_result_offsets[label_id + 1]; ++i)
                {
                    if(h_result_hops[i] == hop)
                    {

                        bool found = false;
                        for(size_t j = 0; (!found) && (j < sources_size); ++j)
                        {
                            found = renumber_results
                                        ? (h_renumber_map[h_renumber_map_offsets[label_id]
                                                          + h_result_srcs[i]]
                                           == check_sources[j])
                                        : (h_result_srcs[i] == check_sources[j]);
                        }

                        EXPECT_TRUE(found)
                            << "encountered source vertex that was not part of previous frontier";
                    }

                    if(prior_sources_behavior, HIPGRAPH_CARRY_OVER)
                    {
                        // Make sure destination isn't already in the source list
                        bool found = false;
                        for(size_t j = 0; (!found) && (j < destinations_size); ++j)
                        {
                            found = renumber_results
                                        ? (h_renumber_map[h_renumber_map_offsets[label_id]
                                                          + h_result_dsts[i]]
                                           == check_destinations[j])
                                        : (h_result_dsts[i] == check_destinations[j]);
                        }

                        if(!found)
                        {
                            check_destinations[destinations_size]
                                = renumber_results ? h_renumber_map[h_renumber_map_offsets[label_id]
                                                                    + h_result_dsts[i]]
                                                   : h_result_dsts[i];
                            ++destinations_size;
                        }
                    }
                    else
                    {
                        check_destinations[destinations_size]
                            = renumber_results ? h_renumber_map[h_renumber_map_offsets[label_id]
                                                                + h_result_dsts[i]]
                                               : h_result_dsts[i];
                        ++destinations_size;
                    }
                }

                vertex_t* tmp      = check_sources;
                check_sources      = check_destinations;
                check_destinations = tmp;
                sources_size       = destinations_size;
                destinations_size  = 0;
            }

            if(prior_sources_behavior == HIPGRAPH_EXCLUDE)
            {
                // Make sure vertex v only appears as source in the first hop after it is encountered
                for(size_t i = h_result_offsets[label_id]; i < h_result_offsets[label_id + 1]; ++i)
                {
                    for(size_t j = i + 1; j < h_result_offsets[label_id + 1]; ++j)
                    {
                        if(h_result_srcs[i] == h_result_srcs[j])
                        {
                            EXPECT_EQ(h_result_hops[i], h_result_hops[j])
                                << "source vertex should not have been used in diferent hops";
                        }
                    }
                }
            }

            if(dedupe_sources)
            {
                // Make sure vertex v only appears as source once for each edge after it appears as destination
                // Externally test this by verifying that vertex v only appears in <= hop size/degree
                for(size_t i = h_result_offsets[label_id]; i < h_result_offsets[label_id + 1]; ++i)
                {
                    if(h_result_hops[i] > 0)
                    {
                        size_t num_occurrences = 1;
                        for(size_t j = i + 1; j < h_result_offsets[label_id + 1]; ++j)
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

        hipgraph_sample_result_free(result);
#endif

        hipgraph_sg_graph_free(graph);
        hipgraph_error_free(ret_error);
    }

    TEST(RocGraphOpsTest, UniformNeighborSampleClean)
    {
        // hipgraph_data_type_id_t vertex_tid    = HIPGRAPH_INT32;
        // hipgraph_data_type_id_t edge_tid      = HIPGRAPH_INT32;
        // hipgraph_data_type_id_t weight_tid    = HIPGRAPH_FLOAT32;
        // hipgraph_data_type_id_t edge_id_tid   = HIPGRAPH_INT32;
        // hipgraph_data_type_id_t edge_type_tid = HIPGRAPH_INT32;

        size_t num_edges        = 9;
        size_t num_vertices     = 6;
        size_t fan_out_size     = 3;
        size_t num_starts       = 2;
        size_t num_start_labels = 2;

        vertex_t src[]          = {0, 0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]          = {1, 3, 3, 4, 0, 1, 3, 5, 5};
        edge_t   edge_ids[]     = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        weight_t weight[]       = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
        int32_t  edge_types[]   = {8, 7, 6, 5, 4, 3, 2, 1, 0};
        vertex_t start[]        = {2, 3};
        int      start_labels[] = {6, 12};
        int      fan_out[]      = {-1, -1, -1};

        hipgraph_bool_t                   with_replacement       = HIPGRAPH_FALSE;
        hipgraph_bool_t                   return_hops            = HIPGRAPH_TRUE;
        hipgraph_prior_sources_behavior_t prior_sources_behavior = HIPGRAPH_DEFAULT;
        hipgraph_bool_t                   dedupe_sources         = HIPGRAPH_FALSE;
        hipgraph_bool_t                   renumber_results       = HIPGRAPH_FALSE;

        generic_uniform_neighbor_sample_test(src,
                                             dst,
                                             weight,
                                             edge_ids,
                                             edge_types,
                                             num_vertices,
                                             num_edges,
                                             start,
                                             start_labels,
                                             num_starts,
                                             num_start_labels,
                                             fan_out,
                                             fan_out_size,
                                             with_replacement,
                                             return_hops,
                                             prior_sources_behavior,
                                             dedupe_sources,
                                             renumber_results);
    }

    TEST(RocGraphOpsTest, UniformNeighborSampleDedupeSources)
    {
        // hipgraph_data_type_id_t vertex_tid    = HIPGRAPH_INT32;
        // hipgraph_data_type_id_t edge_tid      = HIPGRAPH_INT32;
        // hipgraph_data_type_id_t weight_tid    = HIPGRAPH_FLOAT32;
        // hipgraph_data_type_id_t edge_id_tid   = HIPGRAPH_INT32;
        // hipgraph_data_type_id_t edge_type_tid = HIPGRAPH_INT32;

        size_t num_edges        = 9;
        size_t num_vertices     = 6;
        size_t fan_out_size     = 3;
        size_t num_starts       = 2;
        size_t num_start_labels = 2;

        vertex_t src[]          = {0, 0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]          = {1, 3, 3, 4, 0, 1, 3, 5, 5};
        edge_t   edge_ids[]     = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        weight_t weight[]       = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
        int32_t  edge_types[]   = {8, 7, 6, 5, 4, 3, 2, 1, 0};
        vertex_t start[]        = {2, 3};
        int      start_labels[] = {6, 12};
        int      fan_out[]      = {-1, -1, -1};

        hipgraph_bool_t                   with_replacement       = HIPGRAPH_FALSE;
        hipgraph_bool_t                   return_hops            = HIPGRAPH_TRUE;
        hipgraph_prior_sources_behavior_t prior_sources_behavior = HIPGRAPH_DEFAULT;
        hipgraph_bool_t                   dedupe_sources         = HIPGRAPH_TRUE;
        hipgraph_bool_t                   renumber_results       = HIPGRAPH_FALSE;

        generic_uniform_neighbor_sample_test(src,
                                             dst,
                                             weight,
                                             edge_ids,
                                             edge_types,
                                             num_vertices,
                                             num_edges,
                                             start,
                                             start_labels,
                                             num_starts,
                                             num_start_labels,
                                             fan_out,
                                             fan_out_size,
                                             with_replacement,
                                             return_hops,
                                             prior_sources_behavior,
                                             dedupe_sources,
                                             renumber_results);
    }

    TEST(RocGraphOpsTest, UniformNeighborSampleUniqueSources)
    {
        // hipgraph_data_type_id_t vertex_tid    = HIPGRAPH_INT32;
        // hipgraph_data_type_id_t edge_tid      = HIPGRAPH_INT32;
        // hipgraph_data_type_id_t weight_tid    = HIPGRAPH_FLOAT32;
        // hipgraph_data_type_id_t edge_id_tid   = HIPGRAPH_INT32;
        // hipgraph_data_type_id_t edge_type_tid = HIPGRAPH_INT32;

        size_t num_edges        = 9;
        size_t num_vertices     = 6;
        size_t fan_out_size     = 3;
        size_t num_starts       = 2;
        size_t num_start_labels = 2;

        vertex_t src[]          = {0, 0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]          = {1, 2, 3, 4, 0, 1, 3, 5, 5};
        edge_t   edge_ids[]     = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        weight_t weight[]       = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
        int32_t  edge_types[]   = {8, 7, 6, 5, 4, 3, 2, 1, 0};
        vertex_t start[]        = {2, 3};
        int      start_labels[] = {6, 12};
        int      fan_out[]      = {-1, -1, -1};

        hipgraph_bool_t                   with_replacement       = HIPGRAPH_FALSE;
        hipgraph_bool_t                   return_hops            = HIPGRAPH_TRUE;
        hipgraph_prior_sources_behavior_t prior_sources_behavior = HIPGRAPH_EXCLUDE;
        hipgraph_bool_t                   dedupe_sources         = HIPGRAPH_FALSE;
        hipgraph_bool_t                   renumber_results       = HIPGRAPH_FALSE;

        generic_uniform_neighbor_sample_test(src,
                                             dst,
                                             weight,
                                             edge_ids,
                                             edge_types,
                                             num_vertices,
                                             num_edges,
                                             start,
                                             start_labels,
                                             num_starts,
                                             num_start_labels,
                                             fan_out,
                                             fan_out_size,
                                             with_replacement,
                                             return_hops,
                                             prior_sources_behavior,
                                             dedupe_sources,
                                             renumber_results);
    }

    TEST(RocGraphOpsTest, UniformNeighborSampleCarryOverSources)
    {
        // hipgraph_data_type_id_t vertex_tid    = HIPGRAPH_INT32;
        // hipgraph_data_type_id_t edge_tid      = HIPGRAPH_INT32;
        // hipgraph_data_type_id_t weight_tid    = HIPGRAPH_FLOAT32;
        // hipgraph_data_type_id_t edge_id_tid   = HIPGRAPH_INT32;
        // hipgraph_data_type_id_t edge_type_tid = HIPGRAPH_INT32;

        size_t num_edges        = 9;
        size_t num_vertices     = 6;
        size_t fan_out_size     = 3;
        size_t num_starts       = 2;
        size_t num_start_labels = 2;

        vertex_t src[]          = {0, 0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]          = {1, 2, 3, 4, 0, 1, 3, 5, 5};
        edge_t   edge_ids[]     = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        weight_t weight[]       = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
        int32_t  edge_types[]   = {8, 7, 6, 5, 4, 3, 2, 1, 0};
        vertex_t start[]        = {2, 3};
        int      start_labels[] = {6, 12};
        int      fan_out[]      = {-1, -1, -1};

        hipgraph_bool_t                   with_replacement       = HIPGRAPH_FALSE;
        hipgraph_bool_t                   return_hops            = HIPGRAPH_TRUE;
        hipgraph_prior_sources_behavior_t prior_sources_behavior = HIPGRAPH_CARRY_OVER;
        hipgraph_bool_t                   dedupe_sources         = HIPGRAPH_FALSE;
        hipgraph_bool_t                   renumber_results       = HIPGRAPH_FALSE;

        generic_uniform_neighbor_sample_test(src,
                                             dst,
                                             weight,
                                             edge_ids,
                                             edge_types,
                                             num_vertices,
                                             num_edges,
                                             start,
                                             start_labels,
                                             num_starts,
                                             num_start_labels,
                                             fan_out,
                                             fan_out_size,
                                             with_replacement,
                                             return_hops,
                                             prior_sources_behavior,
                                             dedupe_sources,
                                             renumber_results);
    }

    TEST(RocGraphOpsTest, UniformNeighborSampleRenumberResults)
    {
        // hipgraph_data_type_id_t vertex_tid    = HIPGRAPH_INT32;
        // hipgraph_data_type_id_t edge_tid      = HIPGRAPH_INT32;
        // hipgraph_data_type_id_t weight_tid    = HIPGRAPH_FLOAT32;
        // hipgraph_data_type_id_t edge_id_tid   = HIPGRAPH_INT32;
        // hipgraph_data_type_id_t edge_type_tid = HIPGRAPH_INT32;

        size_t num_edges        = 9;
        size_t num_vertices     = 6;
        size_t fan_out_size     = 3;
        size_t num_starts       = 2;
        size_t num_start_labels = 2;

        vertex_t src[]          = {0, 0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]          = {1, 2, 3, 4, 0, 1, 3, 5, 5};
        edge_t   edge_ids[]     = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        weight_t weight[]       = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
        int32_t  edge_types[]   = {8, 7, 6, 5, 4, 3, 2, 1, 0};
        vertex_t start[]        = {2, 3};
        int      start_labels[] = {6, 12};
        int      fan_out[]      = {-1, -1, -1};

        hipgraph_bool_t                   with_replacement       = HIPGRAPH_FALSE;
        hipgraph_bool_t                   return_hops            = HIPGRAPH_TRUE;
        hipgraph_prior_sources_behavior_t prior_sources_behavior = HIPGRAPH_DEFAULT;
        hipgraph_bool_t                   dedupe_sources         = HIPGRAPH_FALSE;
        hipgraph_bool_t                   renumber_results       = HIPGRAPH_TRUE;

        generic_uniform_neighbor_sample_test(src,
                                             dst,
                                             weight,
                                             edge_ids,
                                             edge_types,
                                             num_vertices,
                                             num_edges,
                                             start,
                                             start_labels,
                                             num_starts,
                                             num_start_labels,
                                             fan_out,
                                             fan_out_size,
                                             with_replacement,
                                             return_hops,
                                             prior_sources_behavior,
                                             dedupe_sources,
                                             renumber_results);
    }

    TEST(RocGraphOpsTest, UniformNeighborSampleWithLabels)
    {
        hipgraph_data_type_id_t vertex_tid    = HIPGRAPH_INT32;
        hipgraph_data_type_id_t edge_tid      = HIPGRAPH_INT32;
        hipgraph_data_type_id_t weight_tid    = HIPGRAPH_FLOAT32;
        hipgraph_data_type_id_t edge_id_tid   = HIPGRAPH_INT32;
        hipgraph_data_type_id_t edge_type_tid = HIPGRAPH_INT32;

        size_t num_edges = 8;

        size_t num_starts = 2;

        vertex_t src[]          = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]          = {1, 3, 4, 0, 1, 3, 5, 5};
        edge_t   edge_ids[]     = {0, 1, 2, 3, 4, 5, 6, 7};
        weight_t weight[]       = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
        int32_t  edge_types[]   = {7, 6, 5, 4, 3, 2, 1, 0};
        vertex_t start[]        = {2, 3};
        size_t   start_labels[] = {6, 12};
        int      fan_out[]      = {-1};

        // Create graph
        hipgraph_error_code_t     ret_code  = HIPGRAPH_SUCCESS;
        hipgraph_error_t*         ret_error = nullptr;
        hipgraph_graph_t*         graph     = nullptr;
        hipgraph_sample_result_t* result    = nullptr;

        hipgraph_bool_t                   with_replacement       = HIPGRAPH_TRUE;
        hipgraph_bool_t                   return_hops            = HIPGRAPH_TRUE;
        hipgraph_prior_sources_behavior_t prior_sources_behavior = HIPGRAPH_DEFAULT;
        hipgraph_bool_t                   dedupe_sources         = HIPGRAPH_FALSE;
        hipgraph_bool_t                   renumber_results       = HIPGRAPH_FALSE;
        hipgraph_compression_type_t       compression            = HIPGRAPH_COO;
        hipgraph_bool_t                   compress_per_hop       = HIPGRAPH_FALSE;

        hipgraph_resource_handle_t* p_handle = nullptr;
        p_handle                             = hipgraph_create_resource_handle(nullptr);
        ASSERT_NE(p_handle, nullptr) << "resource handle creation failed.";

        create_sg_test_graph(p_handle,
                             vertex_tid,
                             edge_tid,
                             src,
                             dst,
                             weight_tid,
                             weight,
                             edge_type_tid,
                             edge_types,
                             edge_id_tid,
                             edge_ids,
                             num_edges,
                             HIPGRAPH_FALSE,
                             HIPGRAPH_TRUE,
                             HIPGRAPH_FALSE,
                             HIPGRAPH_FALSE,
                             &graph,
                             &ret_error);

        hipgraph_type_erased_device_array_t*      d_start             = nullptr;
        hipgraph_type_erased_device_array_view_t* d_start_view        = nullptr;
        hipgraph_type_erased_device_array_t*      d_start_labels      = nullptr;
        hipgraph_type_erased_device_array_view_t* d_start_labels_view = nullptr;
        hipgraph_type_erased_host_array_view_t*   h_fan_out_view      = nullptr;

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_starts, HIPGRAPH_INT32, &d_start, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "d_start create failed: " << hipgraph_error_message(ret_error);

        d_start_view = hipgraph_type_erased_device_array_view(d_start);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, d_start_view, (hipgraph_byte_t*)start, &ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_starts, HIPGRAPH_INT32, &d_start_labels, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "d_start_labels create failed: " << hipgraph_error_message(ret_error);

        d_start_labels_view = hipgraph_type_erased_device_array_view(d_start_labels);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, d_start_labels_view, (hipgraph_byte_t*)start_labels, &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "start_labels copy_from_host failed: " << hipgraph_error_message(ret_error);

        h_fan_out_view = hipgraph_type_erased_host_array_view_create(fan_out, 1, HIPGRAPH_INT32);

        hipgraph_rng_state_t* rng_state;
        ret_code = hipgraph_rng_state_create(p_handle, 0, &rng_state, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "rng_state create failed: " << hipgraph_error_message(ret_error);

        hipgraph_sampling_options_t* sampling_options;

        ret_code = hipgraph_sampling_options_create(&sampling_options, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "sampling_options create failed: " << hipgraph_error_message(ret_error);

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
                                                    nullptr,
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
            << "uniform_neighbor_sample failed: " << hipgraph_error_message(ret_error);

        size_t                                    num_vertices = 6;
        hipgraph_type_erased_device_array_view_t* result_srcs;
        hipgraph_type_erased_device_array_view_t* result_dsts;
        hipgraph_type_erased_device_array_view_t* result_edge_id;
        hipgraph_type_erased_device_array_view_t* result_weights;
        hipgraph_type_erased_device_array_view_t* result_edge_types;
        hipgraph_type_erased_device_array_view_t* result_hops;
        hipgraph_type_erased_device_array_view_t* result_offsets;

        result_srcs       = hipgraph_sample_result_get_sources(result);
        result_dsts       = hipgraph_sample_result_get_destinations(result);
        result_edge_id    = hipgraph_sample_result_get_edge_id(result);
        result_weights    = hipgraph_sample_result_get_edge_weight(result);
        result_edge_types = hipgraph_sample_result_get_edge_type(result);
        result_hops       = hipgraph_sample_result_get_hop(result);
        result_offsets    = hipgraph_sample_result_get_offsets(result);

        size_t result_size         = hipgraph_type_erased_device_array_view_size(result_srcs);
        size_t result_offsets_size = hipgraph_type_erased_device_array_view_size(result_offsets);

        vertex_t h_srcs[result_size];
        vertex_t h_dsts[result_size];
        edge_t   h_edge_id[result_size];
        weight_t h_weight[result_size];
        int32_t  h_edge_types[result_size];
        size_t   h_result_offsets[result_offsets_size];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            handle, (hipgraph_byte_t*)h_srcs, result_srcs, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            handle, (hipgraph_byte_t*)h_dsts, result_dsts, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            handle, (hipgraph_byte_t*)h_edge_id, result_edge_id, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            handle, (hipgraph_byte_t*)h_weight, result_weights, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            handle, (hipgraph_byte_t*)h_edge_types, result_edge_types, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        EXPECT_EQ(result_hops, nullptr) << "hops was not empty";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            handle, (hipgraph_byte_t*)h_result_offsets, result_offsets, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        //  NOTE:  The C++ tester does a more thorough validation.  For our purposes
        //  here we will do a simpler validation, merely checking that all edges
        //  are actually part of the graph
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
            M_w[src[i]][dst[i]]         = weight[i];
            M_edge_id[src[i]][dst[i]]   = edge_ids[i];
            M_edge_type[src[i]][dst[i]] = edge_types[i];
        }

        for(int i = 0; i < result_size; ++i)
        {
            EXPECT_EQ(M_w[h_srcs[i]][h_dsts[i]], h_weight[i])
                << "uniform_neighbor_sample got edge that doesn't exist";
            EXPECT_EQ(M_edge_id[h_srcs[i]][h_dsts[i]], h_edge_id[i])
                << "uniform_neighbor_sample got edge that doesn't exist";
            EXPECT_EQ(M_edge_type[h_srcs[i]][h_dsts[i]], h_edge_types[i])
                << "uniform_neighbor_sample got edge that doesn't exist";
        }

        hipgraph_sample_result_free(result);
        hipgraph_sampling_options_free(sampling_options);
#endif

        hipgraph_sg_graph_free(graph);
        hipgraph_error_free(ret_error);
    }

/* This method is not used anywhere */
#if 0

    void create_test_graph_with_edge_ids(const hipgraph_resource_handle_t* p_handle_ignored,
                                         vertex_t*                         h_src,
                                         vertex_t*                         h_dst,
                                         edge_t*                           h_ids,
                                         size_t                            num_edges,
                                         hipgraph_bool_t                   store_transposed,
                                         hipgraph_bool_t                   renumber,
                                         hipgraph_bool_t                   is_symmetric,
                                         hipgraph_graph_t**                p_graph,
                                         hipgraph_error_t**                ret_error)
    {
        hipgraph_error_code_t       ret_code;
        hipgraph_graph_properties_t properties;

        properties.is_symmetric  = is_symmetric;
        properties.is_multigraph = HIPGRAPH_FALSE;

        hipgraph_data_type_id_t vertex_tid = HIPGRAPH_INT32;
        hipgraph_data_type_id_t edge_tid   = HIPGRAPH_INT32;
        hipgraph_data_type_id_t weight_tid = HIPGRAPH_FLOAT32;

        hipgraph_type_erased_device_array_t*      src;
        hipgraph_type_erased_device_array_t*      dst;
        hipgraph_type_erased_device_array_t*      ids;
        hipgraph_type_erased_device_array_view_t* src_view;
        hipgraph_type_erased_device_array_view_t* dst_view;
        hipgraph_type_erased_device_array_view_t* ids_view;
        hipgraph_type_erased_device_array_view_t* wgt_view;

        hipgraph_resource_handle_t*               p_handle        = nullptr;
        p_handle = hipgraph_create_resource_handle(nullptr);
        ASSERT_NE(p_handle, nullptr) << "resource handle creation failed.";

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, vertex_tid, &src, ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "src create failed: " << hipgraph_error_message(*ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, vertex_tid, &dst, ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "dst create failed: " << hipgraph_error_message(*ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, edge_tid, &ids, ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "ids create failed: " << hipgraph_error_message(*ret_error);

        src_view = hipgraph_type_erased_device_array_view(src);
        dst_view = hipgraph_type_erased_device_array_view(dst);
        ids_view = hipgraph_type_erased_device_array_view(ids);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, src_view, (hipgraph_byte_t*)h_src, ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "src copy_from_host failed: " << hipgraph_error_message(*ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, dst_view, (hipgraph_byte_t*)h_dst, ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "dst copy_from_host failed: " << hipgraph_error_message(*ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, ids_view, (hipgraph_byte_t*)h_ids, ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "wgt copy_from_host failed: " << hipgraph_error_message(*ret_error);

        ret_code
            = hipgraph_type_erased_device_array_view_as_type(ids, weight_tid, &wgt_view, ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "wgt cast from ids failed: " << hipgraph_error_message(*ret_error);

        ret_code = hipgraph_sg_graph_create(p_handle,
                                            &properties,
                                            src_view,
                                            dst_view,
                                            wgt_view,
                                            nullptr,
                                            nullptr,
                                            store_transposed,
                                            renumber,
                                            HIPGRAPH_FALSE,
                                            p_graph,
                                            ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "graph creation failed: " << hipgraph_error_message(*ret_error);

        hipgraph_type_erased_device_array_view_free(wgt_view);
        hipgraph_type_erased_device_array_view_free(ids_view);
        hipgraph_type_erased_device_array_view_free(dst_view);
        hipgraph_type_erased_device_array_view_free(src_view);
        hipgraph_type_erased_device_array_free(ids);
        hipgraph_type_erased_device_array_free(dst);
        hipgraph_type_erased_device_array_free(src);
    }
#endif
} // namespace
