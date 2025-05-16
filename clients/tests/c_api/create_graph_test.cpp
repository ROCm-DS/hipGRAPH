// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0
/*
 * Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

/*
 * Copyright (C) 2021-2024, NVIDIA CORPORATION.
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

#include "test_utils.h" /* RUN_TEST */

#include "hipgraph_c/algorithms.h"
#include "hipgraph_c/graph.h"

#include <cstdio>

/*
 * Simple check of creating a graph from a COO on device memory.
 */

using vertex_t = int32_t;
using edge_t   = int32_t;
using weight_t = float;

namespace
{

    using namespace hipGRAPH::testing;
    TEST(PlumbingTest, CreateSgGraphSimple)
    {
        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;
        size_t                num_edges = 8;

        vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[] = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};

        hipgraph_resource_handle_t* p_handle = nullptr;
        hipgraph_graph_t*           graph    = nullptr;
        hipgraph_graph_properties_t properties;

        properties.is_symmetric  = HIPGRAPH_FALSE;
        properties.is_multigraph = HIPGRAPH_FALSE;

        hipgraph_data_type_id_t vertex_tid = HIPGRAPH_INT32;
        // hipgraph_data_type_id_t edge_tid   = HIPGRAPH_INT32;
        hipgraph_data_type_id_t weight_tid = HIPGRAPH_FLOAT32;

        p_handle = hipgraph_create_resource_handle(nullptr);
        ASSERT_NE(p_handle, nullptr) << "resource handle creation failed.";

        hipgraph_type_erased_device_array_t*      src;
        hipgraph_type_erased_device_array_t*      dst;
        hipgraph_type_erased_device_array_t*      wgt;
        hipgraph_type_erased_device_array_view_t* src_view;
        hipgraph_type_erased_device_array_view_t* dst_view;
        hipgraph_type_erased_device_array_view_t* wgt_view;

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, vertex_tid, &src, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "src create failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, vertex_tid, &dst, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "dst create failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, weight_tid, &wgt, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "wgt create failed: " << hipgraph_error_message(ret_error);

        src_view = hipgraph_type_erased_device_array_view(src);
        dst_view = hipgraph_type_erased_device_array_view(dst);
        wgt_view = hipgraph_type_erased_device_array_view(wgt);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, src_view, (hipgraph_byte_t*)h_src, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "src copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, dst_view, (hipgraph_byte_t*)h_dst, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "dst copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, wgt_view, (hipgraph_byte_t*)h_wgt, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "wgt copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_graph_create_sg(p_handle,
                                            &properties,
                                            nullptr,
                                            src_view,
                                            dst_view,
                                            wgt_view,
                                            nullptr,
                                            nullptr,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            &graph,
                                            &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "graph creation failed: " << hipgraph_error_message(ret_error);

        hipgraph_graph_free(graph);

        hipgraph_type_erased_device_array_view_free(wgt_view);
        hipgraph_type_erased_device_array_view_free(dst_view);
        hipgraph_type_erased_device_array_view_free(src_view);
        hipgraph_type_erased_device_array_free(wgt);
        hipgraph_type_erased_device_array_free(dst);
        hipgraph_type_erased_device_array_free(src);

        hipgraph_free_resource_handle(p_handle);
        hipgraph_error_free(ret_error);
    }

    TEST(PlumbingTest, CreateSgGraphCsr)
    {
        GTEST_SKIP() << "unimplemented";
        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;
        size_t                num_edges    = 8;
        size_t                num_vertices = 6;

        /*
  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5};
  */
        edge_t   h_offsets[] = {0, 1, 3, 6, 7, 8, 8};
        vertex_t h_indices[] = {1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_start[]   = {0, 1, 2, 3, 4, 5};
        weight_t h_wgt[]     = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};

        hipgraph_bool_t                   with_replacement       = HIPGRAPH_FALSE;
        hipgraph_bool_t                   return_hops            = HIPGRAPH_TRUE;
        hipgraph_prior_sources_behavior_t prior_sources_behavior = HIPGRAPH_DEFAULT;
        hipgraph_bool_t                   dedupe_sources         = HIPGRAPH_FALSE;
        hipgraph_bool_t                   renumber_results       = HIPGRAPH_FALSE;
        hipgraph_compression_type_t       compression            = HIPGRAPH_COO;
        hipgraph_bool_t                   compress_per_hop       = HIPGRAPH_FALSE;

        hipgraph_resource_handle_t* p_handle = nullptr;
        hipgraph_graph_t*           graph    = nullptr;
        hipgraph_graph_properties_t properties;

        properties.is_symmetric  = HIPGRAPH_FALSE;
        properties.is_multigraph = HIPGRAPH_FALSE;

        hipgraph_data_type_id_t vertex_tid = HIPGRAPH_INT32;
        // hipgraph_data_type_id_t edge_tid   = HIPGRAPH_INT32;
        hipgraph_data_type_id_t weight_tid = HIPGRAPH_FLOAT32;

        p_handle = hipgraph_create_resource_handle(nullptr);
        ASSERT_NE(p_handle, nullptr) << "resource handle creation failed.";

        hipgraph_type_erased_device_array_t*      offsets;
        hipgraph_type_erased_device_array_t*      indices;
        hipgraph_type_erased_device_array_t*      wgt;
        hipgraph_type_erased_device_array_view_t* offsets_view;
        hipgraph_type_erased_device_array_view_t* indices_view;
        hipgraph_type_erased_device_array_view_t* wgt_view;

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_vertices + 1, vertex_tid, &offsets, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "offsets create failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, vertex_tid, &indices, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "indices create failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, weight_tid, &wgt, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "wgt create failed: " << hipgraph_error_message(ret_error);

        offsets_view = hipgraph_type_erased_device_array_view(offsets);
        indices_view = hipgraph_type_erased_device_array_view(indices);
        wgt_view     = hipgraph_type_erased_device_array_view(wgt);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, offsets_view, (hipgraph_byte_t*)h_offsets, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "offsets copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, indices_view, (hipgraph_byte_t*)h_indices, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "indices copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, wgt_view, (hipgraph_byte_t*)h_wgt, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "wgt copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_sg_graph_create_from_csr(p_handle,
                                                     &properties,
                                                     offsets_view,
                                                     indices_view,
                                                     wgt_view,
                                                     nullptr,
                                                     nullptr,
                                                     HIPGRAPH_FALSE,
                                                     HIPGRAPH_FALSE,
                                                     HIPGRAPH_FALSE,
                                                     &graph,
                                                     &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "graph creation failed: " << hipgraph_error_message(ret_error);

        weight_t M[num_vertices][num_vertices];

        for(size_t i = 0; i < num_vertices; ++i)
            for(size_t j = 0; j < num_vertices; ++j)
                M[i][j] = -1;

        for(size_t i = 0; i < num_vertices; ++i)
            for(edge_t j = h_offsets[i]; j < h_offsets[i + 1]; ++j)
            {
                M[i][h_indices[j]] = h_wgt[j];
            }

        int fan_out[] = {-1};

        hipgraph_type_erased_device_array_t*      d_start        = nullptr;
        hipgraph_type_erased_device_array_view_t* d_start_view   = nullptr;
        hipgraph_type_erased_host_array_view_t*   h_fan_out_view = nullptr;
        hipgraph_sample_result_t*                 result         = nullptr;

        h_fan_out_view = hipgraph_type_erased_host_array_view_create(fan_out, 1, HIPGRAPH_INT32);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_vertices, HIPGRAPH_INT32, &d_start, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "d_start create failed: " << hipgraph_error_message(ret_error);

        d_start_view = hipgraph_type_erased_device_array_view(d_start);
        ret_code     = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, d_start_view, (hipgraph_byte_t*)h_start, &ret_error);

        hipgraph_rng_state_t* rng_state;
        ret_code = hipgraph_rng_state_create(p_handle, 0, &rng_state, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "rng_state create failed: " << hipgraph_error_message(ret_error);

        hipgraph_sampling_options_t* sampling_options;

        ret_code = hipgraph_sampling_options_create(&sampling_options, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
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
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    h_fan_out_view,
                                                    rng_state,
                                                    sampling_options,
                                                    HIPGRAPH_FALSE,
                                                    &result,
                                                    &ret_error);

        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "uniform_neighbor_sample failed: " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* srcs;
        hipgraph_type_erased_device_array_view_t* dsts;
        hipgraph_type_erased_device_array_view_t* wgts;

        srcs = hipgraph_sample_result_get_sources(result);
        dsts = hipgraph_sample_result_get_destinations(result);
        wgts = hipgraph_sample_result_get_edge_weight(result);

        size_t result_size = hipgraph_type_erased_device_array_view_size(srcs);

        vertex_t h_result_srcs[result_size];
        vertex_t h_result_dsts[result_size];
        weight_t h_result_wgts[result_size];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_result_srcs, srcs, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_result_dsts, dsts, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_result_wgts, wgts, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        EXPECT_EQ(result_size, num_edges) << "number of edges does not match";

        for(size_t i = 0; i < result_size; ++i)
        {
            EXPECT_EQ(M[h_result_srcs[i]][h_result_dsts[i]], h_result_wgts[i])
                << "uniform_neighbor_sample got edge that doesn't exist at position " << i;
        }

        hipgraph_sample_result_free(result);
        hipgraph_graph_free(graph);
        hipgraph_type_erased_device_array_view_free(wgt_view);
        hipgraph_type_erased_device_array_view_free(indices_view);
        hipgraph_type_erased_device_array_view_free(offsets_view);
        hipgraph_type_erased_device_array_free(wgt);
        hipgraph_type_erased_device_array_free(indices);
        hipgraph_type_erased_device_array_free(offsets);

        hipgraph_free_resource_handle(p_handle);
        hipgraph_error_free(ret_error);
        hipgraph_sampling_options_free(sampling_options);
    }

    TEST(PlumbingTest, CreateSgGraphSymmetricError)
    {
        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;
        size_t                num_edges = 8;

        vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[] = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};

        hipgraph_resource_handle_t* p_handle = nullptr;
        hipgraph_graph_t*           graph    = nullptr;
        hipgraph_graph_properties_t properties;

        properties.is_symmetric  = HIPGRAPH_TRUE;
        properties.is_multigraph = HIPGRAPH_FALSE;

        hipgraph_data_type_id_t vertex_tid = HIPGRAPH_INT32;
        // hipgraph_data_type_id_t edge_tid   = HIPGRAPH_INT32;
        hipgraph_data_type_id_t weight_tid = HIPGRAPH_FLOAT32;

        p_handle = hipgraph_create_resource_handle(nullptr);
        ASSERT_NE(p_handle, nullptr) << "resource handle creation failed.";

        hipgraph_type_erased_device_array_t*      src;
        hipgraph_type_erased_device_array_t*      dst;
        hipgraph_type_erased_device_array_t*      wgt;
        hipgraph_type_erased_device_array_view_t* src_view;
        hipgraph_type_erased_device_array_view_t* dst_view;
        hipgraph_type_erased_device_array_view_t* wgt_view;

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, vertex_tid, &src, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "src create failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, vertex_tid, &dst, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "dst create failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, weight_tid, &wgt, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "wgt create failed: " << hipgraph_error_message(ret_error);

        src_view = hipgraph_type_erased_device_array_view(src);
        dst_view = hipgraph_type_erased_device_array_view(dst);
        wgt_view = hipgraph_type_erased_device_array_view(wgt);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, src_view, (hipgraph_byte_t*)h_src, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "src copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, dst_view, (hipgraph_byte_t*)h_dst, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "dst copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, wgt_view, (hipgraph_byte_t*)h_wgt, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "wgt copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_graph_create_sg(p_handle,
                                            &properties,
                                            nullptr,
                                            src_view,
                                            dst_view,
                                            wgt_view,
                                            nullptr,
                                            nullptr,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_TRUE,
                                            &graph,
                                            &ret_error);
        EXPECT_NE(ret_code, HIPGRAPH_SUCCESS) << "graph creation succeeded but should have failed.";

        if(ret_code == HIPGRAPH_SUCCESS)
            hipgraph_graph_free(graph);

        hipgraph_type_erased_device_array_view_free(wgt_view);
        hipgraph_type_erased_device_array_view_free(dst_view);
        hipgraph_type_erased_device_array_view_free(src_view);
        hipgraph_type_erased_device_array_free(wgt);
        hipgraph_type_erased_device_array_free(dst);
        hipgraph_type_erased_device_array_free(src);

        hipgraph_free_resource_handle(p_handle);
        hipgraph_error_free(ret_error);
    }

    TEST(PlumbingTest, CreateSgGraphWithIsolatedVertices)
    {
        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;
        size_t                num_edges      = 8;
        size_t                num_vertices   = 7;
        double                alpha          = 0.95;
        double                epsilon        = 0.0001;
        size_t                max_iterations = 20;

        vertex_t h_vertices[] = {0, 1, 2, 3, 4, 5, 6};
        vertex_t h_src[]      = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]      = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]      = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        weight_t h_result[]
            = {0.0859168, 0.158029, 0.0616337, 0.179675, 0.113239, 0.339873, 0.0616337};

        hipgraph_resource_handle_t* p_handle = nullptr;
        hipgraph_graph_t*           graph    = nullptr;
        hipgraph_graph_properties_t properties;

        properties.is_symmetric  = HIPGRAPH_FALSE;
        properties.is_multigraph = HIPGRAPH_FALSE;

        hipgraph_data_type_id_t vertex_tid = HIPGRAPH_INT32;
        // hipgraph_data_type_id_t edge_tid   = HIPGRAPH_INT32;
        hipgraph_data_type_id_t weight_tid = HIPGRAPH_FLOAT32;

        p_handle = hipgraph_create_resource_handle(nullptr);
        ASSERT_NE(p_handle, nullptr) << "resource handle creation failed.";

        hipgraph_type_erased_device_array_t*      vertices;
        hipgraph_type_erased_device_array_t*      src;
        hipgraph_type_erased_device_array_t*      dst;
        hipgraph_type_erased_device_array_t*      wgt;
        hipgraph_type_erased_device_array_view_t* vertices_view;
        hipgraph_type_erased_device_array_view_t* src_view;
        hipgraph_type_erased_device_array_view_t* dst_view;
        hipgraph_type_erased_device_array_view_t* wgt_view;

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_vertices, vertex_tid, &vertices, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "vertices create failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, vertex_tid, &src, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "src create failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, vertex_tid, &dst, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "dst create failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, weight_tid, &wgt, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "wgt create failed: " << hipgraph_error_message(ret_error);

        vertices_view = hipgraph_type_erased_device_array_view(vertices);
        src_view      = hipgraph_type_erased_device_array_view(src);
        dst_view      = hipgraph_type_erased_device_array_view(dst);
        wgt_view      = hipgraph_type_erased_device_array_view(wgt);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, vertices_view, (hipgraph_byte_t*)h_vertices, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "vertices copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, src_view, (hipgraph_byte_t*)h_src, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "src copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, dst_view, (hipgraph_byte_t*)h_dst, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "dst copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, wgt_view, (hipgraph_byte_t*)h_wgt, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "wgt copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_graph_create_sg(p_handle,
                                            &properties,
                                            vertices_view,
                                            src_view,
                                            dst_view,
                                            wgt_view,
                                            nullptr,
                                            nullptr,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            &graph,
                                            &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "graph creation failed: " << hipgraph_error_message(ret_error);

        hipgraph_centrality_result_t* result = nullptr;

        // To verify we will call pagerank
        ret_code = hipgraph_pagerank(p_handle,
                                     graph,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     alpha,
                                     epsilon,
                                     max_iterations,
                                     HIPGRAPH_FALSE,
                                     &result,
                                     &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "hipgraph_pagerank failed: " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* result_vertices;
        hipgraph_type_erased_device_array_view_t* pageranks;

        result_vertices = hipgraph_centrality_result_get_vertices(result);
        pageranks       = hipgraph_centrality_result_get_values(result);

        vertex_t h_result_vertices[num_vertices];
        weight_t h_pageranks[num_vertices];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_result_vertices, result_vertices, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_pageranks, pageranks, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        for(size_t i = 0; i < num_vertices; ++i)
        {
            EXPECT_NEAR(h_result[h_result_vertices[i]], h_pageranks[i], 0.001)
                << "pagerank results don't match at position " << i;
        }

        hipgraph_centrality_result_free(result);
        hipgraph_graph_free(graph);

        hipgraph_type_erased_device_array_view_free(wgt_view);
        hipgraph_type_erased_device_array_view_free(dst_view);
        hipgraph_type_erased_device_array_view_free(src_view);
        hipgraph_type_erased_device_array_view_free(vertices_view);
        hipgraph_type_erased_device_array_free(wgt);
        hipgraph_type_erased_device_array_free(dst);
        hipgraph_type_erased_device_array_free(src);
        hipgraph_type_erased_device_array_free(vertices);

        hipgraph_free_resource_handle(p_handle);
        hipgraph_error_free(ret_error);
    }

    TEST(PlumbingTest, CreateSgGraphCsrWithIsolated)
    {
        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;
        size_t                num_edges      = 8;
        size_t                num_vertices   = 7;
        double                alpha          = 0.95;
        double                epsilon        = 0.0001;
        size_t                max_iterations = 20;

        /*
  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5};
  */
        edge_t   h_offsets[] = {0, 1, 3, 6, 7, 8, 8, 8};
        vertex_t h_indices[] = {1, 3, 4, 0, 1, 3, 5, 5};
        // vertex_t h_start[]   = {0, 1, 2, 3, 4, 5};
        weight_t h_wgt[] = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        weight_t h_result[]
            = {0.0859168, 0.158029, 0.0616337, 0.179675, 0.113239, 0.339873, 0.0616337};

        hipgraph_resource_handle_t* p_handle = nullptr;
        hipgraph_graph_t*           graph    = nullptr;
        hipgraph_graph_properties_t properties;

        properties.is_symmetric  = HIPGRAPH_FALSE;
        properties.is_multigraph = HIPGRAPH_FALSE;

        hipgraph_data_type_id_t vertex_tid = HIPGRAPH_INT32;
        // hipgraph_data_type_id_t edge_tid   = HIPGRAPH_INT32;
        hipgraph_data_type_id_t weight_tid = HIPGRAPH_FLOAT32;

        p_handle = hipgraph_create_resource_handle(nullptr);
        ASSERT_NE(p_handle, nullptr) << "resource handle creation failed.";

        hipgraph_type_erased_device_array_t*      offsets;
        hipgraph_type_erased_device_array_t*      indices;
        hipgraph_type_erased_device_array_t*      wgt;
        hipgraph_type_erased_device_array_view_t* offsets_view;
        hipgraph_type_erased_device_array_view_t* indices_view;
        hipgraph_type_erased_device_array_view_t* wgt_view;

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_vertices + 1, vertex_tid, &offsets, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "offsets create failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, vertex_tid, &indices, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "indices create failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, weight_tid, &wgt, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "wgt create failed: " << hipgraph_error_message(ret_error);

        offsets_view = hipgraph_type_erased_device_array_view(offsets);
        indices_view = hipgraph_type_erased_device_array_view(indices);
        wgt_view     = hipgraph_type_erased_device_array_view(wgt);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, offsets_view, (hipgraph_byte_t*)h_offsets, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "offsets copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, indices_view, (hipgraph_byte_t*)h_indices, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "indices copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, wgt_view, (hipgraph_byte_t*)h_wgt, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "wgt copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_sg_graph_create_from_csr(p_handle,
                                                     &properties,
                                                     offsets_view,
                                                     indices_view,
                                                     wgt_view,
                                                     nullptr,
                                                     nullptr,
                                                     HIPGRAPH_FALSE,
                                                     HIPGRAPH_FALSE,
                                                     HIPGRAPH_FALSE,
                                                     &graph,
                                                     &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "graph creation failed: " << hipgraph_error_message(ret_error);

        hipgraph_centrality_result_t* result = nullptr;

        // To verify we will call pagerank
        ret_code = hipgraph_pagerank(p_handle,
                                     graph,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     alpha,
                                     epsilon,
                                     max_iterations,
                                     HIPGRAPH_FALSE,
                                     &result,
                                     &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "hipgraph_pagerank failed: " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* result_vertices;
        hipgraph_type_erased_device_array_view_t* pageranks;

        result_vertices = hipgraph_centrality_result_get_vertices(result);
        pageranks       = hipgraph_centrality_result_get_values(result);

        vertex_t h_result_vertices[num_vertices];
        weight_t h_pageranks[num_vertices];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_result_vertices, result_vertices, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_pageranks, pageranks, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        for(size_t i = 0; i < num_vertices; ++i)
        {
            EXPECT_NEAR(h_result[h_result_vertices[i]], h_pageranks[i], 0.001)
                << "pagerank results don't match at position " << i;
        }

        hipgraph_centrality_result_free(result);
        hipgraph_graph_free(graph);
        hipgraph_type_erased_device_array_view_free(wgt_view);
        hipgraph_type_erased_device_array_view_free(indices_view);
        hipgraph_type_erased_device_array_view_free(offsets_view);
        hipgraph_type_erased_device_array_free(wgt);
        hipgraph_type_erased_device_array_free(indices);
        hipgraph_type_erased_device_array_free(offsets);

        hipgraph_free_resource_handle(p_handle);
        hipgraph_error_free(ret_error);
    }

    TEST(PlumbingTest, CreateSgGraphWithIsolatedVerticesMultiInput)
    {
        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;
        size_t                num_edges      = 66;
        size_t                num_vertices   = 7;
        double                alpha          = 0.95;
        double                epsilon        = 0.0001;
        size_t                max_iterations = 20;

        vertex_t h_vertices[] = {0, 1, 2, 3, 4, 5, 6};
        vertex_t h_src[]      = {0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5, 0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5,
                                 0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5, 0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5,
                                 0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5, 0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5};
        vertex_t h_dst[]      = {1, 3, 4, 0, 1, 3, 5, 5, 5, 5, 5, 1, 3, 4, 0, 1, 3, 5, 5, 5, 5, 5,
                                 1, 3, 4, 0, 1, 3, 5, 5, 5, 5, 5, 1, 3, 4, 0, 1, 3, 5, 5, 5, 5, 5,
                                 1, 3, 4, 0, 1, 3, 5, 5, 5, 5, 5, 1, 3, 4, 0, 1, 3, 5, 5, 5, 5, 5};
        weight_t h_wgt[]
            = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 3.2f, 3.2f, 1.7f, 0.1f, 2.1f, 1.1f,
               5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 3.2f, 3.2f, 1.7f, 0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f,
               7.2f, 3.2f, 3.2f, 3.2f, 1.7f, 0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 3.2f,
               3.2f, 1.7f, 0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 3.2f, 3.2f, 1.7f, 0.1f,
               2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 3.2f, 3.2f, 1.7f};
        weight_t h_result[]
            = {0.0859168, 0.158029, 0.0616337, 0.179675, 0.113239, 0.339873, 0.0616337};

        hipgraph_resource_handle_t* p_handle = nullptr;
        hipgraph_graph_t*           graph    = nullptr;
        hipgraph_graph_properties_t properties;

        properties.is_symmetric  = HIPGRAPH_FALSE;
        properties.is_multigraph = HIPGRAPH_FALSE;

        hipgraph_data_type_id_t vertex_tid = HIPGRAPH_INT32;
        // hipgraph_data_type_id_t edge_tid   = HIPGRAPH_INT32;
        hipgraph_data_type_id_t weight_tid = HIPGRAPH_FLOAT32;

        p_handle = hipgraph_create_resource_handle(nullptr);
        ASSERT_NE(p_handle, nullptr) << "resource handle creation failed.";

        hipgraph_type_erased_device_array_t*      vertices;
        hipgraph_type_erased_device_array_t*      src;
        hipgraph_type_erased_device_array_t*      dst;
        hipgraph_type_erased_device_array_t*      wgt;
        hipgraph_type_erased_device_array_view_t* vertices_view;
        hipgraph_type_erased_device_array_view_t* src_view;
        hipgraph_type_erased_device_array_view_t* dst_view;
        hipgraph_type_erased_device_array_view_t* wgt_view;

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_vertices, vertex_tid, &vertices, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "vertices create failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, vertex_tid, &src, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "src create failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, vertex_tid, &dst, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "dst create failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, weight_tid, &wgt, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "wgt create failed: " << hipgraph_error_message(ret_error);

        vertices_view = hipgraph_type_erased_device_array_view(vertices);
        src_view      = hipgraph_type_erased_device_array_view(src);
        dst_view      = hipgraph_type_erased_device_array_view(dst);
        wgt_view      = hipgraph_type_erased_device_array_view(wgt);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, vertices_view, (hipgraph_byte_t*)h_vertices, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "vertices copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, src_view, (hipgraph_byte_t*)h_src, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "src copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, dst_view, (hipgraph_byte_t*)h_dst, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "dst copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, wgt_view, (hipgraph_byte_t*)h_wgt, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "wgt copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_graph_create_sg(p_handle,
                                            &properties,
                                            vertices_view,
                                            src_view,
                                            dst_view,
                                            wgt_view,
                                            nullptr,
                                            nullptr,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_TRUE,
                                            HIPGRAPH_TRUE,
                                            HIPGRAPH_FALSE,
                                            &graph,
                                            &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "graph creation failed: " << hipgraph_error_message(ret_error);

        hipgraph_centrality_result_t* result = nullptr;

        // To verify we will call pagerank
        ret_code = hipgraph_pagerank(p_handle,
                                     graph,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     alpha,
                                     epsilon,
                                     max_iterations,
                                     HIPGRAPH_FALSE,
                                     &result,
                                     &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "hipgraph_pagerank failed: " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* result_vertices;
        hipgraph_type_erased_device_array_view_t* pageranks;

        result_vertices = hipgraph_centrality_result_get_vertices(result);
        pageranks       = hipgraph_centrality_result_get_values(result);

        vertex_t h_result_vertices[num_vertices];
        weight_t h_pageranks[num_vertices];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_result_vertices, result_vertices, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_pageranks, pageranks, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        for(size_t i = 0; i < num_vertices; ++i)
        {
            EXPECT_NEAR(h_result[h_result_vertices[i]], h_pageranks[i], 0.001)
                << "pagerank results don't match at position " << i;
        }

        hipgraph_centrality_result_free(result);
        hipgraph_graph_free(graph);

        hipgraph_type_erased_device_array_view_free(wgt_view);
        hipgraph_type_erased_device_array_view_free(dst_view);
        hipgraph_type_erased_device_array_view_free(src_view);
        hipgraph_type_erased_device_array_view_free(vertices_view);
        hipgraph_type_erased_device_array_free(wgt);
        hipgraph_type_erased_device_array_free(dst);
        hipgraph_type_erased_device_array_free(src);
        hipgraph_type_erased_device_array_free(vertices);

        hipgraph_free_resource_handle(p_handle);
        hipgraph_error_free(ret_error);
    }

} // namespace
