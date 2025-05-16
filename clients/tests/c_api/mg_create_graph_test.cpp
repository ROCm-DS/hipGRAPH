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

#include "test_utils.h" /* RUN_TEST */
#include "mg_test_utils.h" /* RUN_TEST */

#include "hipgraph_c/algorithms.h"

#include <cmath>
#include <cstdio>
#include <stdlib.h>

/*
 * Simple check of creating a graph from a COO on device memory.
 */
namespace
{
    using namespace hipGRAPH::testing;
    int test_create_mg_graph_simple(const hipgraph_resource_handle_t* p_handle)
    {
        int test_ret_value = 0;

        using vertex_t = int32_t;
        using edge_t   = int32_t;
        using weight_t = float;

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;
        size_t                num_edges    = 8;
        size_t                num_vertices = 6;

        vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[] = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};

        hipgraph_graph_t*           graph = nullptr;
        hipgraph_graph_properties_t properties;

        properties.is_symmetric  = HIPGRAPH_FALSE;
        properties.is_multigraph = HIPGRAPH_FALSE;

        hipgraph_data_type_id_t vertex_tid = HIPGRAPH_INT32;
        hipgraph_data_type_id_t edge_tid   = HIPGRAPH_INT32;
        hipgraph_data_type_id_t weight_tid = HIPGRAPH_FLOAT32;

        hipgraph_type_erased_device_array_t*      src;
        hipgraph_type_erased_device_array_t*      dst;
        hipgraph_type_erased_device_array_t*      wgt;
        hipgraph_type_erased_device_array_view_t* src_view;
        hipgraph_type_erased_device_array_view_t* dst_view;
        hipgraph_type_erased_device_array_view_t* wgt_view;

        int my_rank = hipgraph_resource_handle_get_rank(p_handle);

        for(int i = 0; i < num_edges; ++i)
        {
            h_src[i] += 10 * my_rank;
            h_dst[i] += 10 * my_rank;
        }

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, vertex_tid, &src, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "src create failed. " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, vertex_tid, &dst, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "dst create failed.";

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, weight_tid, &wgt, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "wgt create failed.";

        src_view = hipgraph_type_erased_device_array_view(src);
        dst_view = hipgraph_type_erased_device_array_view(dst);
        wgt_view = hipgraph_type_erased_device_array_view(wgt);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, src_view, (hipgraph_byte_t*)h_src, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "src copy_from_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, dst_view, (hipgraph_byte_t*)h_dst, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "dst copy_from_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, wgt_view, (hipgraph_byte_t*)h_wgt, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "wgt copy_from_host failed.";

        ret_code = hipgraph_graph_create_mg(
            p_handle,
            &properties,
            nullptr,
            (hipgraph_type_erased_device_array_view_t const* const*)&src_view,
            (hipgraph_type_erased_device_array_view_t const* const*)&dst_view,
            (hipgraph_type_erased_device_array_view_t const* const*)&wgt_view,
            nullptr,
            nullptr,
            HIPGRAPH_FALSE,
            1,
            HIPGRAPH_FALSE,
            HIPGRAPH_FALSE,
            HIPGRAPH_TRUE,
            &graph,
            &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "graph creation failed. " << hipgraph_error_message(ret_error);

        hipgraph_graph_free(graph);

        hipgraph_type_erased_device_array_view_free(wgt_view);
        hipgraph_type_erased_device_array_view_free(dst_view);
        hipgraph_type_erased_device_array_view_free(src_view);
        hipgraph_type_erased_device_array_free(wgt);
        hipgraph_type_erased_device_array_free(dst);
        hipgraph_type_erased_device_array_free(src);

        hipgraph_error_free(ret_error);

        return test_ret_value;
    }

    int test_create_mg_graph_multiple_edge_lists(const hipgraph_resource_handle_t* p_handle)
    {
        int test_ret_value = 0;

        using vertex_t = int32_t;
        using edge_t   = int32_t;
        using weight_t = float;

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;
        size_t                num_edges    = 8;
        size_t                num_vertices = 7;

        double alpha          = 0.95;
        double epsilon        = 0.0001;
        size_t max_iterations = 20;

        vertex_t h_vertices[] = {0, 1, 2, 3, 4, 5, 6};
        vertex_t h_src[]      = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]      = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]      = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        weight_t h_result[]
            = {0.0859168, 0.158029, 0.0616337, 0.179675, 0.113239, 0.339873, 0.0616337};

        hipgraph_graph_t*           graph = nullptr;
        hipgraph_graph_properties_t properties;

        properties.is_symmetric  = HIPGRAPH_FALSE;
        properties.is_multigraph = HIPGRAPH_FALSE;

        hipgraph_data_type_id_t vertex_tid = HIPGRAPH_INT32;
        hipgraph_data_type_id_t edge_tid   = HIPGRAPH_INT32;
        hipgraph_data_type_id_t weight_tid = HIPGRAPH_FLOAT32;

        const size_t num_local_arrays = 2;

        hipgraph_type_erased_device_array_t*      vertices[num_local_arrays];
        hipgraph_type_erased_device_array_t*      src[num_local_arrays];
        hipgraph_type_erased_device_array_t*      dst[num_local_arrays];
        hipgraph_type_erased_device_array_t*      wgt[num_local_arrays];
        hipgraph_type_erased_device_array_view_t* vertices_view[num_local_arrays];
        hipgraph_type_erased_device_array_view_t* src_view[num_local_arrays];
        hipgraph_type_erased_device_array_view_t* dst_view[num_local_arrays];
        hipgraph_type_erased_device_array_view_t* wgt_view[num_local_arrays];

        int my_rank   = hipgraph_resource_handle_get_rank(p_handle);
        int comm_size = hipgraph_resource_handle_get_comm_size(p_handle);

        size_t local_num_vertices = num_vertices / comm_size;
        size_t local_start_vertex = my_rank * local_num_vertices;
        size_t local_num_edges    = num_edges / comm_size;
        size_t local_start_edge   = my_rank * local_num_edges;

        local_num_edges
            = (my_rank != (comm_size - 1)) ? local_num_edges : (num_edges - local_start_edge);
        local_num_vertices = (my_rank != (comm_size - 1)) ? local_num_vertices
                                                          : (num_vertices - local_start_vertex);

        for(size_t i = 0; i < num_local_arrays; ++i)
        {
            size_t vertex_count = local_num_vertices / num_local_arrays;
            size_t vertex_start = i * vertex_count;
            vertex_count        = (i != (num_local_arrays - 1)) ? vertex_count
                                                                : (local_num_vertices - vertex_start);

            ret_code = hipgraph_type_erased_device_array_create(
                p_handle, vertex_count, vertex_tid, vertices + i, &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
                << "vertices create failed. " << hipgraph_error_message(ret_error);

            size_t edge_count = (local_num_edges + num_local_arrays - 1) / num_local_arrays;
            size_t edge_start = i * edge_count;
            edge_count        = (edge_count < (local_num_edges - edge_start))
                                    ? edge_count
                                    : (local_num_edges - edge_start);

            ret_code = hipgraph_type_erased_device_array_create(
                p_handle, edge_count, vertex_tid, src + i, &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "src create failed.";

            ret_code = hipgraph_type_erased_device_array_create(
                p_handle, edge_count, vertex_tid, dst + i, &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "dst create failed.";

            ret_code = hipgraph_type_erased_device_array_create(
                p_handle, edge_count, weight_tid, wgt + i, &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "wgt create failed.";

            vertices_view[i] = hipgraph_type_erased_device_array_view(vertices[i]);
            src_view[i]      = hipgraph_type_erased_device_array_view(src[i]);
            dst_view[i]      = hipgraph_type_erased_device_array_view(dst[i]);
            wgt_view[i]      = hipgraph_type_erased_device_array_view(wgt[i]);

            ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
                p_handle,
                vertices_view[i],
                (hipgraph_byte_t*)(h_vertices + local_start_vertex + vertex_start),
                &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "src copy_from_host failed.";

            ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
                p_handle,
                src_view[i],
                (hipgraph_byte_t*)(h_src + local_start_edge + edge_start),
                &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "src copy_from_host failed.";

            ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
                p_handle,
                dst_view[i],
                (hipgraph_byte_t*)(h_dst + local_start_edge + edge_start),
                &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "dst copy_from_host failed.";

            ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
                p_handle,
                wgt_view[i],
                (hipgraph_byte_t*)(h_wgt + local_start_edge + edge_start),
                &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "wgt copy_from_host failed.";
        }

        ret_code = hipgraph_graph_create_mg(
            p_handle,
            &properties,
            (hipgraph_type_erased_device_array_view_t const* const*)vertices_view,
            (hipgraph_type_erased_device_array_view_t const* const*)src_view,
            (hipgraph_type_erased_device_array_view_t const* const*)dst_view,
            (hipgraph_type_erased_device_array_view_t const* const*)wgt_view,
            nullptr,
            nullptr,
            HIPGRAPH_FALSE,
            num_local_arrays,
            HIPGRAPH_FALSE,
            HIPGRAPH_FALSE,
            HIPGRAPH_TRUE,
            &graph,
            &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "graph creation failed. " << hipgraph_error_message(ret_error);

        //
        //  Now call pagerank and check results...
        //
        hipgraph_centrality_result_t* result = nullptr;

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
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "hipgraph_pagerank failed.";

        // NOTE: Because we get back vertex ids and pageranks, we can simply compare
        //       the returned values with the expected results for the entire
        //       graph.  Each GPU will have a subset of the total vertices, so
        //       they will do a subset of the comparisons.
        hipgraph_type_erased_device_array_view_t* result_vertices;
        hipgraph_type_erased_device_array_view_t* pageranks;

        result_vertices = hipgraph_centrality_result_get_vertices(result);
        pageranks       = hipgraph_centrality_result_get_values(result);

        size_t num_local_vertices = hipgraph_type_erased_device_array_view_size(result_vertices);

        vertex_t h_result_vertices[num_local_vertices];
        weight_t h_pageranks[num_local_vertices];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_result_vertices, result_vertices, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_pageranks, pageranks, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        for(int i = 0; (i < num_local_vertices) && (test_ret_value == 0); ++i)
        {
            EXPECT_NEAR(h_result[h_result_vertices[i]], h_pageranks[i], 0.001)
                << "pagerank results don't match";
        }

        hipgraph_centrality_result_free(result);
        hipgraph_graph_free(graph);

        for(size_t i = 0; i < num_local_arrays; ++i)
        {
            hipgraph_type_erased_device_array_view_free(wgt_view[i]);
            hipgraph_type_erased_device_array_view_free(dst_view[i]);
            hipgraph_type_erased_device_array_view_free(src_view[i]);
            hipgraph_type_erased_device_array_view_free(vertices_view[i]);
            hipgraph_type_erased_device_array_free(wgt[i]);
            hipgraph_type_erased_device_array_free(dst[i]);
            hipgraph_type_erased_device_array_free(src[i]);
            hipgraph_type_erased_device_array_free(vertices[i]);
        }

        hipgraph_error_free(ret_error);

        return test_ret_value;
    }

    int test_create_mg_graph_multiple_edge_lists_multi_edge(
        const hipgraph_resource_handle_t* p_handle)
    {
        int test_ret_value = 0;

        using vertex_t = int32_t;
        using edge_t   = int32_t;
        using weight_t = float;

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;
        size_t                num_edges    = 11;
        size_t                num_vertices = 7;

        double alpha          = 0.95;
        double epsilon        = 0.0001;
        size_t max_iterations = 20;

        vertex_t h_vertices[] = {0, 1, 2, 3, 4, 5, 6};
        vertex_t h_src[]      = {0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5};
        vertex_t h_dst[]      = {1, 3, 4, 0, 1, 3, 5, 5, 5, 5, 5};
        weight_t h_wgt[]      = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 3.2f, 3.2f, 1.1f};
        weight_t h_result[]
            = {0.0859168, 0.158029, 0.0616337, 0.179675, 0.113239, 0.339873, 0.0616337};

        hipgraph_graph_t*           graph = nullptr;
        hipgraph_graph_properties_t properties;

        properties.is_symmetric  = HIPGRAPH_FALSE;
        properties.is_multigraph = HIPGRAPH_FALSE;

        hipgraph_data_type_id_t vertex_tid = HIPGRAPH_INT32;
        hipgraph_data_type_id_t edge_tid   = HIPGRAPH_INT32;
        hipgraph_data_type_id_t weight_tid = HIPGRAPH_FLOAT32;

        const size_t num_local_arrays = 2;

        hipgraph_type_erased_device_array_t*      vertices[num_local_arrays];
        hipgraph_type_erased_device_array_t*      src[num_local_arrays];
        hipgraph_type_erased_device_array_t*      dst[num_local_arrays];
        hipgraph_type_erased_device_array_t*      wgt[num_local_arrays];
        hipgraph_type_erased_device_array_view_t* vertices_view[num_local_arrays];
        hipgraph_type_erased_device_array_view_t* src_view[num_local_arrays];
        hipgraph_type_erased_device_array_view_t* dst_view[num_local_arrays];
        hipgraph_type_erased_device_array_view_t* wgt_view[num_local_arrays];

        int my_rank   = hipgraph_resource_handle_get_rank(p_handle);
        int comm_size = hipgraph_resource_handle_get_comm_size(p_handle);

        size_t local_num_vertices = num_vertices / comm_size;
        size_t local_start_vertex = my_rank * local_num_vertices;
        size_t local_num_edges    = num_edges / comm_size;
        size_t local_start_edge   = my_rank * local_num_edges;

        local_num_edges
            = (my_rank != (comm_size - 1)) ? local_num_edges : (num_edges - local_start_edge);
        local_num_vertices = (my_rank != (comm_size - 1)) ? local_num_vertices
                                                          : (num_vertices - local_start_vertex);

        for(size_t i = 0; i < num_local_arrays; ++i)
        {
            size_t vertex_count = (local_num_vertices + num_local_arrays - 1) / num_local_arrays;
            size_t vertex_start = i * vertex_count;
            vertex_count        = (i != (num_local_arrays - 1)) ? vertex_count
                                                                : (local_num_vertices - vertex_start);

            ret_code = hipgraph_type_erased_device_array_create(
                p_handle, vertex_count, vertex_tid, vertices + i, &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
                << "vertices create failed. " << hipgraph_error_message(ret_error);

            size_t edge_count = (local_num_edges + num_local_arrays - 1) / num_local_arrays;
            size_t edge_start = i * edge_count;
            edge_count        = (edge_count < (local_num_edges - edge_start))
                                    ? edge_count
                                    : (local_num_edges - edge_start);

            ret_code = hipgraph_type_erased_device_array_create(
                p_handle, edge_count, vertex_tid, src + i, &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "src create failed.";

            ret_code = hipgraph_type_erased_device_array_create(
                p_handle, edge_count, vertex_tid, dst + i, &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "dst create failed.";

            ret_code = hipgraph_type_erased_device_array_create(
                p_handle, edge_count, weight_tid, wgt + i, &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "wgt create failed.";

            vertices_view[i] = hipgraph_type_erased_device_array_view(vertices[i]);
            src_view[i]      = hipgraph_type_erased_device_array_view(src[i]);
            dst_view[i]      = hipgraph_type_erased_device_array_view(dst[i]);
            wgt_view[i]      = hipgraph_type_erased_device_array_view(wgt[i]);

            ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
                p_handle,
                vertices_view[i],
                (hipgraph_byte_t*)(h_vertices + local_start_vertex + vertex_start),
                &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "src copy_from_host failed.";

            ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
                p_handle,
                src_view[i],
                (hipgraph_byte_t*)(h_src + local_start_edge + edge_start),
                &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "src copy_from_host failed.";

            ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
                p_handle,
                dst_view[i],
                (hipgraph_byte_t*)(h_dst + local_start_edge + edge_start),
                &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "dst copy_from_host failed.";

            ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
                p_handle,
                wgt_view[i],
                (hipgraph_byte_t*)(h_wgt + local_start_edge + edge_start),
                &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "wgt copy_from_host failed.";
        }

        ret_code = hipgraph_graph_create_mg(
            p_handle,
            &properties,
            (hipgraph_type_erased_device_array_view_t const* const*)vertices_view,
            (hipgraph_type_erased_device_array_view_t const* const*)src_view,
            (hipgraph_type_erased_device_array_view_t const* const*)dst_view,
            (hipgraph_type_erased_device_array_view_t const* const*)wgt_view,
            nullptr,
            nullptr,
            HIPGRAPH_FALSE,
            num_local_arrays,
            HIPGRAPH_TRUE,
            HIPGRAPH_TRUE,
            HIPGRAPH_TRUE,
            &graph,
            &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "graph creation failed. " << hipgraph_error_message(ret_error);

        //
        //  Now call pagerank and check results...
        //
        hipgraph_centrality_result_t* result = nullptr;

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
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "hipgraph_pagerank failed.";

        // NOTE: Because we get back vertex ids and pageranks, we can simply compare
        //       the returned values with the expected results for the entire
        //       graph.  Each GPU will have a subset of the total vertices, so
        //       they will do a subset of the comparisons.
        hipgraph_type_erased_device_array_view_t* result_vertices;
        hipgraph_type_erased_device_array_view_t* pageranks;

        result_vertices = hipgraph_centrality_result_get_vertices(result);
        pageranks       = hipgraph_centrality_result_get_values(result);

        size_t num_local_vertices = hipgraph_type_erased_device_array_view_size(result_vertices);

        vertex_t h_result_vertices[num_local_vertices];
        weight_t h_pageranks[num_local_vertices];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_result_vertices, result_vertices, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_pageranks, pageranks, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        for(int i = 0; (i < num_local_vertices) && (test_ret_value == 0); ++i)
        {
            EXPECT_NEAR(h_result[h_result_vertices[i]], h_pageranks[i], 0.001)
                << "pagerank results don't match";
        }

        hipgraph_centrality_result_free(result);
        hipgraph_graph_free(graph);

        for(size_t i = 0; i < num_local_arrays; ++i)
        {
            hipgraph_type_erased_device_array_view_free(wgt_view[i]);
            hipgraph_type_erased_device_array_view_free(dst_view[i]);
            hipgraph_type_erased_device_array_view_free(src_view[i]);
            hipgraph_type_erased_device_array_view_free(vertices_view[i]);
            hipgraph_type_erased_device_array_free(wgt[i]);
            hipgraph_type_erased_device_array_free(dst[i]);
            hipgraph_type_erased_device_array_free(src[i]);
            hipgraph_type_erased_device_array_free(vertices[i]);
        }

        hipgraph_error_free(ret_error);

        return test_ret_value;
    }

} // namespace
