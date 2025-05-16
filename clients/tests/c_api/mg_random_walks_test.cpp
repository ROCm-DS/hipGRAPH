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

#include "mg_test_utils.h" /* RUN_TEST */

#include "hipgraph_c/algorithms.h"
#include "hipgraph_c/graph.h"

#include <cmath>

using vertex_t = int32_t;
using edge_t   = int32_t;
using weight_t = float;

namespace
{
    using namespace hipGRAPH::testing;
    int generic_uniform_random_walks_test(const hipgraph_resource_handle_t* p_handle,
                                          vertex_t*                         h_src,
                                          vertex_t*                         h_dst,
                                          weight_t*                         h_wgt,
                                          size_t                            num_vertices,
                                          size_t                            num_edges,
                                          vertex_t*                         h_start,
                                          size_t                            num_starts,
                                          size_t                            max_depth,
                                          hipgraph_bool_t                   store_transposed)
    {
        int test_ret_value = 0;

        hipgraph_error_code_t ret_code  = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error = nullptr;

        hipgraph_graph_t*              graph  = nullptr;
        hipgraph_random_walk_result_t* result = nullptr;

        hipgraph_type_erased_device_array_t*      d_start      = nullptr;
        hipgraph_type_erased_device_array_view_t* d_start_view = nullptr;

        ret_code = create_mg_test_graph(p_handle,
                                        h_src,
                                        h_dst,
                                        h_wgt,
                                        num_edges,
                                        store_transposed,
                                        HIPGRAPH_FALSE,
                                        &graph,
                                        &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "graph creation failed.";

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_starts, HIPGRAPH_INT32, &d_start, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "d_start create failed.";

        d_start_view = hipgraph_type_erased_device_array_view(d_start);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, d_start_view, (hipgraph_byte_t*)h_start, &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "start copy_from_host failed.";

        ret_code = hipgraph_uniform_random_walks(
            p_handle, graph, d_start_view, max_depth, &result, &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "uniform_random_walks failed. " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* verts;
        hipgraph_type_erased_device_array_view_t* wgts;

        verts = hipgraph_random_walk_result_get_paths(result);
        wgts  = hipgraph_random_walk_result_get_weights(result);

        size_t verts_size = hipgraph_type_erased_device_array_view_size(verts);
        size_t wgts_size  = hipgraph_type_erased_device_array_view_size(wgts);

        vertex_t h_result_verts[verts_size];
        weight_t h_result_wgts[wgts_size];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_result_verts, verts, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_result_wgts, wgts, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        //  NOTE:  The C++ tester does a more thorough validation.  For our purposes
        //  here we will do a simpler validation, merely checking that all edges
        //  are actually part of the graph
        weight_t M[num_vertices][num_vertices];

        for(int i = 0; i < num_vertices; ++i)
            for(int j = 0; j < num_vertices; ++j)
                M[i][j] = -1;

        for(int i = 0; i < num_edges; ++i)
            M[h_src[i]][h_dst[i]] = h_wgt[i];

        EXPECT_EQ(hipgraph_random_walk_result_get_max_path_length(result), max_depth)
            << "path length does not match";

        for(int i = 0; (i < num_starts) && (test_ret_value == 0); ++i)
        {
            EXPECT_EQ(h_start[i], h_result_verts[i * (max_depth + 1)]) << "start of path not found";
            for(size_t j = 0; j < max_depth; ++j)
            {
                int src_index = i * (max_depth + 1) + j;
                int dst_index = src_index + 1;
                if(h_result_verts[dst_index] < 0)
                {
                    if(h_result_verts[src_index] >= 0)
                    {
                        int departing_count = 0;
                        for(int k = 0; k < num_vertices; ++k)
                        {
                            if(M[h_result_verts[src_index]][k] >= 0)
                                departing_count++;
                        }
                        EXPECT_EQ(departing_count, 0)
                            << "uniform_random_walks found no edge when an edge exists";
                    }
                }
                else
                {
                    EXPECT_EQ(M[h_result_verts[src_index]][h_result_verts[dst_index]],
                              h_result_wgts[i * max_depth + j])
                        << "uniform_random_walks got edge that doesn't exist";
                }
            }
        }

        hipgraph_random_walk_result_free(result);
        hipgraph_mg_graph_free(graph);
        hipgraph_error_free(ret_error);

        return test_ret_value;
    }

    int generic_biased_random_walks_test(const hipgraph_resource_handle_t* p_handle,
                                         vertex_t*                         h_src,
                                         vertex_t*                         h_dst,
                                         weight_t*                         h_wgt,
                                         size_t                            num_vertices,
                                         size_t                            num_edges,
                                         vertex_t*                         h_start,
                                         size_t                            num_starts,
                                         size_t                            max_depth,
                                         hipgraph_bool_t                   store_transposed)
    {
        int test_ret_value = 0;

        hipgraph_error_code_t ret_code  = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error = nullptr;

        hipgraph_graph_t*              graph  = nullptr;
        hipgraph_random_walk_result_t* result = nullptr;

        hipgraph_type_erased_device_array_t*      d_start      = nullptr;
        hipgraph_type_erased_device_array_view_t* d_start_view = nullptr;

        ret_code = create_mg_test_graph(p_handle,
                                        h_src,
                                        h_dst,
                                        h_wgt,
                                        num_edges,
                                        store_transposed,
                                        HIPGRAPH_FALSE,
                                        &graph,
                                        &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "graph creation failed.";

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_starts, HIPGRAPH_INT32, &d_start, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "d_start create failed.";

        d_start_view = hipgraph_type_erased_device_array_view(d_start);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, d_start_view, (hipgraph_byte_t*)h_start, &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "start copy_from_host failed.";

        ret_code = hipgraph_biased_random_walks(
            p_handle, graph, d_start_view, HIPGRAPH_FALSE, &result, &ret_error);

#if 1
        EXPECT_NE(ret_code, HIPGRAPH_SUCCESS) << "biased_random_walks should have failed";
#else
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "biased_random_walks failed. " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* verts;
        hipgraph_type_erased_device_array_view_t* wgts;

        verts = hipgraph_random_walk_result_get_paths(result);
        wgts  = hipgraph_random_walk_result_get_weights(result);

        size_t verts_size = hipgraph_type_erased_device_array_view_size(verts);
        size_t wgts_size  = hipgraph_type_erased_device_array_view_size(wgts);

        vertex_t h_result_verts[verts_size];
        vertex_t h_result_wgts[wgts_size];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            handle, (hipgraph_byte_t*)h_verts, verts, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            handle, (hipgraph_byte_t*)h_result_wgts, wgts, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        //  NOTE:  The C++ tester does a more thorough validation.  For our purposes
        //  here we will do a simpler validation, merely checking that all edges
        //  are actually part of the graph
        weight_t M[num_vertices][num_vertices];

        for(int i = 0; i < num_vertices; ++i)
            for(int j = 0; j < num_vertices; ++j)
                M[i][j] = -1;

        for(int i = 0; i < num_edges; ++i)
            M[h_src[i]][h_dst[i]] = h_wgt[i];

        EXPECT_EQ(hipgraph_random_walk_result_get_max_path_length(), max_depth)
            << "path length does not match";

        for(int i = 0; (i < num_starts) && (test_ret_value == 0); ++i)
        {
            EXPECT_EQ(M[h_start[i]][h_result_verts[i * (max_depth + 1)]],
                      h_result_wgts[i * max_depth])
                << "biased_random_walks got edge that doesn't exist";
            for(size_t j = 1; j < hipgraph_random_walk_result_get_max_path_length(); ++j)
                EXPECT_EQ(M[h_start[i * (max_depth + 1) + j - 1]]
                           [h_result_verts[i * (max_depth + 1) + j]],
                          h_result_wgts[i * max_depth + j - 1])
                    << "biased_random_walks got edge that doesn't exist";
        }

        hipgraph_random_walk_result_free(result);
#endif

        hipgraph_mg_graph_free(graph);
        hipgraph_error_free(ret_error);

        return test_ret_value;
    }

    int generic_node2vec_random_walks_test(const hipgraph_resource_handle_t* p_handle,
                                           vertex_t*                         h_src,
                                           vertex_t*                         h_dst,
                                           weight_t*                         h_wgt,
                                           size_t                            num_vertices,
                                           size_t                            num_edges,
                                           vertex_t*                         h_start,
                                           size_t                            num_starts,
                                           size_t                            max_depth,
                                           float                             p,
                                           float                             q,
                                           hipgraph_bool_t                   store_transposed)
    {
        int test_ret_value = 0;

        hipgraph_error_code_t ret_code  = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error = nullptr;

        hipgraph_graph_t*              graph  = nullptr;
        hipgraph_random_walk_result_t* result = nullptr;

        hipgraph_type_erased_device_array_t*      d_start      = nullptr;
        hipgraph_type_erased_device_array_view_t* d_start_view = nullptr;

        ret_code = create_mg_test_graph(p_handle,
                                        h_src,
                                        h_dst,
                                        h_wgt,
                                        num_edges,
                                        store_transposed,
                                        HIPGRAPH_FALSE,
                                        &graph,
                                        &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "graph creation failed.";

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_starts, HIPGRAPH_INT32, &d_start, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "d_start create failed.";

        d_start_view = hipgraph_type_erased_device_array_view(d_start);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, d_start_view, (hipgraph_byte_t*)h_start, &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "start copy_from_host failed.";

        ret_code = hipgraph_node2vec_random_walks(
            p_handle, graph, d_start_view, HIPGRAPH_FALSE, p, q, &result, &ret_error);

#if 1
        EXPECT_NE(ret_code, HIPGRAPH_SUCCESS) << "node2vec_random_walks should have failed";
#else
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "node2vec_random_walks failed. " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* verts;
        hipgraph_type_erased_device_array_view_t* wgts;

        verts = hipgraph_random_walk_result_get_paths(result);
        wgts  = hipgraph_random_walk_result_get_weights(result);

        size_t verts_size = hipgraph_type_erased_device_array_view_size(verts);
        size_t wgts_size  = hipgraph_type_erased_device_array_view_size(wgts);

        vertex_t h_result_verts[verts_size];
        vertex_t h_result_wgts[wgts_size];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            handle, (hipgraph_byte_t*)h_verts, verts, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            handle, (hipgraph_byte_t*)h_result_wgts, wgts, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        //  NOTE:  The C++ tester does a more thorough validation.  For our purposes
        //  here we will do a simpler validation, merely checking that all edges
        //  are actually part of the graph
        weight_t M[num_vertices][num_vertices];

        for(int i = 0; i < num_vertices; ++i)
            for(int j = 0; j < num_vertices; ++j)
                M[i][j] = -1;

        for(int i = 0; i < num_edges; ++i)
            M[h_src[i]][h_dst[i]] = h_wgt[i];

        EXPECT_EQ(hipgraph_random_walk_result_get_max_path_length(), max_depth)
            << "path length does not match";

        for(int i = 0; (i < num_starts) && (test_ret_value == 0); ++i)
        {
            EXPECT_EQ(M[h_start[i]][h_result_verts[i * (max_depth + 1)]],
                      h_result_wgts[i * max_depth])
                << "node2vec_random_walks got edge that doesn't exist";
            for(size_t j = 1; j < hipgraph_random_walk_result_get_max_path_length(); ++j)
                EXPECT_EQ(M[h_start[i * (max_depth + 1) + j - 1]]
                           [h_result_verts[i * (max_depth + 1) + j]],
                          h_result_wgts[i * max_depth + j - 1])
                    << "node2vec_random_walks got edge that doesn't exist";
        }

        hipgraph_random_walk_result_free(result);
#endif

        hipgraph_mg_graph_free(graph);
        hipgraph_error_free(ret_error);

        return test_ret_value;
    }

    int test_uniform_random_walks(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 8;
        size_t num_vertices = 6;
        size_t num_starts   = 2;
        size_t max_depth    = 3;

        vertex_t src[]   = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]   = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t wgt[]   = {0, 1, 2, 3, 4, 5, 6, 7};
        vertex_t start[] = {2, 2};

        return generic_uniform_random_walks_test(p_handle,
                                                 src,
                                                 dst,
                                                 wgt,
                                                 num_vertices,
                                                 num_edges,
                                                 start,
                                                 num_starts,
                                                 max_depth,
                                                 HIPGRAPH_FALSE);
    }

    int test_biased_random_walks(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 8;
        size_t num_vertices = 6;
        size_t num_starts   = 2;

        vertex_t src[]   = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]   = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t wgt[]   = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        vertex_t start[] = {2, 2};

        return generic_biased_random_walks_test(p_handle,
                                                src,
                                                dst,
                                                wgt,
                                                num_vertices,
                                                num_edges,
                                                start,
                                                num_starts,
                                                HIPGRAPH_FALSE,
                                                HIPGRAPH_FALSE);
    }

    int test_node2vec_random_walks(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 8;
        size_t num_vertices = 6;
        size_t num_starts   = 2;

        vertex_t src[]   = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]   = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t wgt[]   = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        vertex_t start[] = {2, 2};

        weight_t p = 5;
        weight_t q = 8;

        return generic_node2vec_random_walks_test(p_handle,
                                                  src,
                                                  dst,
                                                  wgt,
                                                  num_vertices,
                                                  num_edges,
                                                  start,
                                                  num_starts,
                                                  p,
                                                  q,
                                                  HIPGRAPH_FALSE,
                                                  HIPGRAPH_FALSE);
    }
