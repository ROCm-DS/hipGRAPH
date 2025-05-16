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

#include <cmath>

using vertex_t = int32_t;
using edge_t   = int32_t;
using weight_t = float;

namespace
{
    using namespace hipGRAPH::testing;
    void generic_bfs_test(vertex_t*       h_src,
                          vertex_t*       h_dst,
                          weight_t*       h_wgt,
                          vertex_t*       h_seeds,
                          vertex_t const* expected_distances,
                          vertex_t const* expected_predecessors,
                          size_t          num_vertices,
                          size_t          num_edges,
                          size_t          num_seeds,
                          size_t          depth_limit,
                          hipgraph_bool_t store_transposed)
    {
        hipgraph_error_code_t ret_code  = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error = nullptr;

        hipgraph_resource_handle_t*               p_handle      = nullptr;
        hipgraph_graph_t*                         p_graph       = nullptr;
        hipgraph_paths_result_t*                  p_result      = nullptr;
        hipgraph_type_erased_device_array_t*      p_sources     = nullptr;
        hipgraph_type_erased_device_array_view_t* p_source_view = nullptr;

        p_handle = hipgraph_create_resource_handle(nullptr);
        ASSERT_NE(p_handle, nullptr) << "resource handle creation failed.";

        create_test_graph(p_handle,
                          h_src,
                          h_dst,
                          h_wgt,
                          num_edges,
                          store_transposed,
                          HIPGRAPH_FALSE,
                          HIPGRAPH_FALSE,
                          &p_graph,
                          &ret_error);

        /*
        * FIXME: in create_graph_test.c, variables are defined but then hard-coded to
        * the constant HIPGRAPH_INT32. It would be better to pass the types into the functions
        * in both cases so that the test cases could be parameterized in the main.
        */
        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_seeds, HIPGRAPH_INT32, &p_sources, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "p_sources create failed: " << hipgraph_error_message(ret_error);

        p_source_view = hipgraph_type_erased_device_array_view(p_sources);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, p_source_view, (hipgraph_byte_t*)h_seeds, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "h_seeds copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_bfs(p_handle,
                                p_graph,
                                p_source_view,
                                HIPGRAPH_FALSE,
                                depth_limit,
                                HIPGRAPH_TRUE,
                                HIPGRAPH_FALSE,
                                &p_result,
                                &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "hipgraph_bfs failed: " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* vertices;
        hipgraph_type_erased_device_array_view_t* distances;
        hipgraph_type_erased_device_array_view_t* predecessors;

        vertices     = hipgraph_paths_result_get_vertices(p_result);
        distances    = hipgraph_paths_result_get_distances(p_result);
        predecessors = hipgraph_paths_result_get_predecessors(p_result);

        vertex_t h_vertices[num_vertices];
        vertex_t h_distances[num_vertices];
        vertex_t h_predecessors[num_vertices];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_vertices, vertices, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "vertices copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_distances, distances, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "distances copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_predecessors, predecessors, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "predecessors copy_to_host failed: " << hipgraph_error_message(ret_error);

        for(size_t i = 0; i < num_vertices; ++i)
        {
            EXPECT_EQ(expected_distances[h_vertices[i]], h_distances[i])
                << "bfs distances don't match at position " << i;

            EXPECT_EQ(expected_predecessors[h_vertices[i]], h_predecessors[i])
                << "bfs predecessors don't match at position " << i;
        }

        hipgraph_type_erased_device_array_free(p_sources);
        hipgraph_paths_result_free(p_result);
        hipgraph_sg_graph_free(p_graph);
        hipgraph_free_resource_handle(p_handle);
        hipgraph_error_free(ret_error);
    };

    TEST(AlgorithmTest, BfsExceptions)
    {
        size_t num_edges   = 8;
        size_t depth_limit = 1;
        size_t num_seeds   = 1;

        vertex_t src[]   = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]   = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t wgt[]   = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        int64_t  seeds[] = {0};

        hipgraph_error_code_t ret_code  = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error = nullptr;

        hipgraph_resource_handle_t*               p_handle      = nullptr;
        hipgraph_graph_t*                         p_graph       = nullptr;
        hipgraph_paths_result_t*                  p_result      = nullptr;
        hipgraph_type_erased_device_array_t*      p_sources     = nullptr;
        hipgraph_type_erased_device_array_view_t* p_source_view = nullptr;

        p_handle = hipgraph_create_resource_handle(nullptr);
        ASSERT_NE(p_handle, nullptr) << "resource handle creation failed.";

        create_test_graph(p_handle,
                          src,
                          dst,
                          wgt,
                          num_edges,
                          HIPGRAPH_FALSE,
                          HIPGRAPH_FALSE,
                          HIPGRAPH_FALSE,
                          &p_graph,
                          &ret_error);

        /*
        * FIXME: in create_graph_test.c, variables are defined but then hard-coded to
        * the constant HIPGRAPH_INT32. It would be better to pass the types into the functions
        * in both cases so that the test cases could be parameterized in the main.
        */
        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_seeds, HIPGRAPH_INT64, &p_sources, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "p_sources create failed: " << hipgraph_error_message(ret_error);

        p_source_view = hipgraph_type_erased_device_array_view(p_sources);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, p_source_view, (hipgraph_byte_t*)seeds, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "seeds copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_bfs(p_handle,
                                p_graph,
                                p_source_view,
                                HIPGRAPH_FALSE,
                                depth_limit,
                                HIPGRAPH_TRUE,
                                HIPGRAPH_FALSE,
                                &p_result,
                                &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_INVALID_INPUT) << "hipgraph_bfs expected to fail";
    }

    TEST(AlgorithmTest, Bfs)
    {
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t src[]                   = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]                   = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t wgt[]                   = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t seeds[]                 = {0};
        vertex_t expected_distances[]    = {0, 1, 2147483647, 2, 2, 3};
        vertex_t expected_predecessors[] = {-1, 0, -1, 1, 1, 3};

        // Bfs wants store_transposed = HIPGRAPH_FALSE
        generic_bfs_test(src,
                         dst,
                         wgt,
                         seeds,
                         expected_distances,
                         expected_predecessors,
                         num_vertices,
                         num_edges,
                         1,
                         10,
                         HIPGRAPH_FALSE);
    }

    TEST(AlgorithmTest, BfsWithTranspose)
    {
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t src[]                   = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]                   = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t wgt[]                   = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t seeds[]                 = {0};
        vertex_t expected_distances[]    = {0, 1, 2147483647, 2, 2, 3};
        vertex_t expected_predecessors[] = {-1, 0, -1, 1, 1, 3};

        // Bfs wants store_transposed = HIPGRAPH_FALSE
        //    This call will force hipgraph_bfs to transpose the graph
        generic_bfs_test(src,
                         dst,
                         wgt,
                         seeds,
                         expected_distances,
                         expected_predecessors,
                         num_vertices,
                         num_edges,
                         1,
                         10,
                         HIPGRAPH_TRUE);
    }

} // namespace
