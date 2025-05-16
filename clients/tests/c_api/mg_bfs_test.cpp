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
    int generic_bfs_test(const hipgraph_resource_handle_t* p_handle,
                         vertex_t*                         h_src,
                         vertex_t*                         h_dst,
                         weight_t*                         h_wgt,
                         vertex_t*                         h_seeds,
                         vertex_t const*                   expected_distances,
                         vertex_t const*                   expected_predecessors,
                         size_t                            num_vertices,
                         size_t                            num_edges,
                         size_t                            num_seeds,
                         size_t                            depth_limit,
                         hipgraph_bool_t                   store_transposed)
    {
        int test_ret_value = 0;

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_graph_t*                         p_graph       = nullptr;
        hipgraph_paths_result_t*                  paths_result  = nullptr;
        hipgraph_type_erased_device_array_t*      p_sources     = nullptr;
        hipgraph_type_erased_device_array_view_t* p_source_view = nullptr;

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_seeds, HIPGRAPH_INT32, &p_sources, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "p_sources create failed.";

        p_source_view = hipgraph_type_erased_device_array_view(p_sources);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, p_source_view, (hipgraph_byte_t*)h_seeds, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "src copy_from_host failed.";

        ret_code = create_mg_test_graph(p_handle,
                                        h_src,
                                        h_dst,
                                        h_wgt,
                                        num_edges,
                                        store_transposed,
                                        HIPGRAPH_FALSE,
                                        &p_graph,
                                        &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "create_mg_test_graph failed.";

        ret_code = hipgraph_bfs(p_handle,
                                p_graph,
                                p_source_view,
                                HIPGRAPH_FALSE,
                                10000000,
                                HIPGRAPH_TRUE,
                                HIPGRAPH_TRUE,
                                &paths_result,
                                &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "hipgraph_bfs failed. " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* vertices;
        hipgraph_type_erased_device_array_view_t* distances;
        hipgraph_type_erased_device_array_view_t* predecessors;

        vertices     = hipgraph_paths_result_get_vertices(paths_result);
        predecessors = hipgraph_paths_result_get_predecessors(paths_result);
        distances    = hipgraph_paths_result_get_distances(paths_result);

        vertex_t h_vertices[num_vertices];
        vertex_t h_predecessors[num_vertices];
        vertex_t h_distances[num_vertices];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_vertices, vertices, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_distances, distances, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_predecessors, predecessors, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        size_t num_local_vertices = hipgraph_type_erased_device_array_view_size(vertices);

        for(int i = 0; (i < num_local_vertices) && (test_ret_value == 0); ++i)
        {
            EXPECT_EQ(expected_distances[h_vertices[i]], h_distances[i])
                << "bfs distances don't match";

            EXPECT_EQ(expected_predecessors[h_vertices[i]], h_predecessors[i])
                << "bfs predecessors don't match";
        }

        hipgraph_paths_result_free(paths_result);
        hipgraph_mg_graph_free(p_graph);
        hipgraph_error_free(ret_error);

        return test_ret_value;
    }

    int test_bfs(const hipgraph_resource_handle_t* p_handle)
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
        return generic_bfs_test(p_handle,
                                src,
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
