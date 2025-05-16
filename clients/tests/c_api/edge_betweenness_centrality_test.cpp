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

#include "hipgraph_c/algorithms.h"
#include "hipgraph_c/graph.h"
#include "hipgraph_c/random.h"

#include <cmath>

using vertex_t = int32_t;
using edge_t   = int32_t;
using weight_t = float;

namespace
{
    using namespace hipGRAPH::testing;
    void generic_edge_betweenness_centrality_test(vertex_t*       h_src,
                                                  vertex_t*       h_dst,
                                                  weight_t*       h_wgt,
                                                  vertex_t*       h_seeds,
                                                  weight_t*       h_result,
                                                  size_t          num_vertices,
                                                  size_t          num_edges,
                                                  size_t          num_seeds,
                                                  hipgraph_bool_t store_transposed,
                                                  size_t          num_vertices_to_sample)
    {
        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_resource_handle_t*               p_handle   = nullptr;
        hipgraph_graph_t*                         graph      = nullptr;
        hipgraph_edge_centrality_result_t*        result     = nullptr;
        hipgraph_rng_state_t*                     rng_state  = nullptr;
        hipgraph_type_erased_device_array_t*      seeds      = nullptr;
        hipgraph_type_erased_device_array_view_t* seeds_view = nullptr;

        p_handle = hipgraph_create_resource_handle(nullptr);
        ASSERT_NE(p_handle, nullptr) << "resource handle creation failed.";

        ret_code = hipgraph_rng_state_create(p_handle, 0, &rng_state, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "failed to create rng_state: " << hipgraph_error_message(ret_error);

        create_test_graph(p_handle,
                          h_src,
                          h_dst,
                          h_wgt,
                          num_edges,
                          store_transposed,
                          HIPGRAPH_FALSE,
                          HIPGRAPH_FALSE,
                          &graph,
                          &ret_error);

        if(h_seeds == nullptr)
        {
            ret_code = hipgraph_select_random_vertices(
                p_handle, graph, rng_state, num_vertices_to_sample, &seeds, &ret_error);
            ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
                << "select random seeds failed: " << hipgraph_error_message(ret_error);

            seeds_view = hipgraph_type_erased_device_array_view(seeds);
        }
        else
        {
            ret_code = hipgraph_type_erased_device_array_create(
                p_handle, num_seeds, HIPGRAPH_INT32, &seeds, &ret_error);
            ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
                << "seeds create failed: " << hipgraph_error_message(ret_error);

            seeds_view = hipgraph_type_erased_device_array_view(seeds);
            ret_code   = hipgraph_type_erased_device_array_view_copy_from_host(
                p_handle, seeds_view, (hipgraph_byte_t*)h_seeds, &ret_error);
            ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
                << "seeds copy_from_host failed: " << hipgraph_error_message(ret_error);
        }

        ret_code = hipgraph_edge_betweenness_centrality(
            p_handle, graph, seeds_view, HIPGRAPH_FALSE, HIPGRAPH_FALSE, &result, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "hipgraph_edge_betweenness_centrality failed: " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* srcs;
        hipgraph_type_erased_device_array_view_t* dsts;
        hipgraph_type_erased_device_array_view_t* centralities;

        srcs         = hipgraph_edge_centrality_result_get_src_vertices(result);
        dsts         = hipgraph_edge_centrality_result_get_dst_vertices(result);
        centralities = hipgraph_edge_centrality_result_get_values(result);

        size_t num_local_edges = hipgraph_type_erased_device_array_view_size(srcs);

        vertex_t h_hipgraph_src[num_local_edges];
        vertex_t h_hipgraph_dst[num_local_edges];
        weight_t h_centralities[num_local_edges];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_hipgraph_src, srcs, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_hipgraph_dst, dsts, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_centralities, centralities, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        weight_t M[num_vertices][num_vertices];

        for(size_t i = 0; i < num_vertices; ++i)
            for(size_t j = 0; j < num_vertices; ++j)
            {
                M[i][j] = 0.0;
            }

        for(size_t i = 0; i < num_edges; ++i)
        {
            M[h_src[i]][h_dst[i]] = h_result[i];
        }

        for(size_t i = 0; i < num_local_edges; ++i)
        {
            EXPECT_NEAR(M[h_hipgraph_src[i]][h_hipgraph_dst[i]], h_centralities[i], 0.001)
                << "betweenness centrality results don't match at position " << i;
        }

        hipgraph_edge_centrality_result_free(result);
        hipgraph_sg_graph_free(graph);
        hipgraph_free_resource_handle(p_handle);
        hipgraph_error_free(ret_error);
    }

    TEST(AlgorithmTest, EdgeBetweennessCentrality)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 6;

        vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
        weight_t h_wgt[]    = {0.1f,
                               2.1f,
                               1.1f,
                               5.1f,
                               3.1f,
                               4.1f,
                               7.2f,
                               3.2f,
                               0.1f,
                               2.1f,
                               1.1f,
                               5.1f,
                               3.1f,
                               4.1f,
                               7.2f,
                               3.2f};
        weight_t h_result[] = {0,
                               2,
                               3,
                               1.83333,
                               2,
                               2,
                               3,
                               2,
                               3.16667,
                               2.83333,
                               4.33333,
                               0,
                               2,
                               2.83333,
                               3.66667,
                               2.33333};

        // Eigenvector centrality wants store_transposed = HIPGRAPH_TRUE
        generic_edge_betweenness_centrality_test(
            h_src, h_dst, h_wgt, nullptr, h_result, num_vertices, num_edges, 0, HIPGRAPH_TRUE, 5);
    }

} // namespace
