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
    void generic_betweenness_centrality_test(vertex_t*       h_src,
                                             vertex_t*       h_dst,
                                             weight_t*       h_wgt,
                                             vertex_t*       h_seeds,
                                             weight_t*       h_result,
                                             size_t          num_vertices,
                                             size_t          num_edges,
                                             size_t          num_seeds,
                                             hipgraph_bool_t store_transposed,
                                             hipgraph_bool_t is_symmetric,
                                             hipgraph_bool_t normalized,
                                             hipgraph_bool_t include_endpoints,
                                             size_t          num_vertices_to_sample)
    {
        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_resource_handle_t*               p_handle   = nullptr;
        hipgraph_graph_t*                         p_graph    = nullptr;
        hipgraph_centrality_result_t*             p_result   = nullptr;
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
                          is_symmetric,
                          &p_graph,
                          &ret_error);

        if(h_seeds == nullptr)
        {
            ret_code = hipgraph_select_random_vertices(
                p_handle, p_graph, rng_state, num_vertices_to_sample, &seeds, &ret_error);
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

        ret_code = hipgraph_betweenness_centrality(p_handle,
                                                   p_graph,
                                                   seeds_view,
                                                   normalized,
                                                   include_endpoints,
                                                   HIPGRAPH_FALSE,
                                                   &p_result,
                                                   &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "hipgraph_betweenness_centrality failed: " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* vertices;
        hipgraph_type_erased_device_array_view_t* centralities;

        vertices     = hipgraph_centrality_result_get_vertices(p_result);
        centralities = hipgraph_centrality_result_get_values(p_result);

        vertex_t h_vertices[num_vertices];
        weight_t h_centralities[num_vertices];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_vertices, vertices, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_centralities, centralities, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        for(size_t i = 0; i < num_vertices; ++i)
        {
            EXPECT_NEAR(h_result[h_vertices[i]], h_centralities[i], 0.0001)
                << "centralities results don't match at position " << i;
        }

        hipgraph_centrality_result_free(p_result);

        hipgraph_type_erased_device_array_view_free(seeds_view);
        hipgraph_type_erased_device_array_free(seeds);
        hipgraph_sg_graph_free(p_graph);
        hipgraph_free_resource_handle(p_handle);
        hipgraph_error_free(ret_error);
    }

    TEST(AlgorithmTest, BetweennessCentralityFull)
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
        weight_t h_result[] = {0, 3.66667, 0.833333, 2.16667, 0.833333, 0.5};

        generic_betweenness_centrality_test(h_src,
                                            h_dst,
                                            h_wgt,
                                            nullptr,
                                            h_result,
                                            num_vertices,
                                            num_edges,
                                            0,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_TRUE,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            6);
    }

    TEST(AlgorithmTest, BetweennessCentralityFullDirected)
    {
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        weight_t h_result[] = {0, 4, 0, 2, 1, 0};

        generic_betweenness_centrality_test(h_src,
                                            h_dst,
                                            h_wgt,
                                            nullptr,
                                            h_result,
                                            num_vertices,
                                            num_edges,
                                            0,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            6);
    }

    TEST(AlgorithmTest, BetweennessCentralitySpecificNormalized)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 6;
        size_t num_seeds    = 2;

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
        vertex_t h_seeds[]  = {0, 3};
        weight_t h_result[] = {0, 0.475, 0.2, 0.1, 0.05, 0.075};

        generic_betweenness_centrality_test(h_src,
                                            h_dst,
                                            h_wgt,
                                            h_seeds,
                                            h_result,
                                            num_vertices,
                                            num_edges,
                                            num_seeds,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_TRUE,
                                            HIPGRAPH_FALSE,
                                            num_seeds);
    }

    TEST(AlgorithmTest, BetweennessCentralitySpecificUnnormalized)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 6;
        size_t num_seeds    = 2;

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
        vertex_t h_seeds[]  = {0, 3};
        weight_t h_result[] = {0, 3.16667, 1.33333, 0.666667, 0.333333, 0.5};

        generic_betweenness_centrality_test(h_src,
                                            h_dst,
                                            h_wgt,
                                            h_seeds,
                                            h_result,
                                            num_vertices,
                                            num_edges,
                                            num_seeds,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            num_seeds);
    }

    TEST(AlgorithmTest, BetweennessCentralityTestEndpoints)
    {
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        weight_t h_result[] = {0.166667, 0.3, 0.166667, 0.2, 0.166667, 0.166667};

        generic_betweenness_centrality_test(h_src,
                                            h_dst,
                                            h_wgt,
                                            nullptr,
                                            h_result,
                                            num_vertices,
                                            num_edges,
                                            0,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_TRUE,
                                            HIPGRAPH_TRUE,
                                            6);
    }

    TEST(AlgorithmTest, BetweennessCentralityFullDirectedNormalizedKarate)
    {
        size_t num_edges    = 156;
        size_t num_vertices = 34;

        vertex_t h_src[]
            = {1,  2,  3,  4,  5,  6,  7,  8,  10, 11, 12, 13, 17, 19, 21, 31, 2,  3,  7,  13,
               17, 19, 21, 30, 3,  7,  8,  9,  13, 27, 28, 32, 7,  12, 13, 6,  10, 6,  10, 16,
               16, 30, 32, 33, 33, 33, 32, 33, 32, 33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29,
               32, 33, 25, 27, 31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33, 0,  0,
               0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,
               1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  4,  4,  5,  5,  5,  6,  8,
               8,  8,  9,  13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22, 23, 23, 23, 23, 23,
               24, 24, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32};

        vertex_t h_dst[]
            = {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,
               1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  4,  4,  5,  5,  5,
               6,  8,  8,  8,  9,  13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22, 23, 23, 23,
               23, 23, 24, 24, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 1,  2,
               3,  4,  5,  6,  7,  8,  10, 11, 12, 13, 17, 19, 21, 31, 2,  3,  7,  13, 17, 19,
               21, 30, 3,  7,  8,  9,  13, 27, 28, 32, 7,  12, 13, 6,  10, 6,  10, 16, 16, 30,
               32, 33, 33, 33, 32, 33, 32, 33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29, 32, 33,
               25, 27, 31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33};

        weight_t h_wgt[]
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

        weight_t h_result[]
            = {462.142914, 56.957146, 151.701584, 12.576191,  0.666667,   31.666668, 31.666668,
               0.000000,   59.058739, 0.895238,   0.666667,   0.000000,   0.000000,  48.431747,
               0.000000,   0.000000,  0.000000,   0.000000,   0.000000,   34.293652, 0.000000,
               0.000000,   0.000000,  18.600000,  2.333333,   4.055556,   0.000000,  23.584126,
               1.895238,   3.085714,  15.219049,  146.019043, 153.380981, 321.103180};

        generic_betweenness_centrality_test(h_src,
                                            h_dst,
                                            h_wgt,
                                            nullptr,
                                            h_result,
                                            num_vertices,
                                            num_edges,
                                            0,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            HIPGRAPH_FALSE,
                                            34);
    }

} // namespace
