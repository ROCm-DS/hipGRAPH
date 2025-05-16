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
    int generic_betweenness_centrality_test(const hipgraph_resource_handle_t* p_handle,
                                            vertex_t*                         h_src,
                                            vertex_t*                         h_dst,
                                            weight_t*                         h_wgt,
                                            vertex_t*                         h_seeds,
                                            weight_t*                         h_result,
                                            size_t                            num_vertices,
                                            size_t                            num_edges,
                                            size_t                            num_seeds,
                                            hipgraph_bool_t                   store_transposed,
                                            hipgraph_bool_t                   is_symmetric,
                                            hipgraph_bool_t                   normalized,
                                            hipgraph_bool_t                   include_endpoints,
                                            size_t num_vertices_to_sample)
    {
        int test_ret_value = 0;

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_graph_t*                         p_graph    = nullptr;
        hipgraph_centrality_result_t*             p_result   = nullptr;
        hipgraph_rng_state_t*                     rng_state  = nullptr;
        hipgraph_type_erased_device_array_t*      seeds      = nullptr;
        hipgraph_type_erased_device_array_view_t* seeds_view = nullptr;

        int rank = hipgraph_resource_handle_get_rank(p_handle);

        ret_code = hipgraph_rng_state_create(p_handle, rank, &rng_state, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "failed to create rng_state.";

        ret_code = create_mg_test_graph(p_handle,
                                        h_src,
                                        h_dst,
                                        h_wgt,
                                        num_edges,
                                        store_transposed,
                                        is_symmetric,
                                        &p_graph,
                                        &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "create_mg_test_graph failed.";

        if(h_seeds == nullptr)
        {
            ret_code = hipgraph_select_random_vertices(
                p_handle, p_graph, rng_state, num_vertices_to_sample, &seeds, &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "select random seeds failed.";

            seeds_view = hipgraph_type_erased_device_array_view(seeds);
        }
        else
        {
            if(rank > 0)
                num_seeds = 0;

            ret_code = hipgraph_type_erased_device_array_create(
                p_handle, num_seeds, HIPGRAPH_INT32, &seeds, &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "seeds create failed.";

            seeds_view = hipgraph_type_erased_device_array_view(seeds);
            ret_code   = hipgraph_type_erased_device_array_view_copy_from_host(
                p_handle, seeds_view, (hipgraph_byte_t*)h_seeds, &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "seeds copy_from_host failed.";
        }

        ret_code = hipgraph_betweenness_centrality(p_handle,
                                                   p_graph,
                                                   seeds_view,
                                                   normalized,
                                                   include_endpoints,
                                                   HIPGRAPH_FALSE,
                                                   &p_result,
                                                   &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "hipgraph_betweenness_centrality failed.";

        // NOTE: Because we get back vertex ids and centralities, we can simply compare
        //       the returned values with the expected results for the entire
        //       graph.  Each GPU will have a subset of the total vertices, so
        //       they will do a subset of the comparisons.
        hipgraph_type_erased_device_array_view_t* vertices;
        hipgraph_type_erased_device_array_view_t* centralities;

        vertices     = hipgraph_centrality_result_get_vertices(p_result);
        centralities = hipgraph_centrality_result_get_values(p_result);

        vertex_t h_vertices[num_vertices];
        weight_t h_centralities[num_vertices];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_vertices, vertices, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_centralities, centralities, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        size_t num_local_vertices = hipgraph_type_erased_device_array_view_size(vertices);

        for(int i = 0; (i < num_local_vertices) && (test_ret_value == 0); ++i)
        {
            EXPECT_NEAR(h_result[h_vertices[i]], h_centralities[i], 0.00001)
                << "betweenness centrality results don't match";
        }

        hipgraph_centrality_result_free(p_result);

        hipgraph_type_erased_device_array_view_free(seeds_view);
        hipgraph_type_erased_device_array_free(seeds);
        hipgraph_mg_graph_free(p_graph);
        hipgraph_error_free(ret_error);

        return test_ret_value;
    }

    int test_betweenness_centrality(const hipgraph_resource_handle_t* p_handle)
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
        weight_t h_result[] = {0, 3.66667, 0.83333, 2.16667, 0.83333, 0.5};

        // NOTE: Randomly selecting vertices in MG varies by the GPU topology,
        //  so we'll specify selecting all to get deterministic results for the test.
        //
        // Betweenness centrality wants store_transposed = HIPGRAPH_FALSE
        return generic_betweenness_centrality_test(p_handle,
                                                   h_src,
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
                                                   num_vertices);
    }
    int test_betweenness_centrality_normalized(const hipgraph_resource_handle_t* p_handle)
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
        weight_t h_result[] = {0, .366667, .083333, .216667, 0.0833333, 0.05};

        // NOTE: Randomly selecting vertices in MG varies by the GPU topology,
        //  so we'll specify selecting all to get deterministic results for the test.
        //
        // Betweenness centrality wants store_transposed = HIPGRAPH_FALSE
        return generic_betweenness_centrality_test(p_handle,
                                                   h_src,
                                                   h_dst,
                                                   h_wgt,
                                                   nullptr,
                                                   h_result,
                                                   num_vertices,
                                                   num_edges,
                                                   0,
                                                   HIPGRAPH_FALSE,
                                                   HIPGRAPH_TRUE,
                                                   HIPGRAPH_TRUE,
                                                   HIPGRAPH_FALSE,
                                                   num_vertices);
    }

    int test_betweenness_centrality_full(const hipgraph_resource_handle_t* p_handle)
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

        return generic_betweenness_centrality_test(p_handle,
                                                   h_src,
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

    int test_betweenness_centrality_full_directed(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        weight_t h_result[] = {0, 4, 0, 2, 1, 0};

        return generic_betweenness_centrality_test(p_handle,
                                                   h_src,
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

    int test_betweenness_centrality_specific_normalized(const hipgraph_resource_handle_t* p_handle)
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

        return generic_betweenness_centrality_test(p_handle,
                                                   h_src,
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

    int test_betweenness_centrality_specific_unnormalized(
        const hipgraph_resource_handle_t* p_handle)
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

        return generic_betweenness_centrality_test(p_handle,
                                                   h_src,
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

    int test_betweenness_centrality_test_endpoints(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        weight_t h_result[] = {0.166667, 0.3, 0.166667, 0.2, 0.166667, 0.166667};

        return generic_betweenness_centrality_test(p_handle,
                                                   h_src,
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

    int test_betweenness_centrality_full_directed_normalized_karate(
        const hipgraph_resource_handle_t* p_handle)
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

        return generic_betweenness_centrality_test(p_handle,
                                                   h_src,
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
