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
#include "hipgraph_c/array.h"
#include "hipgraph_c/graph.h"

#include <cmath>

using vertex_t = int32_t;
using edge_t   = int32_t;
using weight_t = float;

typedef enum
{
    JACCARD,
    SORENSEN,
    OVERLAP
} similarity_t;

namespace
{
    using namespace hipGRAPH::testing;
    void generic_similarity_test(vertex_t*       h_src,
                                 vertex_t*       h_dst,
                                 weight_t*       h_wgt,
                                 vertex_t*       h_first,
                                 vertex_t*       h_second,
                                 weight_t*       h_result,
                                 size_t          num_vertices,
                                 size_t          num_edges,
                                 size_t          num_pairs,
                                 hipgraph_bool_t store_transposed,
                                 hipgraph_bool_t use_weight,
                                 similarity_t    test_type)
    {
        hipgraph_data_type_id_t vertex_tid = HIPGRAPH_INT32;

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_resource_handle_t*               p_handle     = nullptr;
        hipgraph_graph_t*                         graph        = nullptr;
        hipgraph_similarity_result_t*             result       = nullptr;
        hipgraph_vertex_pairs_t*                  vertex_pairs = nullptr;
        hipgraph_type_erased_device_array_t*      v1           = nullptr;
        hipgraph_type_erased_device_array_t*      v2           = nullptr;
        hipgraph_type_erased_device_array_view_t* v1_view      = nullptr;
        hipgraph_type_erased_device_array_view_t* v2_view      = nullptr;

        p_handle = hipgraph_create_resource_handle(nullptr);
        ASSERT_NE(p_handle, nullptr) << "resource handle creation failed.";

        create_test_graph(p_handle,
                          h_src,
                          h_dst,
                          h_wgt,
                          num_edges,
                          store_transposed,
                          HIPGRAPH_FALSE,
                          HIPGRAPH_TRUE,
                          &graph,
                          &ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_pairs, vertex_tid, &v1, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "v1 create failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_pairs, vertex_tid, &v2, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "v2 create failed: " << hipgraph_error_message(ret_error);

        v1_view = hipgraph_type_erased_device_array_view(v1);
        v2_view = hipgraph_type_erased_device_array_view(v2);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, v1_view, (hipgraph_byte_t*)h_first, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "h_first copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, v2_view, (hipgraph_byte_t*)h_second, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "h_second copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_create_vertex_pairs(
            p_handle, graph, v1_view, v2_view, &vertex_pairs, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "create vertex pairs failed: " << hipgraph_error_message(ret_error);

        switch(test_type)
        {
        case JACCARD:
            ret_code = hipgraph_jaccard_coefficients(
                p_handle, graph, vertex_pairs, use_weight, HIPGRAPH_FALSE, &result, &ret_error);
            break;
        case SORENSEN:
            ret_code = hipgraph_sorensen_coefficients(
                p_handle, graph, vertex_pairs, use_weight, HIPGRAPH_FALSE, &result, &ret_error);
            break;
        case OVERLAP:
            ret_code = hipgraph_overlap_coefficients(
                p_handle, graph, vertex_pairs, use_weight, HIPGRAPH_FALSE, &result, &ret_error);
            break;
        }

        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "hipgraph similarity failed: " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* similarity_coefficient;

        similarity_coefficient = hipgraph_similarity_result_get_similarity(result);

        weight_t h_similarity_coefficient[num_pairs];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle,
            (hipgraph_byte_t*)h_similarity_coefficient,
            similarity_coefficient,
            &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        for(size_t i = 0; i < num_pairs; ++i)
        {
            EXPECT_NEAR(h_similarity_coefficient[i], h_result[i], 0.001)
                << "similarity results don't match at position " << i;
        }

        if(result != nullptr)
            hipgraph_similarity_result_free(result);
        if(vertex_pairs != nullptr)
            hipgraph_vertex_pairs_free(vertex_pairs);
        hipgraph_sg_graph_free(graph);
        hipgraph_free_resource_handle(p_handle);
        hipgraph_error_free(ret_error);
    }

    void generic_all_pairs_similarity_test(vertex_t*       h_src,
                                           vertex_t*       h_dst,
                                           weight_t*       h_wgt,
                                           vertex_t*       h_first,
                                           vertex_t*       h_second,
                                           weight_t*       h_result,
                                           size_t          num_vertices,
                                           size_t          num_edges,
                                           size_t          num_pairs,
                                           hipgraph_bool_t store_transposed,
                                           hipgraph_bool_t use_weight,
                                           size_t          topk,
                                           similarity_t    test_type)
    {
        // hipgraph_data_type_id_t vertex_tid     = HIPGRAPH_INT32;

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_resource_handle_t*   p_handle = nullptr;
        hipgraph_graph_t*             graph    = nullptr;
        hipgraph_similarity_result_t* result   = nullptr;
        // hipgraph_type_erased_device_array_t*      vertices      = nullptr;
        hipgraph_type_erased_device_array_view_t* vertices_view = nullptr;

        p_handle = hipgraph_create_resource_handle(nullptr);
        ASSERT_NE(p_handle, nullptr) << "resource handle creation failed.";

        create_test_graph(p_handle,
                          h_src,
                          h_dst,
                          h_wgt,
                          num_edges,
                          store_transposed,
                          HIPGRAPH_FALSE,
                          HIPGRAPH_TRUE,
                          &graph,
                          &ret_error);

        switch(test_type)
        {
        case JACCARD:
            ret_code = hipgraph_all_pairs_jaccard_coefficients(p_handle,
                                                               graph,
                                                               vertices_view,
                                                               use_weight,
                                                               topk,
                                                               HIPGRAPH_FALSE,
                                                               &result,
                                                               &ret_error);
            break;
        case SORENSEN:
            ret_code = hipgraph_all_pairs_sorensen_coefficients(p_handle,
                                                                graph,
                                                                vertices_view,
                                                                use_weight,
                                                                topk,
                                                                HIPGRAPH_FALSE,
                                                                &result,
                                                                &ret_error);
            break;
        case OVERLAP:
            ret_code = hipgraph_all_pairs_overlap_coefficients(p_handle,
                                                               graph,
                                                               vertices_view,
                                                               use_weight,
                                                               topk,
                                                               HIPGRAPH_FALSE,
                                                               &result,
                                                               &ret_error);
            break;
        }

        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "hipgraph similarity failed: " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* similarity_coefficient;

        hipgraph_vertex_pairs_t* vertex_pairs;
        vertex_pairs           = hipgraph_similarity_result_get_vertex_pairs(result);
        similarity_coefficient = hipgraph_similarity_result_get_similarity(result);

        hipgraph_type_erased_device_array_view_t* result_v1;
        hipgraph_type_erased_device_array_view_t* result_v2;

        result_v1               = hipgraph_vertex_pairs_get_first(vertex_pairs);
        result_v2               = hipgraph_vertex_pairs_get_second(vertex_pairs);
        size_t result_num_pairs = hipgraph_type_erased_device_array_view_size(result_v1);

        EXPECT_EQ(result_num_pairs, num_pairs) << "Incorrect number of results";

        vertex_t h_result_v1[result_num_pairs];
        vertex_t h_result_v2[result_num_pairs];
        weight_t h_similarity_coefficient[result_num_pairs];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_result_v1, result_v1, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_result_v2, result_v2, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle,
            (hipgraph_byte_t*)h_similarity_coefficient,
            similarity_coefficient,
            &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        weight_t result_matrix[num_vertices][num_vertices];
        for(size_t i = 0; i < num_vertices; ++i)
            for(size_t j = 0; j < num_vertices; ++j)
                result_matrix[i][j] = 0;

        for(size_t i = 0; i < num_pairs; ++i)
            result_matrix[h_result_v1[i]][h_result_v2[i]] = h_similarity_coefficient[i];

        for(size_t i = 0; i < num_pairs; ++i)
        {
            EXPECT_NEAR(result_matrix[h_first[i]][h_second[i]], h_result[i], 0.001)
                << "similarity results don't match at position " << i;
        }

        if(result != nullptr)
            hipgraph_similarity_result_free(result);
        hipgraph_sg_graph_free(graph);
        hipgraph_free_resource_handle(p_handle);
        hipgraph_error_free(ret_error);
    }

    TEST(AlgorithmTest, Jaccard)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 6;
        size_t num_pairs    = 10;

        vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
        weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t h_first[]  = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3};
        vertex_t h_second[] = {1, 3, 4, 2, 3, 5, 3, 4, 5, 4};
        weight_t h_result[]
            = {0.2, 0.666667, 0.333333, 0.4, 0.166667, 0.5, 0.2, 0.25, 0.25, 0.666667};

        generic_similarity_test(h_src,
                                h_dst,
                                h_wgt,
                                h_first,
                                h_second,
                                h_result,
                                num_vertices,
                                num_edges,
                                num_pairs,
                                HIPGRAPH_FALSE,
                                HIPGRAPH_FALSE,
                                JACCARD);
    }

    TEST(AlgorithmTest, WeightedJaccard)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 7;
        size_t num_pairs    = 3;

        vertex_t h_src[] = {0, 1, 2, 0, 1, 2, 3, 3, 3, 4, 4, 4, 0, 5, 2, 6};
        vertex_t h_dst[] = {3, 3, 3, 4, 4, 4, 0, 1, 2, 0, 1, 2, 5, 0, 6, 2};
        weight_t h_wgt[]
            = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5, 4.0, 4.0};

        vertex_t h_first[]  = {0, 0, 1};
        vertex_t h_second[] = {1, 2, 3};
        weight_t h_result[] = {0.357143, 0.208333, 0.0};

        generic_similarity_test(h_src,
                                h_dst,
                                h_wgt,
                                h_first,
                                h_second,
                                h_result,
                                num_vertices,
                                num_edges,
                                num_pairs,
                                HIPGRAPH_FALSE,
                                HIPGRAPH_TRUE,
                                JACCARD);
    }

    TEST(AlgorithmTest, Sorensen)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 6;
        size_t num_pairs    = 10;

        vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
        weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t h_first[]  = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3};
        vertex_t h_second[] = {1, 3, 4, 2, 3, 5, 3, 4, 5, 4};
        weight_t h_result[]
            = {0.333333, 0.8, 0.5, 0.571429, 0.285714, 0.666667, 0.333333, 0.4, 0.4, 0.8};

        generic_similarity_test(h_src,
                                h_dst,
                                h_wgt,
                                h_first,
                                h_second,
                                h_result,
                                num_vertices,
                                num_edges,
                                num_pairs,
                                HIPGRAPH_FALSE,
                                HIPGRAPH_FALSE,
                                SORENSEN);
    }

    TEST(AlgorithmTest, WeightedSorensen)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 7;
        size_t num_pairs    = 3;

        vertex_t h_src[] = {0, 1, 2, 0, 1, 2, 3, 3, 3, 4, 4, 4, 0, 5, 2, 6};
        vertex_t h_dst[] = {3, 3, 3, 4, 4, 4, 0, 1, 2, 0, 1, 2, 5, 0, 6, 2};
        weight_t h_wgt[]
            = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5, 4.0, 4.0};

        vertex_t h_first[]  = {0, 0, 1};
        vertex_t h_second[] = {1, 2, 3};
        weight_t h_result[] = {0.526316, 0.344828, 0.000000};

        generic_similarity_test(h_src,
                                h_dst,
                                h_wgt,
                                h_first,
                                h_second,
                                h_result,
                                num_vertices,
                                num_edges,
                                num_pairs,
                                HIPGRAPH_FALSE,
                                HIPGRAPH_TRUE,
                                SORENSEN);
    }

    TEST(AlgorithmTest, Overlap)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 6;
        size_t num_pairs    = 10;

        vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
        weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t h_first[]  = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3};
        vertex_t h_second[] = {1, 3, 4, 2, 3, 5, 3, 4, 5, 4};
        weight_t h_result[] = {0.5, 1, 0.5, 0.666667, 0.333333, 1, 0.333333, 0.5, 0.5, 1};

        generic_similarity_test(h_src,
                                h_dst,
                                h_wgt,
                                h_first,
                                h_second,
                                h_result,
                                num_vertices,
                                num_edges,
                                num_pairs,
                                HIPGRAPH_FALSE,
                                HIPGRAPH_FALSE,
                                OVERLAP);
    }

    TEST(AlgorithmTest, WeightedOverlap)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 7;
        size_t num_pairs    = 3;

        vertex_t h_src[] = {0, 1, 2, 0, 1, 2, 3, 3, 3, 4, 4, 4, 0, 5, 2, 6};
        vertex_t h_dst[] = {3, 3, 3, 4, 4, 4, 0, 1, 2, 0, 1, 2, 5, 0, 6, 2};
        weight_t h_wgt[]
            = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5, 4.0, 4.0};

        vertex_t h_first[]  = {0, 0, 1};
        vertex_t h_second[] = {1, 2, 3};
        weight_t h_result[] = {0.714286, 0.416667, 0.000000};

        generic_similarity_test(h_src,
                                h_dst,
                                h_wgt,
                                h_first,
                                h_second,
                                h_result,
                                num_vertices,
                                num_edges,
                                num_pairs,
                                HIPGRAPH_FALSE,
                                HIPGRAPH_TRUE,
                                OVERLAP);
    }

    TEST(AlgorithmTest, AllPairsJaccard)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 6;
        size_t num_pairs    = 22;

        vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
        weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t h_first[]  = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5};
        vertex_t h_second[] = {1, 2, 3, 4, 0, 2, 3, 5, 0, 1, 3, 4, 5, 0, 1, 2, 4, 0, 2, 3, 1, 2};
        weight_t h_result[]
            = {0.2,  0.25, 0.666667, 0.333333, 0.2, 0.4,      0.166667, 0.5,  0.25,     0.4, 0.2,
               0.25, 0.25, 0.666667, 0.166667, 0.2, 0.666667, 0.333333, 0.25, 0.666667, 0.5, 0.25};

        generic_all_pairs_similarity_test(h_src,
                                          h_dst,
                                          h_wgt,
                                          h_first,
                                          h_second,
                                          h_result,
                                          num_vertices,
                                          num_edges,
                                          num_pairs,
                                          HIPGRAPH_FALSE,
                                          HIPGRAPH_FALSE,
                                          SIZE_MAX,
                                          JACCARD);
    }

    TEST(AlgorithmTest, WeightedAllPairsJaccard)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 7;
        size_t num_pairs    = 16;

        vertex_t h_src[] = {0, 1, 2, 0, 1, 2, 3, 3, 3, 4, 4, 4, 0, 5, 2, 6};
        vertex_t h_dst[] = {3, 3, 3, 4, 4, 4, 0, 1, 2, 0, 1, 2, 5, 0, 6, 2};
        weight_t h_wgt[]
            = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5, 4.0, 4.0};

        vertex_t h_first[]  = {0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6};
        vertex_t h_second[] = {1, 2, 0, 2, 0, 1, 4, 5, 6, 3, 5, 6, 3, 4, 3, 4};
        weight_t h_result[] = {0.357143,
                               0.208333,
                               0.357143,
                               0.411765,
                               0.208333,
                               0.411765,
                               0.4,
                               0.0833333,
                               0.272727,
                               0.4,
                               0.222222,
                               0.352941,
                               0.0833333,
                               0.222222,
                               0.272727,
                               0.352941};

        generic_all_pairs_similarity_test(h_src,
                                          h_dst,
                                          h_wgt,
                                          h_first,
                                          h_second,
                                          h_result,
                                          num_vertices,
                                          num_edges,
                                          num_pairs,
                                          HIPGRAPH_FALSE,
                                          HIPGRAPH_TRUE,
                                          SIZE_MAX,
                                          JACCARD);
    }

    TEST(AlgorithmTest, AllPairsSorensen)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 6;
        size_t num_pairs    = 22;

        vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
        weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t h_first[]  = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5};
        vertex_t h_second[] = {1, 2, 3, 4, 0, 2, 3, 5, 0, 1, 3, 4, 5, 0, 1, 2, 4, 0, 2, 3, 1, 2};
        weight_t h_result[]
            = {0.333333, 0.4,      0.8,      0.5, 0.333333, 0.571429, 0.285714, 0.666667,
               0.4,      0.571429, 0.333333, 0.4, 0.4,      0.8,      0.285714, 0.333333,
               0.8,      0.5,      0.4,      0.8, 0.666667, 0.4};

        generic_all_pairs_similarity_test(h_src,
                                          h_dst,
                                          h_wgt,
                                          h_first,
                                          h_second,
                                          h_result,
                                          num_vertices,
                                          num_edges,
                                          num_pairs,
                                          HIPGRAPH_FALSE,
                                          HIPGRAPH_FALSE,
                                          SIZE_MAX,
                                          SORENSEN);
    }

    TEST(AlgorithmTest, WeightedAllPairsSorensen)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 7;
        size_t num_pairs    = 16;

        vertex_t h_src[] = {0, 1, 2, 0, 1, 2, 3, 3, 3, 4, 4, 4, 0, 5, 2, 6};
        vertex_t h_dst[] = {3, 3, 3, 4, 4, 4, 0, 1, 2, 0, 1, 2, 5, 0, 6, 2};
        weight_t h_wgt[]
            = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5, 4.0, 4.0};

        vertex_t h_first[]  = {0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6};
        vertex_t h_second[] = {1, 2, 0, 2, 0, 1, 4, 5, 6, 3, 5, 6, 3, 4, 3, 4};
        weight_t h_result[] = {0.526316,
                               0.344828,
                               0.526316,
                               0.583333,
                               0.344828,
                               0.583333,
                               0.571429,
                               0.153846,
                               0.428571,
                               0.571429,
                               0.363636,
                               0.521739,
                               0.153846,
                               0.363636,
                               0.428571,
                               0.521739};

        generic_all_pairs_similarity_test(h_src,
                                          h_dst,
                                          h_wgt,
                                          h_first,
                                          h_second,
                                          h_result,
                                          num_vertices,
                                          num_edges,
                                          num_pairs,
                                          HIPGRAPH_FALSE,
                                          HIPGRAPH_TRUE,
                                          SIZE_MAX,
                                          SORENSEN);
    }

    TEST(AlgorithmTest, AllPairsOverlap)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 6;
        size_t num_pairs    = 22;

        vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
        weight_t h_wgt[] = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};

        vertex_t h_first[]  = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5};
        vertex_t h_second[] = {1, 2, 3, 4, 0, 2, 3, 5, 0, 1, 3, 4, 5, 0, 1, 2, 4, 0, 2, 3, 1, 2};
        weight_t h_result[]
            = {0.5, 0.5, 1, 0.5,      0.5,      0.666667, 0.333333, 1,   0.5, 0.666667, 0.333333,
               0.5, 0.5, 1, 0.333333, 0.333333, 1,        0.5,      0.5, 1,   1,        0.5};

        generic_all_pairs_similarity_test(h_src,
                                          h_dst,
                                          h_wgt,
                                          h_first,
                                          h_second,
                                          h_result,
                                          num_vertices,
                                          num_edges,
                                          num_pairs,
                                          HIPGRAPH_FALSE,
                                          HIPGRAPH_FALSE,
                                          SIZE_MAX,
                                          OVERLAP);
    }

    TEST(AlgorithmTest, WeightedAllPairsOverlap)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 7;
        size_t num_pairs    = 16;

        vertex_t h_src[] = {0, 1, 2, 0, 1, 2, 3, 3, 3, 4, 4, 4, 0, 5, 2, 6};
        vertex_t h_dst[] = {3, 3, 3, 4, 4, 4, 0, 1, 2, 0, 1, 2, 5, 0, 6, 2};
        weight_t h_wgt[]
            = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5, 4.0, 4.0};

        vertex_t h_first[]  = {0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6};
        vertex_t h_second[] = {1, 2, 0, 2, 0, 1, 4, 5, 6, 3, 5, 6, 3, 4, 3, 4};
        weight_t h_result[] = {0.714286,
                               0.416667,
                               0.714286,
                               1,
                               0.416667,
                               1,
                               1,
                               0.166667,
                               0.5,
                               1,
                               0.571429,
                               0.75,
                               0.166667,
                               0.571429,
                               0.5,
                               0.75};

        generic_all_pairs_similarity_test(h_src,
                                          h_dst,
                                          h_wgt,
                                          h_first,
                                          h_second,
                                          h_result,
                                          num_vertices,
                                          num_edges,
                                          num_pairs,
                                          HIPGRAPH_FALSE,
                                          HIPGRAPH_TRUE,
                                          SIZE_MAX,
                                          OVERLAP);
    }

    TEST(AlgorithmTest, AllPairsJaccardTopk)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 6;
        size_t topk         = 6;
        size_t num_pairs    = 6;

        vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
        weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t h_first[]  = {0, 1, 3, 3, 4, 5};
        vertex_t h_second[] = {3, 5, 0, 4, 3, 1};
        weight_t h_result[] = {0.666667, 0.5, 0.666667, 0.666667, 0.666667, 0.5};

        generic_all_pairs_similarity_test(h_src,
                                          h_dst,
                                          h_wgt,
                                          h_first,
                                          h_second,
                                          h_result,
                                          num_vertices,
                                          num_edges,
                                          num_pairs,
                                          HIPGRAPH_FALSE,
                                          HIPGRAPH_FALSE,
                                          topk,
                                          JACCARD);
    }

    TEST(AlgorithmTest, WeightedAllPairsJaccardTopk)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 7;
        size_t num_pairs    = 6;
        size_t topk         = 6;

        vertex_t h_src[] = {0, 1, 2, 0, 1, 2, 3, 3, 3, 4, 4, 4, 0, 5, 2, 6};
        vertex_t h_dst[] = {3, 3, 3, 4, 4, 4, 0, 1, 2, 0, 1, 2, 5, 0, 6, 2};
        weight_t h_wgt[]
            = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5, 4.0, 4.0};

        vertex_t h_first[]  = {0, 1, 1, 2, 3, 4};
        vertex_t h_second[] = {1, 0, 2, 1, 4, 3};
        weight_t h_result[] = {0.357143, 0.357143, 0.411765, 0.411765, 0.4, 0.4};

        generic_all_pairs_similarity_test(h_src,
                                          h_dst,
                                          h_wgt,
                                          h_first,
                                          h_second,
                                          h_result,
                                          num_vertices,
                                          num_edges,
                                          num_pairs,
                                          HIPGRAPH_FALSE,
                                          HIPGRAPH_TRUE,
                                          topk,
                                          JACCARD);
    }

    TEST(AlgorithmTest, AllPairsSorensenTopk)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 6;
        size_t num_pairs    = 6;
        size_t topk         = 6;

        vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
        weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t h_first[]  = {0, 1, 3, 3, 4, 5};
        vertex_t h_second[] = {3, 5, 0, 4, 3, 1};
        weight_t h_result[] = {0.8, 0.666667, 0.8, 0.8, 0.8, 0.666667};

        generic_all_pairs_similarity_test(h_src,
                                          h_dst,
                                          h_wgt,
                                          h_first,
                                          h_second,
                                          h_result,
                                          num_vertices,
                                          num_edges,
                                          num_pairs,
                                          HIPGRAPH_FALSE,
                                          HIPGRAPH_FALSE,
                                          topk,
                                          SORENSEN);
    }

    TEST(AlgorithmTest, WeightedAllPairsSorensenTopk)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 7;
        size_t num_pairs    = 6;
        size_t topk         = 6;

        vertex_t h_src[] = {0, 1, 2, 0, 1, 2, 3, 3, 3, 4, 4, 4, 0, 5, 2, 6};
        vertex_t h_dst[] = {3, 3, 3, 4, 4, 4, 0, 1, 2, 0, 1, 2, 5, 0, 6, 2};
        weight_t h_wgt[]
            = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5, 4.0, 4.0};

        vertex_t h_first[]  = {0, 1, 1, 2, 3, 4};
        vertex_t h_second[] = {1, 0, 2, 1, 4, 3};
        weight_t h_result[] = {0.526316, 0.526316, 0.583333, 0.583333, 0.571429, 0.571429};

        generic_all_pairs_similarity_test(h_src,
                                          h_dst,
                                          h_wgt,
                                          h_first,
                                          h_second,
                                          h_result,
                                          num_vertices,
                                          num_edges,
                                          num_pairs,
                                          HIPGRAPH_FALSE,
                                          HIPGRAPH_TRUE,
                                          topk,
                                          SORENSEN);
    }

    TEST(AlgorithmTest, AllPairsOverlapTopk)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 6;
        size_t num_pairs    = 6;
        size_t topk         = 6;

        vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
        weight_t h_wgt[] = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};

        vertex_t h_first[]  = {0, 1, 3, 3, 4, 5};
        vertex_t h_second[] = {3, 5, 0, 4, 3, 1};
        weight_t h_result[] = {1, 1, 1, 1, 1, 1};

        generic_all_pairs_similarity_test(h_src,
                                          h_dst,
                                          h_wgt,
                                          h_first,
                                          h_second,
                                          h_result,
                                          num_vertices,
                                          num_edges,
                                          num_pairs,
                                          HIPGRAPH_FALSE,
                                          HIPGRAPH_FALSE,
                                          topk,
                                          OVERLAP);
    }

    TEST(AlgorithmTest, WeightedAllPairsOverlapTopk)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 7;
        size_t num_pairs    = 6;
        size_t topk         = 6;

        vertex_t h_src[] = {0, 1, 2, 0, 1, 2, 3, 3, 3, 4, 4, 4, 0, 5, 2, 6};
        vertex_t h_dst[] = {3, 3, 3, 4, 4, 4, 0, 1, 2, 0, 1, 2, 5, 0, 6, 2};
        weight_t h_wgt[]
            = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5, 4.0, 4.0};

        vertex_t h_first[]  = {1, 2, 3, 4, 4, 6};
        vertex_t h_second[] = {2, 1, 4, 3, 6, 4};
        weight_t h_result[] = {1, 1, 1, 1, 0.75, 0.75};

        generic_all_pairs_similarity_test(h_src,
                                          h_dst,
                                          h_wgt,
                                          h_first,
                                          h_second,
                                          h_result,
                                          num_vertices,
                                          num_edges,
                                          num_pairs,
                                          HIPGRAPH_FALSE,
                                          HIPGRAPH_TRUE,
                                          topk,
                                          OVERLAP);
    }

} // namespace
