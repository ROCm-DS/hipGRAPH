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
    int generic_similarity_test(const hipgraph_resource_handle_t* p_handle,
                                vertex_t*                         h_src,
                                vertex_t*                         h_dst,
                                weight_t*                         h_wgt,
                                vertex_t*                         h_first,
                                vertex_t*                         h_second,
                                weight_t*                         h_result,
                                size_t                            num_vertices,
                                size_t                            num_edges,
                                size_t                            num_pairs,
                                hipgraph_bool_t                   store_transposed,
                                hipgraph_bool_t                   use_weight,
                                similarity_t                      test_type)
    {
        int                     test_ret_value = 0;
        hipgraph_data_type_id_t vertex_tid     = HIPGRAPH_INT32;

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_graph_t*                         graph        = nullptr;
        hipgraph_similarity_result_t*             result       = nullptr;
        hipgraph_vertex_pairs_t*                  vertex_pairs = nullptr;
        hipgraph_type_erased_device_array_t*      v1           = nullptr;
        hipgraph_type_erased_device_array_t*      v2           = nullptr;
        hipgraph_type_erased_device_array_view_t* v1_view      = nullptr;
        hipgraph_type_erased_device_array_view_t* v2_view      = nullptr;

        ret_code = create_test_graph(p_handle,
                                     h_src,
                                     h_dst,
                                     h_wgt,
                                     num_edges,
                                     store_transposed,
                                     HIPGRAPH_FALSE,
                                     HIPGRAPH_TRUE,
                                     &graph,
                                     &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "create_test_graph failed. " << hipgraph_error_message(ret_error);

        if(hipgraph_resource_handle_get_rank(p_handle) != 0)
        {
            num_pairs = 0;
        }

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_pairs, vertex_tid, &v1, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "v1 create failed.";

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_pairs, vertex_tid, &v2, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "v2 create failed.";

        v1_view = hipgraph_type_erased_device_array_view(v1);
        v2_view = hipgraph_type_erased_device_array_view(v2);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, v1_view, (hipgraph_byte_t*)h_first, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "h_first copy_from_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, v2_view, (hipgraph_byte_t*)h_second, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "h_second copy_from_host failed.";

        ret_code = hipgraph_create_vertex_pairs(
            p_handle, graph, v1_view, v2_view, &vertex_pairs, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "create vertex pairs failed.";

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

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "hipgraph similarity failed. " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* similarity_coefficient;

        similarity_coefficient = hipgraph_similarity_result_get_similarity(result);

        weight_t h_similarity_coefficient[num_pairs];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle,
            (hipgraph_byte_t*)h_similarity_coefficient,
            similarity_coefficient,
            &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        for(int i = 0; (i < num_pairs) && (test_ret_value == 0); ++i)
        {
            EXPECT_NEAR(h_similarity_coefficient[i], h_result[i], 0.001)
                << "similarity results don't match";
        }

        if(result != nullptr)
            hipgraph_similarity_result_free(result);
        if(vertex_pairs != nullptr)
            hipgraph_vertex_pairs_free(vertex_pairs);
        hipgraph_mg_graph_free(graph);
        hipgraph_error_free(ret_error);

        return test_ret_value;
    }

    int test_jaccard(const hipgraph_resource_handle_t* p_handle)
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

        return generic_similarity_test(p_handle,
                                       h_src,
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

    int test_weighted_jaccard(const hipgraph_resource_handle_t* p_handle)
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

        return generic_similarity_test(p_handle,
                                       h_src,
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

    int test_sorensen(const hipgraph_resource_handle_t* p_handle)
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

        return generic_similarity_test(p_handle,
                                       h_src,
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

    int test_weighted_sorensen(const hipgraph_resource_handle_t* p_handle)
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

        return generic_similarity_test(p_handle,
                                       h_src,
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

    int test_overlap(const hipgraph_resource_handle_t* p_handle)
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

        return generic_similarity_test(p_handle,
                                       h_src,
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

    int test_weighted_overlap(const hipgraph_resource_handle_t* p_handle)
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

        return generic_similarity_test(p_handle,
                                       h_src,
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

} // namespace
