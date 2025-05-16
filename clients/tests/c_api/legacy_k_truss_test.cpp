// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0
/*
 * Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

/*
 * Copyright (C) 2023-2024, NVIDIA CORPORATION.
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
    void generic_k_truss_test(vertex_t*       h_src,
                              vertex_t*       h_dst,
                              weight_t*       h_wgt,
                              vertex_t*       h_expected_src,
                              vertex_t*       h_expected_dst,
                              weight_t*       h_expected_wgt,
                              size_t*         h_expected_offsets,
                              size_t          num_vertices,
                              size_t          num_edges,
                              size_t          k,
                              size_t          num_expected_offsets,
                              size_t          num_expected_edges,
                              hipgraph_bool_t store_transposed)
    {

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_data_type_id_t vertex_tid    = HIPGRAPH_INT32;
        hipgraph_data_type_id_t edge_tid      = HIPGRAPH_INT32;
        hipgraph_data_type_id_t weight_tid    = HIPGRAPH_FLOAT32;
        hipgraph_data_type_id_t edge_id_tid   = HIPGRAPH_INT32;
        hipgraph_data_type_id_t edge_type_tid = HIPGRAPH_INT32;

        hipgraph_resource_handle_t*               p_handle   = nullptr;
        hipgraph_graph_t*                         graph      = nullptr;
        hipgraph_type_erased_device_array_t*      seeds      = nullptr;
        hipgraph_type_erased_device_array_view_t* seeds_view = nullptr;
        hipgraph_induced_subgraph_result_t*       result     = nullptr;

        p_handle = hipgraph_create_resource_handle(nullptr);
        ASSERT_NE(p_handle, nullptr) << "resource handle creation failed.";

        create_sg_test_graph(p_handle,
                             vertex_tid,
                             edge_tid,
                             h_src,
                             h_dst,
                             weight_tid,
                             h_wgt,
                             edge_type_tid,
                             nullptr,
                             edge_id_tid,
                             nullptr,
                             num_edges,
                             store_transposed,
                             HIPGRAPH_FALSE,
                             HIPGRAPH_TRUE,
                             HIPGRAPH_FALSE,
                             &graph,
                             &ret_error);

        ret_code
            = hipgraph_k_truss_subgraph(p_handle, graph, k, HIPGRAPH_FALSE, &result, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "hipgraph_k_truss_subgraph failed: " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* src;
        hipgraph_type_erased_device_array_view_t* dst;
        hipgraph_type_erased_device_array_view_t* wgt;
        hipgraph_type_erased_device_array_view_t* offsets;

        src     = hipgraph_induced_subgraph_get_sources(result);
        dst     = hipgraph_induced_subgraph_get_destinations(result);
        wgt     = hipgraph_induced_subgraph_get_edge_weights(result);
        offsets = hipgraph_induced_subgraph_get_subgraph_offsets(result);

        size_t num_result_edges   = hipgraph_type_erased_device_array_view_size(src);
        size_t num_result_offsets = hipgraph_type_erased_device_array_view_size(offsets);

        vertex_t h_result_src[num_result_edges];
        vertex_t h_result_dst[num_result_edges];
        weight_t h_result_wgt[num_result_edges];
        size_t   h_result_offsets[num_result_offsets];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_result_src, src, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_result_dst, dst, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        if(wgt != nullptr)
        {
            ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (hipgraph_byte_t*)h_result_wgt, wgt, &ret_error);
            ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
                << "copy_to_host failed: " << hipgraph_error_message(ret_error);
        }

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_result_offsets, offsets, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        EXPECT_EQ(num_result_edges, num_expected_edges) << "results not the same size";

        for(size_t i = 0; i < num_expected_offsets; ++i)
        {
            EXPECT_EQ(h_expected_offsets[i], h_result_offsets[i])
                << "graph offsets should match at position " << i;
        }

        for(size_t i = 0; i < num_expected_edges; ++i)
        {
            hipgraph_bool_t found = HIPGRAPH_FALSE;
            for(size_t j = 0; (j < num_expected_edges) && !found; ++j)
            {
                if((h_expected_src[i] == h_result_src[j]) && (h_expected_dst[i] == h_result_dst[j]))
                    if(wgt != nullptr)
                    {
                        found = (nearlyEqual(h_expected_wgt[i], h_result_wgt[j], 0.001));
                    }
                    else
                    {
                        found = HIPGRAPH_TRUE;
                    }
            }
            EXPECT_TRUE(found) << "extracted an edge that doesn't match at position " << i;
        }

        hipgraph_type_erased_device_array_view_free(src);
        hipgraph_type_erased_device_array_view_free(dst);
        hipgraph_type_erased_device_array_view_free(wgt);
        hipgraph_type_erased_device_array_view_free(offsets);
        hipgraph_induced_subgraph_result_free(result);

        hipgraph_sg_graph_free(graph);
        hipgraph_free_resource_handle(p_handle);
        hipgraph_error_free(ret_error);
    }

    TEST(BrokenTest, LegacyKTruss)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 6;
        size_t k            = 3;

        vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
        weight_t h_wgt[] = {0.1f,
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

        vertex_t h_result_src[]       = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
        vertex_t h_result_dst[]       = {1, 2, 0, 2, 3, 0, 1, 3, 1, 2};
        weight_t h_result_wgt[]       = {0.1, 5.1, 0.1, 3.1, 2.1, 5.1, 3.1, 4.1, 2.1, 4.1};
        size_t   h_result_offsets[]   = {0, 10};
        size_t   num_expected_edges   = 10;
        size_t   num_expected_offsets = 2;

        generic_k_truss_test(h_src,
                             h_dst,
                             h_wgt,
                             h_result_src,
                             h_result_dst,
                             h_result_wgt,
                             h_result_offsets,
                             num_vertices,
                             num_edges,
                             k,
                             num_expected_offsets,
                             num_expected_edges,
                             HIPGRAPH_FALSE);
    }

    TEST(BrokenTest, LegacyKTrussNoWeights)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 6;
        size_t k            = 3;

        vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};

        vertex_t h_result_src[]       = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
        vertex_t h_result_dst[]       = {1, 2, 0, 2, 3, 0, 1, 3, 1, 2};
        size_t   h_result_offsets[]   = {0, 10};
        size_t   num_expected_edges   = 10;
        size_t   num_expected_offsets = 2;

        generic_k_truss_test(h_src,
                             h_dst,
                             nullptr,
                             h_result_src,
                             h_result_dst,
                             nullptr,
                             h_result_offsets,
                             num_vertices,
                             num_edges,
                             k,
                             num_expected_offsets,
                             num_expected_edges,
                             HIPGRAPH_FALSE);
    }

} // namespace
