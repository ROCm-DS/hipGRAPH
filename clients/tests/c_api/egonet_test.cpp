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

#include <cmath>

using vertex_t = int32_t;
using edge_t   = int32_t;
using weight_t = float;

namespace
{
    using namespace hipGRAPH::testing;
    void generic_egonet_test(vertex_t*       h_src,
                             vertex_t*       h_dst,
                             weight_t*       h_wgt,
                             vertex_t*       h_seeds,
                             vertex_t*       h_expected_src,
                             vertex_t*       h_expected_dst,
                             size_t*         h_expected_offsets,
                             size_t          num_vertices,
                             size_t          num_edges,
                             size_t          num_seeds,
                             size_t          radius,
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
                             HIPGRAPH_FALSE,
                             HIPGRAPH_FALSE,
                             &graph,
                             &ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_seeds, HIPGRAPH_INT32, &seeds, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "seeds create failed: " << hipgraph_error_message(ret_error);

        seeds_view = hipgraph_type_erased_device_array_view(seeds);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, seeds_view, (hipgraph_byte_t*)h_seeds, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "src copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_extract_ego(
            p_handle, graph, seeds_view, radius, HIPGRAPH_FALSE, &result, &ret_error);

        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "hipgraph_egonet failed: " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* src;
        hipgraph_type_erased_device_array_view_t* dst;
        // hipgraph_type_erased_device_array_view_t* wgt;
        hipgraph_type_erased_device_array_view_t* offsets;

        src = hipgraph_induced_subgraph_get_sources(result);
        dst = hipgraph_induced_subgraph_get_destinations(result);
        // wgt     = hipgraph_induced_subgraph_get_edge_weights(result);
        offsets = hipgraph_induced_subgraph_get_subgraph_offsets(result);

        size_t num_result_edges   = hipgraph_type_erased_device_array_view_size(src);
        size_t num_result_offsets = hipgraph_type_erased_device_array_view_size(offsets);

        vertex_t h_result_src[num_result_edges];
        vertex_t h_result_dst[num_result_edges];
        // weight_t h_result_wgt[num_result_edges];
        size_t h_result_offsets[num_result_offsets];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_result_src, src, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_result_dst, dst, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        // ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
        //     resource_handle, (hipgraph_byte_t*)h_result_wgt, wgt, &ret_error);
        // ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_result_offsets, offsets, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed.: " << hipgraph_error_message(ret_error);

        EXPECT_EQ((num_seeds + 1), num_result_offsets) << "number of offsets doesn't match";

        for(size_t i = 0; i < num_result_offsets; ++i)
        {
            EXPECT_EQ(h_result_offsets[i], h_expected_offsets[i])
                << "offsets don't match at position " << i;
        }

        weight_t M[num_vertices][num_vertices];

        for(size_t i = 0; i < num_seeds; ++i)
        {
            for(size_t r = 0; r < num_vertices; ++r)
                for(size_t c = 0; c < num_vertices; ++c)
                    M[r][c] = 0;

            for(size_t e = h_expected_offsets[i]; e < h_expected_offsets[i + 1]; ++e)
                M[h_expected_src[e]][h_expected_dst[e]] = 1;

            for(size_t e = h_result_offsets[i]; e < h_result_offsets[i + 1]; ++e)
            {
                EXPECT_GT(M[h_result_src[e]][h_result_dst[e]], 0)
                    << "found different edges at position " << i;
            }
        }

        hipgraph_type_erased_device_array_view_free(src);
        hipgraph_type_erased_device_array_view_free(dst);
        // hipgraph_type_erased_device_array_view_free(wgt);
        hipgraph_type_erased_device_array_view_free(offsets);
        hipgraph_induced_subgraph_result_free(result);

        hipgraph_sg_graph_free(graph);
        hipgraph_free_resource_handle(p_handle);
        hipgraph_error_free(ret_error);
    }

    TEST(AlgorithmTest, Egonet)
    {
        size_t num_edges    = 9;
        size_t num_vertices = 6;
        size_t radius       = 2;
        size_t num_seeds    = 2;

        vertex_t h_src[]   = {0, 1, 1, 2, 2, 2, 3, 3, 4};
        vertex_t h_dst[]   = {1, 3, 4, 0, 1, 3, 4, 5, 5};
        weight_t h_wgt[]   = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 6.1f};
        vertex_t h_seeds[] = {0, 1};

        vertex_t h_result_src[]     = {0, 1, 1, 3, 1, 1, 3, 3, 4};
        vertex_t h_result_dst[]     = {1, 3, 4, 4, 3, 4, 4, 5, 5};
        size_t   h_result_offsets[] = {0, 4, 9};

        // Egonet wants store_transposed = HIPGRAPH_FALSE
        generic_egonet_test(h_src,
                            h_dst,
                            h_wgt,
                            h_seeds,
                            h_result_src,
                            h_result_dst,
                            h_result_offsets,
                            num_vertices,
                            num_edges,
                            num_seeds,
                            radius,
                            HIPGRAPH_FALSE);
    }

    TEST(AlgorithmTest, EgonetNoWeights)
    {
        size_t num_edges    = 9;
        size_t num_vertices = 6;
        size_t radius       = 2;
        size_t num_seeds    = 2;

        vertex_t h_src[]   = {0, 1, 1, 2, 2, 2, 3, 3, 4};
        vertex_t h_dst[]   = {1, 3, 4, 0, 1, 3, 4, 5, 5};
        vertex_t h_seeds[] = {0, 1};

        vertex_t h_result_src[]     = {0, 1, 1, 3, 1, 1, 3, 3, 4};
        vertex_t h_result_dst[]     = {1, 3, 4, 4, 3, 4, 4, 5, 5};
        size_t   h_result_offsets[] = {0, 4, 9};

        // Egonet wants store_transposed = HIPGRAPH_FALSE
        generic_egonet_test(h_src,
                            h_dst,
                            nullptr,
                            h_seeds,
                            h_result_src,
                            h_result_dst,
                            h_result_offsets,
                            num_vertices,
                            num_edges,
                            num_seeds,
                            radius,
                            HIPGRAPH_FALSE);
    }

} // namespace
