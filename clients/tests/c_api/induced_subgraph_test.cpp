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

#include "hipgraph_c/graph.h"
#include "hipgraph_c/graph_functions.h"

#include <cstdio>

using vertex_t = int32_t;
using edge_t   = int32_t;
using weight_t = float;

/*
 * Simple check of creating a graph from a COO on device memory.
 */
namespace
{
    using namespace hipGRAPH::testing;
    void generic_induced_subgraph_test(vertex_t*       h_src,
                                       vertex_t*       h_dst,
                                       weight_t*       h_wgt,
                                       size_t          num_vertices,
                                       size_t          num_edges,
                                       hipgraph_bool_t store_transposed,
                                       size_t*         h_subgraph_offsets,
                                       vertex_t*       h_subgraph_vertices,
                                       size_t          num_subgraph_offsets,
                                       vertex_t*       h_result_src,
                                       vertex_t*       h_result_dst,
                                       weight_t*       h_result_wgt,
                                       size_t*         h_result_offsets,
                                       size_t          num_results)
    {

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_resource_handle_t*               p_handle               = nullptr;
        hipgraph_graph_t*                         graph                  = nullptr;
        hipgraph_type_erased_device_array_t*      subgraph_offsets       = nullptr;
        hipgraph_type_erased_device_array_t*      subgraph_vertices      = nullptr;
        hipgraph_type_erased_device_array_view_t* subgraph_offsets_view  = nullptr;
        hipgraph_type_erased_device_array_view_t* subgraph_vertices_view = nullptr;

        hipgraph_induced_subgraph_result_t* result = nullptr;

        hipgraph_data_type_id_t vertex_tid = HIPGRAPH_INT32;
        hipgraph_data_type_id_t size_t_tid = HIPGRAPH_SIZE_T;

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
                          &graph,
                          &ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_subgraph_offsets, size_t_tid, &subgraph_offsets, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "subgraph_offsets create failed: " << hipgraph_error_message(ret_error);

        ret_code
            = hipgraph_type_erased_device_array_create(p_handle,
                                                       h_subgraph_offsets[num_subgraph_offsets - 1],
                                                       vertex_tid,
                                                       &subgraph_vertices,
                                                       &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "subgraph_offsets create failed: " << hipgraph_error_message(ret_error);

        subgraph_offsets_view  = hipgraph_type_erased_device_array_view(subgraph_offsets);
        subgraph_vertices_view = hipgraph_type_erased_device_array_view(subgraph_vertices);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, subgraph_offsets_view, (hipgraph_byte_t*)h_subgraph_offsets, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "subgraph_offsets copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, subgraph_vertices_view, (hipgraph_byte_t*)h_subgraph_vertices, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "subgraph_vertices copy_from_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_extract_induced_subgraph(p_handle,
                                                     graph,
                                                     subgraph_offsets_view,
                                                     subgraph_vertices_view,
                                                     HIPGRAPH_FALSE,
                                                     &result,
                                                     &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "hipgraph_extract_induced_subgraph failed: " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* extracted_src;
        hipgraph_type_erased_device_array_view_t* extracted_dst;
        hipgraph_type_erased_device_array_view_t* extracted_wgt;
        hipgraph_type_erased_device_array_view_t* extracted_graph_offsets;

        extracted_src           = hipgraph_induced_subgraph_get_sources(result);
        extracted_dst           = hipgraph_induced_subgraph_get_destinations(result);
        extracted_wgt           = hipgraph_induced_subgraph_get_edge_weights(result);
        extracted_graph_offsets = hipgraph_induced_subgraph_get_subgraph_offsets(result);

        size_t extracted_size = hipgraph_type_erased_device_array_view_size(extracted_src);

        vertex_t h_extracted_src[extracted_size];
        vertex_t h_extracted_dst[extracted_size];
        weight_t h_extracted_wgt[extracted_size];
        size_t   h_extracted_graph_offsets[num_subgraph_offsets];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_extracted_src, extracted_src, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_extracted_dst, extracted_dst, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_extracted_wgt, extracted_wgt, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle,
            (hipgraph_byte_t*)h_extracted_graph_offsets,
            extracted_graph_offsets,
            &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        EXPECT_EQ(extracted_size, num_results) << "results not the same size";

        for(size_t i = 0; i < num_subgraph_offsets; ++i)
        {
            EXPECT_EQ(h_extracted_graph_offsets[i], h_result_offsets[i])
                << "graph offsets should match at position " << i;
        }

        for(size_t i = 0; i < num_results; ++i)
        {
            hipgraph_bool_t found = HIPGRAPH_FALSE;
            for(size_t j = 0; (j < num_results) && !found; ++j)
            {
                if((h_extracted_src[i] == h_result_src[j])
                   && (h_extracted_dst[i] == h_result_dst[j])
                   && (nearlyEqual(h_extracted_wgt[i], h_result_wgt[j], 0.001)))
                    found = HIPGRAPH_TRUE;
            }
            EXPECT_TRUE(found) << "extracted an edge that doesn't match at position " << i;
        }
    }

    TEST(PlumbingTest, InducedSubgraph)
    {
        size_t num_edges            = 8;
        size_t num_vertices         = 6;
        size_t num_subgraph_offsets = 2;
        size_t num_results          = 5;

        vertex_t h_src[]               = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]               = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]               = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        size_t   h_subgraph_offsets[]  = {0, 4};
        vertex_t h_subgraph_vertices[] = {0, 1, 2, 3};
        vertex_t h_result_src[]        = {0, 1, 2, 2, 2};
        vertex_t h_result_dst[]        = {1, 3, 0, 1, 3};
        weight_t h_result_wgt[]        = {0.1f, 2.1f, 5.1f, 3.1f, 4.1f};
        size_t   h_result_offsets[]    = {0, 5};

        generic_induced_subgraph_test(h_src,
                                      h_dst,
                                      h_wgt,
                                      num_vertices,
                                      num_edges,
                                      HIPGRAPH_FALSE,
                                      h_subgraph_offsets,
                                      h_subgraph_vertices,
                                      num_subgraph_offsets,
                                      h_result_src,
                                      h_result_dst,
                                      h_result_wgt,
                                      h_result_offsets,
                                      num_results);
    }

} // namespace
