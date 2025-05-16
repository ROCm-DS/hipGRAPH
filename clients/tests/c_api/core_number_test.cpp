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

namespace
{
    using namespace hipGRAPH::testing;
    void generic_core_number_test(vertex_t*       h_src,
                                  vertex_t*       h_dst,
                                  weight_t*       h_wgt,
                                  vertex_t*       h_result,
                                  size_t          num_vertices,
                                  size_t          num_edges,
                                  hipgraph_bool_t store_transposed)
    {
        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_resource_handle_t* p_handle = nullptr;
        hipgraph_graph_t*           p_graph  = nullptr;
        hipgraph_core_result_t*     p_result = nullptr;

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
                          &p_graph,
                          &ret_error);

        ret_code = hipgraph_core_number(p_handle,
                                        p_graph,
                                        HIPGRAPH_K_CORE_DEGREE_TYPE_IN,
                                        HIPGRAPH_FALSE,
                                        &p_result,
                                        &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "hipgraph_core_number failed: " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* vertices;
        hipgraph_type_erased_device_array_view_t* core_numbers;

        vertices     = hipgraph_core_result_get_vertices(p_result);
        core_numbers = hipgraph_core_result_get_core_numbers(p_result);

        vertex_t h_vertices[num_vertices];
        vertex_t h_core_numbers[num_vertices];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_vertices, vertices, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_core_numbers, core_numbers, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        for(size_t i = 0; i < num_vertices; ++i)
        {
            EXPECT_NEAR(h_result[h_vertices[i]], h_core_numbers[i], 0.001)
                << "core number results don't match at position " << i;
        }

        hipgraph_core_result_free(p_result);
        hipgraph_sg_graph_free(p_graph);
        hipgraph_free_resource_handle(p_handle);
        hipgraph_error_free(ret_error);
    }

    TEST(AlgorithmTest, CoreNumber)
    {
        size_t num_edges    = 22;
        size_t num_vertices = 7;

        vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5, 3, 1, 4, 5, 5, 6};
        vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 3, 1, 6, 5};
        weight_t h_wgt[]    = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        vertex_t h_result[] = {2, 3, 2, 3, 3, 3, 1};

        generic_core_number_test(
            h_src, h_dst, h_wgt, h_result, num_vertices, num_edges, HIPGRAPH_FALSE);
    }

} // namespace
