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

#include <array>

#include <cinttypes>
#include <cmath>
#include <cstdio>

#include "hipgraph_c/algorithms.h"
#include "hipgraph_c/error.h"
#include "hipgraph_c/graph.h"

#include "test_utils.h"

namespace
{

    using namespace hipGRAPH::testing;

    using vertex_t = int32_t;
    using edge_t   = int32_t;
    using weight_t = float;

    void generic_scc_test(vertex_t*       h_src,
                          vertex_t*       h_dst,
                          weight_t*       h_wgt,
                          vertex_t*       h_result,
                          size_t          num_vertices,
                          size_t          num_edges,
                          hipgraph_bool_t store_transposed)
    {

        hipgraph_error_code_t ret_code  = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error = nullptr;

        hipgraph_resource_handle_t* p_handle = nullptr;
        hipgraph_graph_t*           p_graph  = nullptr;
        hipgraph_labeling_result_t* p_result = nullptr;

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
                          &p_graph,
                          &ret_error);

        ret_code = hipgraph_strongly_connected_components(
            p_handle, p_graph, HIPGRAPH_FALSE, &p_result, &ret_error);

        // FIXME: Actual implementation will be something like this
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS) << "hipgraph_strongly_connected_components failed: "
                                              << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* vertices;
        hipgraph_type_erased_device_array_view_t* components;

        vertices   = hipgraph_labeling_result_get_vertices(p_result);
        components = hipgraph_labeling_result_get_labels(p_result);

        vertex_t h_vertices[num_vertices];
        vertex_t h_components[num_vertices];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_vertices, vertices, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_components, components, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ASSERT_EQ((num_vertices > std::numeric_limits<vertex_t>::max())
                      ? HIPGRAPH_UNSUPPORTED_TYPE_COMBINATION
                      : HIPGRAPH_SUCCESS,
                  HIPGRAPH_SUCCESS)
            << "The number of vertices exceeds range of vertex_t" << std::endl;

        vertex_t component_check[num_vertices];
        for(size_t i = 0; i < num_vertices; ++i)
        {
            component_check[i] = vertex_t(num_vertices);
        }

        for(size_t i = 0; i < num_vertices; ++i)
        {
            if(component_check[h_result[i]] == vertex_t(num_vertices))
                component_check[h_result[i]] = h_components[i];
        }

        for(size_t i = 0; i < num_vertices; ++i)
        {
            EXPECT_EQ(h_components[i], component_check[h_result[i]])
                << "component results don't match at position " << i;
        }

        hipgraph_type_erased_device_array_view_free(components);
        hipgraph_type_erased_device_array_view_free(vertices);
        hipgraph_labeling_result_free(p_result);

        hipgraph_sg_graph_free(p_graph);
        hipgraph_free_resource_handle(p_handle);
        hipgraph_error_free(ret_error);
    }

    TEST(AlgorithmTest, StronglyConnectedComponents)
    {
        size_t num_edges    = 19;
        size_t num_vertices = 12;

        vertex_t h_src[] = {0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 6, 7, 7, 8, 8, 8, 9, 10};
        vertex_t h_dst[] = {1, 2, 3, 4, 0, 1, 3, 4, 5, 3, 5, 7, 9, 10, 6, 7, 9, 11, 11};
        weight_t h_wgt[] = {1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0};

        vertex_t h_result[] = {0, 0, 0, 3, 3, 5, 6, 7, 8, 9, 10, 11};

        // SCC wants store_transposed = HIPGRAPH_FALSE
        generic_scc_test(h_src, h_dst, h_wgt, h_result, num_vertices, num_edges, HIPGRAPH_FALSE);
    }

} // namespace
