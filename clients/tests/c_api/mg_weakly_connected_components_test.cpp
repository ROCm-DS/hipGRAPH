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
    int generic_wcc_test(const hipgraph_resource_handle_t* p_handle,
                         vertex_t*                         h_src,
                         vertex_t*                         h_dst,
                         weight_t*                         h_wgt,
                         vertex_t*                         h_result,
                         size_t                            num_vertices,
                         size_t                            num_edges,
                         hipgraph_bool_t                   store_transposed)
    {
        int test_ret_value = 0;

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_graph_t*           p_graph  = nullptr;
        hipgraph_labeling_result_t* p_result = nullptr;

        ret_code = create_mg_test_graph(p_handle,
                                        h_src,
                                        h_dst,
                                        h_wgt,
                                        num_edges,
                                        store_transposed,
                                        HIPGRAPH_TRUE,
                                        &p_graph,
                                        &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "create_mg_test_graph failed.";

        ret_code = hipgraph_weakly_connected_components(
            p_handle, p_graph, HIPGRAPH_FALSE, &p_result, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "hipgraph_weakly_connected_components failed.";

        // NOTE: Because we get back vertex ids and components, we can simply compare
        //       the returned values with the expected results for the entire
        //       graph.  Each GPU will have a subset of the total vertices, so
        //       they will do a subset of the comparisons.
        hipgraph_type_erased_device_array_view_t* vertices;
        hipgraph_type_erased_device_array_view_t* components;

        vertices   = hipgraph_labeling_result_get_vertices(p_result);
        components = hipgraph_labeling_result_get_labels(p_result);

        size_t num_local_vertices = hipgraph_type_erased_device_array_view_size(vertices);

        vertex_t h_vertices[num_local_vertices];
        vertex_t h_components[num_local_vertices];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_vertices, vertices, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_components, components, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        vertex_t component_check[num_vertices];
        for(vertex_t i = 0; i < num_vertices; ++i)
        {
            component_check[i] = num_vertices;
        }

        vertex_t num_errors = 0;
        for(vertex_t i = 0; i < num_local_vertices; ++i)
        {
            if(component_check[h_components[i]] == num_vertices)
            {
                component_check[h_components[i]] = h_result[h_vertices[i]];
            }
            else if(component_check[h_components[i]] != h_result[h_vertices[i]])
            {
                ++num_errors;
            }
        }

        EXPECT_EQ(num_errors, 0) << "weakly connected components results don't match";

        hipgraph_type_erased_device_array_view_free(components);
        hipgraph_type_erased_device_array_view_free(vertices);
        hipgraph_labeling_result_free(p_result);
        hipgraph_mg_graph_free(p_graph);
        hipgraph_error_free(ret_error);

        return test_ret_value;
    }

    int test_weakly_connected_components(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 32;
        size_t num_vertices = 12;

        vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 6, 7, 7,  8, 8, 8, 9,  10,
                            1, 3, 4, 0, 1, 3, 5, 5, 7, 9, 10, 6, 7, 9, 11, 11};
        vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 7, 9, 10, 6, 7, 9, 11, 11,
                            0, 1, 1, 2, 2, 2, 3, 4, 6, 7, 7,  8, 8, 8, 9,  10};
        weight_t h_wgt[]
            = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        vertex_t h_result[] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};

        // WCC wants store_transposed = HIPGRAPH_FALSE
        return generic_wcc_test(
            p_handle, h_src, h_dst, h_wgt, h_result, num_vertices, num_edges, HIPGRAPH_FALSE);
    }

} // namespace
