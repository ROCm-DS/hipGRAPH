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

namespace
{
    using namespace hipGRAPH::testing;
    int generic_k_core_test(const hipgraph_resource_handle_t* resource_handle,
                            vertex_t*                         h_src,
                            vertex_t*                         h_dst,
                            weight_t*                         h_wgt,
                            vertex_t*                         h_result_src,
                            vertex_t*                         h_result_dst,
                            weight_t*                         h_result_wgt,
                            size_t                            num_vertices,
                            size_t                            num_edges,
                            size_t                            num_result_edges,
                            size_t                            k,
                            hipgraph_bool_t                   store_transposed)
    {
        int test_ret_value = 0;

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_graph_t*         graph         = nullptr;
        hipgraph_core_result_t*   core_result   = nullptr;
        hipgraph_k_core_result_t* k_core_result = nullptr;

        ret_code = create_mg_test_graph(resource_handle,
                                        h_src,
                                        h_dst,
                                        h_wgt,
                                        num_edges,
                                        store_transposed,
                                        HIPGRAPH_TRUE,
                                        &graph,
                                        &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "create_test_graph failed. " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_core_number(resource_handle,
                                        graph,
                                        HIPGRAPH_K_CORE_DEGREE_TYPE_IN,
                                        HIPGRAPH_FALSE,
                                        &core_result,
                                        &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "hipgraph_core_number failed. " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_k_core(resource_handle,
                                   graph,
                                   k,
                                   HIPGRAPH_K_CORE_DEGREE_TYPE_IN,
                                   core_result,
                                   HIPGRAPH_FALSE,
                                   &k_core_result,
                                   &ret_error);

        hipgraph_type_erased_device_array_view_t* src_vertices;
        hipgraph_type_erased_device_array_view_t* dst_vertices;
        hipgraph_type_erased_device_array_view_t* weights;

        src_vertices = hipgraph_k_core_result_get_src_vertices(k_core_result);
        dst_vertices = hipgraph_k_core_result_get_dst_vertices(k_core_result);
        weights      = hipgraph_k_core_result_get_weights(k_core_result);

        size_t number_of_result_edges = hipgraph_type_erased_device_array_view_size(src_vertices);

        vertex_t h_src_vertices[number_of_result_edges];
        vertex_t h_dst_vertices[number_of_result_edges];
        weight_t h_weights[number_of_result_edges];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            resource_handle, (hipgraph_byte_t*)h_src_vertices, src_vertices, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            resource_handle, (hipgraph_byte_t*)h_dst_vertices, dst_vertices, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            resource_handle, (hipgraph_byte_t*)h_weights, weights, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        weight_t M[num_vertices][num_vertices];
        for(int i = 0; i < num_vertices; ++i)
            for(int j = 0; j < num_vertices; ++j)
                M[i][j] = 0;

        for(int i = 0; i < num_result_edges; ++i)
            M[h_result_src[i]][h_result_dst[i]] = h_result_wgt[i];

        for(int i = 0; (i < number_of_result_edges) && (test_ret_value == 0); ++i)
        {
            EXPECT_EQ(M[h_src_vertices[i]][h_dst_vertices[i]], h_weights[i])
                << "edge does not match";
        }

        hipgraph_k_core_result_free(k_core_result);
        hipgraph_core_result_free(core_result);
        hipgraph_mg_graph_free(graph);
        hipgraph_error_free(ret_error);

        return test_ret_value;
    }

    int test_k_core(const hipgraph_resource_handle_t* resource_handle)
    {
        size_t num_edges        = 22;
        size_t num_vertices     = 7;
        size_t num_result_edges = 12;
        size_t k                = 3;

        vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5, 3, 1, 4, 5, 5, 6};
        vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 3, 1, 6, 5};
        weight_t h_wgt[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        vertex_t h_result_src[] = {1, 1, 3, 4, 3, 4, 3, 4, 5, 5, 1, 5};
        vertex_t h_result_dst[] = {3, 4, 5, 5, 1, 3, 4, 1, 3, 4, 5, 1};
        weight_t h_result_wgt[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

        return generic_k_core_test(resource_handle,
                                   h_src,
                                   h_dst,
                                   h_wgt,
                                   h_result_src,
                                   h_result_dst,
                                   h_result_wgt,
                                   num_vertices,
                                   num_edges,
                                   num_result_edges,
                                   k,
                                   HIPGRAPH_FALSE);
    }

} // namespace
