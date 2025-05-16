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
    void generic_louvain_test(vertex_t*       h_src,
                              vertex_t*       h_dst,
                              weight_t*       h_wgt,
                              vertex_t*       h_result,
                              weight_t        expected_modularity,
                              size_t          num_vertices,
                              size_t          num_edges,
                              size_t          max_level,
                              double          threshold,
                              double          resolution,
                              hipgraph_bool_t store_transposed)
    {

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_resource_handle_t*                p_handle = nullptr;
        hipgraph_graph_t*                          p_graph  = nullptr;
        hipgraph_hierarchical_clustering_result_t* p_result = nullptr;

        hipgraph_data_type_id_t vertex_tid    = HIPGRAPH_INT32;
        hipgraph_data_type_id_t edge_tid      = HIPGRAPH_INT32;
        hipgraph_data_type_id_t weight_tid    = HIPGRAPH_FLOAT32;
        hipgraph_data_type_id_t edge_id_tid   = HIPGRAPH_INT32;
        hipgraph_data_type_id_t edge_type_tid = HIPGRAPH_INT32;

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
                             &p_graph,
                             &ret_error);

        ret_code = hipgraph_louvain(p_handle,
                                    p_graph,
                                    max_level,
                                    threshold,
                                    resolution,
                                    HIPGRAPH_FALSE,
                                    &p_result,
                                    &ret_error);

        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "hipgraph_louvain failed: " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* vertices;
        hipgraph_type_erased_device_array_view_t* clusters;

        vertices          = hipgraph_hierarchical_clustering_result_get_vertices(p_result);
        clusters          = hipgraph_hierarchical_clustering_result_get_clusters(p_result);
        double modularity = hipgraph_hierarchical_clustering_result_get_modularity(p_result);

        vertex_t h_vertices[num_vertices];
        edge_t   h_clusters[num_vertices];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_vertices, vertices, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_clusters, clusters, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "copy_to_host failed: " << hipgraph_error_message(ret_error);

        for(size_t i = 0; i < num_vertices; ++i)
        {
            EXPECT_EQ(h_result[h_vertices[i]], h_clusters[i])
                << "cluster results don't match at position " << i;
        }

        EXPECT_NEAR(modularity, expected_modularity, 0.001) << "modularity doesn't match";

        hipgraph_hierarchical_clustering_result_free(p_result);

        hipgraph_sg_graph_free(p_graph);
        hipgraph_free_resource_handle(p_handle);
        hipgraph_error_free(ret_error);
    }

    TEST(AlgorithmTest, Louvain)
    {
        size_t   num_edges    = 16;
        size_t   num_vertices = 6;
        size_t   max_level    = 10;
        weight_t threshold    = 1e-7;
        weight_t resolution   = 1.0;

        vertex_t h_src[]             = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_dst[]             = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
        weight_t h_wgt[]             = {0.1f,
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
        vertex_t h_result[]          = {0, 0, 0, 1, 1, 1};
        weight_t expected_modularity = 0.215969;

        // Louvain wants store_transposed = HIPGRAPH_FALSE
        generic_louvain_test(h_src,
                             h_dst,
                             h_wgt,
                             h_result,
                             expected_modularity,
                             num_vertices,
                             num_edges,
                             max_level,
                             threshold,
                             resolution,
                             HIPGRAPH_FALSE);
    }

    TEST(AlgorithmTest, LouvainNoWeight)
    {
        size_t   num_edges    = 16;
        size_t   num_vertices = 6;
        size_t   max_level    = 10;
        weight_t threshold    = 1e-7;
        weight_t resolution   = 1.0;

        vertex_t h_src[]             = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_dst[]             = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_result[]          = {1, 1, 1, 1, 0, 0};
        weight_t expected_modularity = 0.125;

        // Louvain wants store_transposed = HIPGRAPH_FALSE
        generic_louvain_test(h_src,
                             h_dst,
                             nullptr,
                             h_result,
                             expected_modularity,
                             num_vertices,
                             num_edges,
                             max_level,
                             threshold,
                             resolution,
                             HIPGRAPH_FALSE);
    }

} // namespace
