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
    void generic_spectral_test(vertex_t*       h_src,
                               vertex_t*       h_dst,
                               weight_t*       h_wgt,
                               vertex_t*       h_result,
                               weight_t        expected_modularity,
                               weight_t        expected_edge_cut,
                               weight_t        expected_ratio_cut,
                               size_t          num_vertices,
                               size_t          num_edges,
                               size_t          num_clusters,
                               size_t          num_eigenvectors,
                               double          evs_tolerance,
                               int             evs_max_iterations,
                               double          k_means_tolerance,
                               int             k_means_max_iterations,
                               hipgraph_bool_t store_transposed)
    {

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_resource_handle_t*   p_handle = nullptr;
        hipgraph_graph_t*             graph    = nullptr;
        hipgraph_clustering_result_t* result   = nullptr;

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
                             &graph,
                             &ret_error);

        ret_code = hipgraph_spectral_modularity_maximization(p_handle,
                                                             graph,
                                                             num_clusters,
                                                             num_eigenvectors,
                                                             evs_tolerance,
                                                             evs_max_iterations,
                                                             k_means_tolerance,
                                                             k_means_max_iterations,
                                                             HIPGRAPH_FALSE,
                                                             &result,
                                                             &ret_error);

        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "hipgraph_spectral_modularity_maximization failed: "
            << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* vertices;
        hipgraph_type_erased_device_array_view_t* clusters;
        double                                    modularity;
        double                                    edge_cut;
        double                                    ratio_cut;

        vertices = hipgraph_clustering_result_get_vertices(result);
        clusters = hipgraph_clustering_result_get_clusters(result);

        ret_code = hipgraph_analyze_clustering_modularity(
            p_handle, graph, num_clusters, vertices, clusters, &modularity, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS) << hipgraph_error_message(ret_error);

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

        for(int i = 0; i < num_vertices; ++i)
        {
            EXPECT_EQ(h_result[h_vertices[i]], h_clusters[i])
                << "cluster results don't match at position " << i;
        }

        EXPECT_NEAR(modularity, expected_modularity, 0.001) << "modularity doesn't match";

        EXPECT_NEAR(edge_cut, expected_edge_cut, 0.001) << "edge_cut doesn't match";

        EXPECT_NEAR(ratio_cut, expected_ratio_cut, 0.001) << "ratio_cut doesn't match";

        hipgraph_clustering_result_free(result);

        hipgraph_sg_graph_free(graph);
        hipgraph_free_resource_handle(p_handle);
        hipgraph_error_free(ret_error);
    }

    void generic_balanced_cut_test(vertex_t*       h_src,
                                   vertex_t*       h_dst,
                                   weight_t*       h_wgt,
                                   vertex_t*       h_result,
                                   weight_t        expected_modularity,
                                   weight_t        expected_edge_cut,
                                   weight_t        expected_ratio_cut,
                                   size_t          num_vertices,
                                   size_t          num_edges,
                                   size_t          num_clusters,
                                   size_t          num_eigenvectors,
                                   double          evs_tolerance,
                                   int             evs_max_iterations,
                                   double          k_means_tolerance,
                                   int             k_means_max_iterations,
                                   hipgraph_bool_t store_transposed)
    {

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_data_type_id_t vertex_tid    = HIPGRAPH_INT32;
        hipgraph_data_type_id_t edge_tid      = HIPGRAPH_INT32;
        hipgraph_data_type_id_t weight_tid    = HIPGRAPH_FLOAT32;
        hipgraph_data_type_id_t edge_id_tid   = HIPGRAPH_INT32;
        hipgraph_data_type_id_t edge_type_tid = HIPGRAPH_INT32;

        hipgraph_resource_handle_t*   p_handle = nullptr;
        hipgraph_graph_t*             graph    = nullptr;
        hipgraph_clustering_result_t* result   = nullptr;

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

        ret_code = hipgraph_balanced_cut_clustering(p_handle,
                                                    graph,
                                                    num_clusters,
                                                    num_eigenvectors,
                                                    evs_tolerance,
                                                    evs_max_iterations,
                                                    k_means_tolerance,
                                                    k_means_max_iterations,
                                                    HIPGRAPH_FALSE,
                                                    &result,
                                                    &ret_error);

        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "hipgraph_spectral_modularity_maximization failed: "
            << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* vertices;
        hipgraph_type_erased_device_array_view_t* clusters;
        double                                    modularity;
        double                                    edge_cut;
        double                                    ratio_cut;

        vertices = hipgraph_clustering_result_get_vertices(result);
        clusters = hipgraph_clustering_result_get_clusters(result);

        ret_code = hipgraph_analyze_clustering_modularity(
            p_handle, graph, num_clusters, vertices, clusters, &modularity, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS) << hipgraph_error_message(ret_error);

        ret_code = hipgraph_analyze_clustering_edge_cut(
            p_handle, graph, num_clusters, vertices, clusters, &edge_cut, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS) << hipgraph_error_message(ret_error);

        ret_code = hipgraph_analyze_clustering_ratio_cut(
            p_handle, graph, num_clusters, vertices, clusters, &ratio_cut, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS) << hipgraph_error_message(ret_error);

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

        for(int i = 0; i < num_vertices; ++i)
        {
            EXPECT_EQ(h_result[h_vertices[i]], h_clusters[i])
                << "cluster results don't match at position " << i;
        }

        EXPECT_NEAR(modularity, expected_modularity, 0.001) << "modularity doesn't match";

        EXPECT_NEAR(edge_cut, expected_edge_cut, 0.001) << "edge_cut doesn't match";

        EXPECT_NEAR(ratio_cut, expected_ratio_cut, 0.001) << "ratio_cut doesn't match";

        hipgraph_clustering_result_free(result);

        hipgraph_sg_graph_free(graph);
        hipgraph_free_resource_handle(p_handle);
        hipgraph_error_free(ret_error);
    }

    TEST(BrokenTest, Spectral)
    {
        size_t num_clusters           = 2;
        size_t num_eigenvectors       = 2;
        size_t num_edges              = 14;
        size_t num_vertices           = 6;
        double evs_tolerance          = 0.001;
        int    evs_max_iterations     = 100;
        double k_means_tolerance      = 0.001;
        int    k_means_max_iterations = 100;

        vertex_t h_src[] = {0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5};
        vertex_t h_dst[] = {1, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 5, 3, 4};
        weight_t h_wgt[]
            = {0.1f, 0.2f, 0.1f, 1.2f, 0.2f, 1.2f, 2.3f, 2.3f, 3.4f, 3.5f, 3.4f, 4.5f, 3.5f, 4.5f};
        vertex_t h_result[]          = {0, 0, 0, 1, 1, 1};
        weight_t expected_modularity = 0.136578;
        weight_t expected_edge_cut   = 0;
        weight_t expected_ratio_cut  = 0;

        // spectral clustering wants store_transposed = HIPGRAPH_FALSE
        generic_spectral_test(h_src,
                              h_dst,
                              h_wgt,
                              h_result,
                              expected_modularity,
                              expected_edge_cut,
                              expected_ratio_cut,
                              num_vertices,
                              num_edges,
                              num_clusters,
                              num_eigenvectors,
                              evs_tolerance,
                              evs_max_iterations,
                              k_means_tolerance,
                              k_means_max_iterations,
                              HIPGRAPH_FALSE);
    }

    TEST(BrokenTest, BalancedCutUnequalWeight)
    {
        size_t num_clusters           = 2;
        size_t num_eigenvectors       = 2;
        size_t num_edges              = 14;
        size_t num_vertices           = 6;
        double evs_tolerance          = 0.001;
        int    evs_max_iterations     = 100;
        double k_means_tolerance      = 0.001;
        int    k_means_max_iterations = 100;

        vertex_t h_src[] = {0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5};
        vertex_t h_dst[] = {1, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 5, 3, 4};
        weight_t h_wgt[]
            = {0.1f, 0.2f, 0.1f, 1.2f, 0.2f, 1.2f, 2.3f, 2.3f, 3.4f, 3.5f, 3.4f, 4.5f, 3.5f, 4.5f};
        vertex_t h_result[]          = {0, 0, 1, 0, 0, 0};
        weight_t expected_modularity = -0.02963;
        weight_t expected_edge_cut   = 3.7;
        weight_t expected_ratio_cut  = 4.44;

        // balanced cut clustering wants store_transposed = HIPGRAPH_FALSE
        generic_balanced_cut_test(h_src,
                                  h_dst,
                                  h_wgt,
                                  h_result,
                                  expected_modularity,
                                  expected_edge_cut,
                                  expected_ratio_cut,
                                  num_vertices,
                                  num_edges,
                                  num_clusters,
                                  num_eigenvectors,
                                  evs_tolerance,
                                  evs_max_iterations,
                                  k_means_tolerance,
                                  k_means_max_iterations,
                                  HIPGRAPH_FALSE);
    }

    TEST(BrokenTest, BalancedCutEqualWeight)
    {
        size_t num_clusters           = 2;
        size_t num_eigenvectors       = 2;
        size_t num_edges              = 14;
        size_t num_vertices           = 6;
        double evs_tolerance          = 0.001;
        int    evs_max_iterations     = 100;
        double k_means_tolerance      = 0.001;
        int    k_means_max_iterations = 100;

        vertex_t h_src[]             = {0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5};
        vertex_t h_dst[]             = {1, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 5, 3, 4};
        weight_t h_wgt[]             = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        vertex_t h_result[]          = {1, 1, 1, 0, 0, 0};
        weight_t expected_modularity = 0.357143;
        weight_t expected_edge_cut   = 1;
        weight_t expected_ratio_cut  = 0.666667;

        // balanced cut clustering wants store_transposed = HIPGRAPH_FALSE
        generic_balanced_cut_test(h_src,
                                  h_dst,
                                  h_wgt,
                                  h_result,
                                  expected_modularity,
                                  expected_edge_cut,
                                  expected_ratio_cut,
                                  num_vertices,
                                  num_edges,
                                  num_clusters,
                                  num_eigenvectors,
                                  evs_tolerance,
                                  evs_max_iterations,
                                  k_means_tolerance,
                                  k_means_max_iterations,
                                  HIPGRAPH_FALSE);
    }

    TEST(BrokenTest, BalancedCutNoWeight)
    {
        size_t num_clusters           = 2;
        size_t num_eigenvectors       = 2;
        size_t num_edges              = 14;
        size_t num_vertices           = 6;
        double evs_tolerance          = 0.001;
        int    evs_max_iterations     = 100;
        double k_means_tolerance      = 0.001;
        int    k_means_max_iterations = 100;

        vertex_t h_src[]             = {0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5};
        vertex_t h_dst[]             = {1, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 5, 3, 4};
        vertex_t h_result[]          = {1, 1, 1, 0, 0, 0};
        weight_t expected_modularity = 0.357143;
        weight_t expected_edge_cut   = 1;
        weight_t expected_ratio_cut  = 0.666667;

        // balanced cut clustering wants store_transposed = HIPGRAPH_FALSE
        generic_balanced_cut_test(h_src,
                                  h_dst,
                                  nullptr,
                                  h_result,
                                  expected_modularity,
                                  expected_edge_cut,
                                  expected_ratio_cut,
                                  num_vertices,
                                  num_edges,
                                  num_clusters,
                                  num_eigenvectors,
                                  evs_tolerance,
                                  evs_max_iterations,
                                  k_means_tolerance,
                                  k_means_max_iterations,
                                  HIPGRAPH_FALSE);
    }

} // namespace
