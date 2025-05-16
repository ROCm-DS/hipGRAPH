// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0
/*
 * Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

/*
 * Copyright (C) 2024, NVIDIA CORPORATION.
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
    int generic_ecg_test(const hipgraph_resource_handle_t* p_handle,
                         vertex_t*                         h_src,
                         vertex_t*                         h_dst,
                         weight_t*                         h_wgt,
                         size_t                            num_vertices,
                         size_t                            num_edges,
                         double                            min_weight,
                         size_t                            ensemble_size,
                         size_t                            max_level,
                         double                            threshold,
                         double                            resolution,
                         hipgraph_bool_t                   store_transposed)
    {
        int test_ret_value = 0;

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_graph_t*                          graph  = nullptr;
        hipgraph_hierarchical_clustering_result_t* result = nullptr;

        int rank = hipgraph_resource_handle_get_rank(p_handle);

        ret_code = create_mg_test_graph(p_handle,
                                        h_src,
                                        h_dst,
                                        h_wgt,
                                        num_edges,
                                        store_transposed,
                                        HIPGRAPH_FALSE,
                                        &graph,
                                        &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "create_test_graph failed. " << hipgraph_error_message(ret_error);

        hipgraph_rng_state_t* rng_state;
        ret_code = hipgraph_rng_state_create(p_handle, rank, &rng_state, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "rng_state create failed.";

        ret_code = hipgraph_ecg(p_handle,
                                rng_state,
                                graph,
                                min_weight,
                                ensemble_size,
                                max_level,
                                threshold,
                                resolution,
                                HIPGRAPH_FALSE,
                                &result,
                                &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "hipgraph_ecg failed. " << hipgraph_error_message(ret_error);

        if(test_ret_value == 0)
        {
            hipgraph_type_erased_device_array_view_t* vertices;
            hipgraph_type_erased_device_array_view_t* clusters;

            vertices          = hipgraph_hierarchical_clustering_result_get_vertices(result);
            clusters          = hipgraph_hierarchical_clustering_result_get_clusters(result);
            double modularity = hipgraph_hierarchical_clustering_result_get_modularity(result);

            vertex_t h_vertices[num_vertices];
            edge_t   h_clusters[num_vertices];

            ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (hipgraph_byte_t*)h_vertices, vertices, &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

            ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (hipgraph_byte_t*)h_clusters, clusters, &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

            size_t num_local_vertices = hipgraph_type_erased_device_array_view_size(vertices);

            // Louvain and permute_range are both tested, here we only make
            // sure that function calls succeed as expected.

            hipgraph_hierarchical_clustering_result_free(result);
        }

        hipgraph_mg_graph_free(graph);
        hipgraph_error_free(ret_error);

        return test_ret_value;
    }

    int test_ecg(const hipgraph_resource_handle_t* p_handle)
    {
        size_t   num_edges     = 16;
        size_t   num_vertices  = 6;
        size_t   max_level     = 10;
        weight_t threshold     = 1e-7;
        weight_t resolution    = 1.0;
        weight_t min_weight    = 0.001;
        size_t   ensemble_size = 10;

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

        // Louvain wants store_transposed = HIPGRAPH_FALSE
        return generic_ecg_test(p_handle,
                                h_src,
                                h_dst,
                                h_wgt,
                                num_vertices,
                                num_edges,
                                min_weight,
                                ensemble_size,
                                max_level,
                                threshold,
                                resolution,
                                HIPGRAPH_FALSE);
    }

    int test_ecg_no_weight(const hipgraph_resource_handle_t* p_handle)
    {
        size_t   num_edges     = 16;
        size_t   num_vertices  = 6;
        size_t   max_level     = 10;
        weight_t threshold     = 1e-7;
        weight_t resolution    = 1.0;
        weight_t min_weight    = 0.001;
        size_t   ensemble_size = 10;

        vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};

        // Louvain wants store_transposed = HIPGRAPH_FALSE
        return generic_ecg_test(p_handle,
                                h_src,
                                h_dst,
                                nullptr,
                                num_vertices,
                                num_edges,
                                min_weight,
                                ensemble_size,
                                max_level,
                                threshold,
                                resolution,
                                HIPGRAPH_FALSE);
    }

} // namespace
