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

#include <float.h>
#include <cmath>

using vertex_t = int32_t;
using edge_t   = int32_t;

const float EPSILON = 0.001;

namespace
{
    using namespace hipGRAPH::testing;
    int generic_sssp_test(const hipgraph_resource_handle_t* p_handle,
                          vertex_t*                         h_src,
                          vertex_t*                         h_dst,
                          float*                            h_wgt,
                          vertex_t                          source,
                          float const*                      expected_distances,
                          vertex_t const*                   expected_predecessors,
                          size_t                            num_vertices,
                          size_t                            num_edges,
                          float                             cutoff,
                          hipgraph_bool_t                   store_transposed)
    {
        int test_ret_value = 0;

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_graph_t*        p_graph  = nullptr;
        hipgraph_paths_result_t* p_result = nullptr;

        ret_code = create_mg_test_graph(p_handle,
                                        h_src,
                                        h_dst,
                                        h_wgt,
                                        num_edges,
                                        store_transposed,
                                        HIPGRAPH_FALSE,
                                        &p_graph,
                                        &ret_error);

        ret_code = hipgraph_sssp(p_handle,
                                 p_graph,
                                 source,
                                 cutoff,
                                 HIPGRAPH_TRUE,
                                 HIPGRAPH_FALSE,
                                 &p_result,
                                 &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "hipgraph_sssp failed. " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* vertices;
        hipgraph_type_erased_device_array_view_t* distances;
        hipgraph_type_erased_device_array_view_t* predecessors;

        vertices     = hipgraph_paths_result_get_vertices(p_result);
        distances    = hipgraph_paths_result_get_distances(p_result);
        predecessors = hipgraph_paths_result_get_predecessors(p_result);

        vertex_t h_vertices[num_vertices];
        float    h_distances[num_vertices];
        vertex_t h_predecessors[num_vertices];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_vertices, vertices, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_distances, distances, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_predecessors, predecessors, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        size_t num_local_vertices = hipgraph_type_erased_device_array_view_size(vertices);

        for(int i = 0; (i < num_local_vertices) && (test_ret_value == 0); ++i)
        {
            EXPECT_NEAR(expected_distances[h_vertices[i]], h_distances[i], EPSILON)
                << "sssp distances don't match";

            EXPECT_EQ(expected_predecessors[h_vertices[i]], h_predecessors[i])
                << "sssp predecessors don't match";
        }

        hipgraph_type_erased_device_array_view_free(vertices);
        hipgraph_type_erased_device_array_view_free(distances);
        hipgraph_type_erased_device_array_view_free(predecessors);
        hipgraph_paths_result_free(p_result);
        hipgraph_sg_graph_free(p_graph);
        hipgraph_error_free(ret_error);

        return test_ret_value;
    }

    int generic_sssp_test_double(const hipgraph_resource_handle_t* p_handle,
                                 vertex_t*                         h_src,
                                 vertex_t*                         h_dst,
                                 double*                           h_wgt,
                                 vertex_t                          source,
                                 double const*                     expected_distances,
                                 vertex_t const*                   expected_predecessors,
                                 size_t                            num_vertices,
                                 size_t                            num_edges,
                                 double                            cutoff,
                                 hipgraph_bool_t                   store_transposed)
    {
        int test_ret_value = 0;

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_graph_t*        p_graph  = nullptr;
        hipgraph_paths_result_t* p_result = nullptr;

        ret_code = create_mg_test_graph_double(p_handle,
                                               h_src,
                                               h_dst,
                                               h_wgt,
                                               num_edges,
                                               store_transposed,
                                               HIPGRAPH_FALSE,
                                               &p_graph,
                                               &ret_error);

        ret_code = hipgraph_sssp(p_handle,
                                 p_graph,
                                 source,
                                 cutoff,
                                 HIPGRAPH_TRUE,
                                 HIPGRAPH_FALSE,
                                 &p_result,
                                 &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "hipgraph_sssp failed. " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* vertices;
        hipgraph_type_erased_device_array_view_t* distances;
        hipgraph_type_erased_device_array_view_t* predecessors;

        vertices     = hipgraph_paths_result_get_vertices(p_result);
        distances    = hipgraph_paths_result_get_distances(p_result);
        predecessors = hipgraph_paths_result_get_predecessors(p_result);

        vertex_t h_vertices[num_vertices];
        double   h_distances[num_vertices];
        vertex_t h_predecessors[num_vertices];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_vertices, vertices, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_distances, distances, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_predecessors, predecessors, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        size_t num_local_vertices = hipgraph_type_erased_device_array_view_size(vertices);

        for(int i = 0; (i < num_local_vertices) && (test_ret_value == 0); ++i)
        {
            EXPECT_EQ(nearlyEqualDouble(expected_distances[h_vertices[i]], h_distances[i], EPSILON))
                << "sssp distances don't match";

            EXPECT_TRUE(expected_predecessors[h_vertices[i]], h_predecessors[i])
                << "sssp predecessors don't match";
        }

        hipgraph_type_erased_device_array_view_free(vertices);
        hipgraph_type_erased_device_array_view_free(distances);
        hipgraph_type_erased_device_array_view_free(predecessors);
        hipgraph_paths_result_free(p_result);
        hipgraph_sg_graph_free(p_graph);
        hipgraph_error_free(ret_error);

        return test_ret_value;
    }

    int test_sssp(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t src[]                   = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]                   = {1, 3, 4, 0, 1, 3, 5, 5};
        float    wgt[]                   = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        float    expected_distances[]    = {0.0f, 0.1f, FLT_MAX, 2.2f, 1.2f, 4.4f};
        vertex_t expected_predecessors[] = {-1, 0, -1, 1, 1, 4};

        // Bfs wants store_transposed = HIPGRAPH_FALSE
        return generic_sssp_test(p_handle,
                                 src,
                                 dst,
                                 wgt,
                                 0,
                                 expected_distances,
                                 expected_predecessors,
                                 num_vertices,
                                 num_edges,
                                 10,
                                 HIPGRAPH_FALSE);
    }

    int test_sssp_with_transpose(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t src[]                   = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]                   = {1, 3, 4, 0, 1, 3, 5, 5};
        float    wgt[]                   = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        float    expected_distances[]    = {0.0f, 0.1f, FLT_MAX, 2.2f, 1.2f, 4.4f};
        vertex_t expected_predecessors[] = {-1, 0, -1, 1, 1, 4};

        // Bfs wants store_transposed = HIPGRAPH_FALSE
        //    This call will force hipgraph_sssp to transpose the graph
        return generic_sssp_test(p_handle,
                                 src,
                                 dst,
                                 wgt,
                                 0,
                                 expected_distances,
                                 expected_predecessors,
                                 num_vertices,
                                 num_edges,
                                 10,
                                 HIPGRAPH_TRUE);
    }

    int test_sssp_with_transpose_double(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t src[]                   = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t dst[]                   = {1, 3, 4, 0, 1, 3, 5, 5};
        double   wgt[]                   = {0.1d, 2.1d, 1.1d, 5.1d, 3.1d, 4.1d, 7.2d, 3.2d};
        double   expected_distances[]    = {0.0d, 0.1d, DBL_MAX, 2.2d, 1.2d, 4.4d};
        vertex_t expected_predecessors[] = {-1, 0, -1, 1, 1, 4};

        // Bfs wants store_transposed = HIPGRAPH_FALSE
        //    This call will force hipgraph_sssp to transpose the graph
        return generic_sssp_test_double(p_handle,
                                        src,
                                        dst,
                                        wgt,
                                        0,
                                        expected_distances,
                                        expected_predecessors,
                                        num_vertices,
                                        num_edges,
                                        10,
                                        HIPGRAPH_TRUE);
    }

} // namespace
