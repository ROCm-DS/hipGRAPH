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
    int generic_two_hop_nbr_test(const hipgraph_resource_handle_t* resource_handle,
                                 vertex_t*                         h_src,
                                 vertex_t*                         h_dst,
                                 weight_t*                         h_wgt,
                                 vertex_t*                         h_sources,
                                 vertex_t*                         h_result_v1,
                                 vertex_t*                         h_result_v2,
                                 size_t                            num_vertices,
                                 size_t                            num_edges,
                                 size_t                            num_sources,
                                 size_t                            num_result_pairs,
                                 hipgraph_bool_t                   store_transposed)
    {
        int test_ret_value = 0;

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_graph_t*                         graph               = nullptr;
        hipgraph_type_erased_device_array_t*      start_vertices      = nullptr;
        hipgraph_type_erased_device_array_view_t* start_vertices_view = nullptr;
        hipgraph_vertex_pairs_t*                  result              = nullptr;

        int rank = hipgraph_resource_handle_get_rank(resource_handle);

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

        if(num_sources > 0)
        {
            if(rank == 0)
            {
                ret_code = hipgraph_type_erased_device_array_create(
                    resource_handle, num_sources, HIPGRAPH_INT32, &start_vertices, &ret_error);
                EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "p_sources create failed.";

                start_vertices_view = hipgraph_type_erased_device_array_view(start_vertices);

                ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
                    resource_handle, start_vertices_view, (hipgraph_byte_t*)h_sources, &ret_error);
                EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "src copy_from_host failed.";
            }
            else
            {
                start_vertices_view
                    = hipgraph_type_erased_device_array_view_create(nullptr, 0, HIPGRAPH_INT32);
            }
        }

        ret_code = hipgraph_two_hop_neighbors(
            resource_handle, graph, start_vertices_view, HIPGRAPH_FALSE, &result, &ret_error);

        hipgraph_type_erased_device_array_view_t const* v1;
        hipgraph_type_erased_device_array_view_t const* v2;

        v1 = hipgraph_vertex_pairs_get_first(result);
        v2 = hipgraph_vertex_pairs_get_second(result);

        size_t number_of_pairs = hipgraph_type_erased_device_array_view_size(v1);

        vertex_t h_v1[number_of_pairs];
        vertex_t h_v2[number_of_pairs];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            resource_handle, (hipgraph_byte_t*)h_v1, v1, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            resource_handle, (hipgraph_byte_t*)h_v2, v2, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        hipgraph_bool_t M[num_vertices][num_vertices];
        for(int i = 0; i < num_vertices; ++i)
            for(int j = 0; j < num_vertices; ++j)
                M[i][j] = HIPGRAPH_FALSE;

        for(int i = 0; i < num_result_pairs; ++i)
            M[h_result_v1[i]][h_result_v2[i]] = HIPGRAPH_TRUE;

        for(int i = 0; (i < number_of_pairs) && (test_ret_value == 0); ++i)
        {
            EXPECT_TRUE(M[h_v1[i]][h_v2[i]]) << "result not found";
        }

        hipgraph_vertex_pairs_free(result);
        hipgraph_type_erased_device_array_view_free(start_vertices_view);
        hipgraph_type_erased_device_array_free(start_vertices);
        hipgraph_mg_graph_free(graph);
        hipgraph_error_free(ret_error);

        return test_ret_value;
    }

    int test_two_hop_nbr_all(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges        = 22;
        size_t num_vertices     = 7;
        size_t num_sources      = 0;
        size_t num_result_pairs = 43;

        vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5, 3, 1, 4, 5, 5, 6};
        vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 3, 1, 6, 5};
        weight_t h_wgt[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

        vertex_t h_result_v1[] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3,
                                  3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6};
        vertex_t h_result_v2[] = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 0, 1, 2,
                                  3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 1, 3, 4, 6};

        return generic_two_hop_nbr_test(p_handle,
                                        h_src,
                                        h_dst,
                                        h_wgt,
                                        nullptr,
                                        h_result_v1,
                                        h_result_v2,
                                        num_vertices,
                                        num_edges,
                                        num_sources,
                                        num_result_pairs,
                                        HIPGRAPH_FALSE);
    }

    int test_two_hop_nbr_one(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges        = 22;
        size_t num_vertices     = 7;
        size_t num_sources      = 1;
        size_t num_result_pairs = 6;

        vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5, 3, 1, 4, 5, 5, 6};
        vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 3, 1, 6, 5};
        weight_t h_wgt[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

        vertex_t h_sources[] = {0};

        vertex_t h_result_v1[] = {0, 0, 0, 0, 0, 0};
        vertex_t h_result_v2[] = {0, 1, 2, 3, 4, 5};

        return generic_two_hop_nbr_test(p_handle,
                                        h_src,
                                        h_dst,
                                        h_wgt,
                                        h_sources,
                                        h_result_v1,
                                        h_result_v2,
                                        num_vertices,
                                        num_edges,
                                        num_sources,
                                        num_result_pairs,
                                        HIPGRAPH_FALSE);
    }

} // namespace
