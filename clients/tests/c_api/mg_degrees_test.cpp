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
    int generic_degrees_test(const hipgraph_resource_handle_t* p_handle,
                             vertex_t*                         h_src,
                             vertex_t*                         h_dst,
                             weight_t*                         h_wgt,
                             size_t                            num_vertices,
                             size_t                            num_edges,
                             vertex_t*                         h_vertices,
                             size_t                            num_vertices_to_compute,
                             hipgraph_bool_t                   in_degrees,
                             hipgraph_bool_t                   out_degrees,
                             hipgraph_bool_t                   store_transposed,
                             hipgraph_bool_t                   is_symmetric,
                             edge_t*                           h_in_degrees,
                             edge_t*                           h_out_degrees)
    {
        int test_ret_value = 0;

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_graph_t*          graph  = nullptr;
        hipgraph_degrees_result_t* result = nullptr;

        ret_code = create_mg_test_graph(p_handle,
                                        h_src,
                                        h_dst,
                                        h_wgt,
                                        num_edges,
                                        store_transposed,
                                        is_symmetric,
                                        &graph,
                                        &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "create_test_graph failed. " << hipgraph_error_message(ret_error);

        if(h_vertices == nullptr)
        {
            if(in_degrees && out_degrees)
            {
                ret_code = hipgraph_degrees(
                    p_handle, graph, nullptr, HIPGRAPH_FALSE, &result, &ret_error);
            }
            else if(in_degrees)
            {
                ret_code = hipgraph_in_degrees(
                    p_handle, graph, nullptr, HIPGRAPH_FALSE, &result, &ret_error);
            }
            else
            {
                ret_code = hipgraph_out_degrees(
                    p_handle, graph, nullptr, HIPGRAPH_FALSE, &result, &ret_error);
            }

            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "hipgraph_extract_degrees failed.";
        }
        else
        {
            hipgraph_type_erased_device_array_t*      vertices      = nullptr;
            hipgraph_type_erased_device_array_view_t* vertices_view = nullptr;

            int rank = hipgraph_resource_handle_get_rank(p_handle);

            size_t num_to_allocate = 0;
            if(rank == 0)
                num_to_allocate = num_vertices_to_compute;

            ret_code = hipgraph_type_erased_device_array_create(
                p_handle, num_to_allocate, HIPGRAPH_INT32, &vertices, &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "seeds create failed.";

            vertices_view = hipgraph_type_erased_device_array_view(vertices);

            ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
                p_handle, vertices_view, (hipgraph_byte_t*)h_vertices, &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "src copy_from_host failed.";

            if(in_degrees && out_degrees)
            {
                ret_code = hipgraph_degrees(
                    p_handle, graph, vertices_view, HIPGRAPH_FALSE, &result, &ret_error);
            }
            else if(in_degrees)
            {
                ret_code = hipgraph_in_degrees(
                    p_handle, graph, vertices_view, HIPGRAPH_FALSE, &result, &ret_error);
            }
            else
            {
                ret_code = hipgraph_out_degrees(
                    p_handle, graph, vertices_view, HIPGRAPH_FALSE, &result, &ret_error);
            }

            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "hipgraph_extract_degrees failed.";
        }

        hipgraph_type_erased_device_array_view_t* result_vertices;
        hipgraph_type_erased_device_array_view_t* result_in_degrees;
        hipgraph_type_erased_device_array_view_t* result_out_degrees;

        result_vertices    = hipgraph_degrees_result_get_vertices(result);
        result_in_degrees  = hipgraph_degrees_result_get_in_degrees(result);
        result_out_degrees = hipgraph_degrees_result_get_out_degrees(result);

        size_t num_result_vertices = hipgraph_type_erased_device_array_view_size(result_vertices);

        vertex_t h_result_vertices[num_result_vertices];
        edge_t   h_result_in_degrees[num_result_vertices];
        edge_t   h_result_out_degrees[num_result_vertices];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_result_vertices, result_vertices, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        if(result_in_degrees != nullptr)
        {
            ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (hipgraph_byte_t*)h_result_in_degrees, result_in_degrees, &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";
        }

        if(result_out_degrees != nullptr)
        {
            ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (hipgraph_byte_t*)h_result_out_degrees, result_out_degrees, &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";
        }

        if(h_vertices != nullptr)
        {
            size_t xxx = hipgraph_size_t_allreduce(p_handle, num_result_vertices);
            EXPECT_EQ(hipgraph_size_t_allreduce(p_handle, num_result_vertices),
                      num_vertices_to_compute)
                << "results not the same size";
        }
        else
        {
            size_t xxx = hipgraph_size_t_allreduce(p_handle, num_result_vertices);
            EXPECT_EQ(hipgraph_size_t_allreduce(p_handle, num_result_vertices), num_vertices)
                << "results not the same size";
        }

        for(size_t i = 0; (i < num_result_vertices) && (test_ret_value == 0); ++i)
        {
            if(h_in_degrees != nullptr)
            {
                EXPECT_EQ(h_result_in_degrees[i], h_in_degrees[h_result_vertices[i]])
                    << "in degree did not match";
            }

            if(h_out_degrees != nullptr)
            {
                EXPECT_EQ(h_result_out_degrees[i], h_out_degrees[h_result_vertices[i]])
                    << "out degree did not match";
            }
        }

        hipgraph_degrees_result_free(result);
        hipgraph_graph_free(graph);
        hipgraph_error_free(ret_error);
        return test_ret_value;
    }

    int test_degrees(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t h_src[]         = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]         = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]         = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t h_in_degrees[]  = {1, 2, 0, 2, 1, 2};
        vertex_t h_out_degrees[] = {1, 2, 3, 1, 1, 0};

        // Pagerank wants store_transposed = HIPGRAPH_TRUE
        return generic_degrees_test(p_handle,
                                    h_src,
                                    h_dst,
                                    h_wgt,
                                    num_vertices,
                                    num_edges,
                                    nullptr,
                                    0,
                                    HIPGRAPH_TRUE,
                                    HIPGRAPH_TRUE,
                                    HIPGRAPH_TRUE,
                                    HIPGRAPH_FALSE,
                                    h_in_degrees,
                                    h_out_degrees);
    }

    int test_degrees_symmetric(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 16;
        size_t num_vertices = 6;

        vertex_t h_src[]         = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_dst[]         = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
        weight_t h_wgt[]         = {0.1f,
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
        vertex_t h_in_degrees[]  = {2, 4, 3, 3, 2, 2};
        vertex_t h_out_degrees[] = {2, 4, 3, 3, 2, 2};

        // Pagerank wants store_transposed = HIPGRAPH_TRUE
        return generic_degrees_test(p_handle,
                                    h_src,
                                    h_dst,
                                    h_wgt,
                                    num_vertices,
                                    num_edges,
                                    nullptr,
                                    0,
                                    HIPGRAPH_TRUE,
                                    HIPGRAPH_TRUE,
                                    HIPGRAPH_TRUE,
                                    HIPGRAPH_TRUE,
                                    h_in_degrees,
                                    h_out_degrees);
    }

    int test_in_degrees(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t h_src[]        = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]        = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]        = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t h_in_degrees[] = {1, 2, 0, 2, 1, 2};

        return generic_degrees_test(p_handle,
                                    h_src,
                                    h_dst,
                                    h_wgt,
                                    num_vertices,
                                    num_edges,
                                    nullptr,
                                    0,
                                    HIPGRAPH_TRUE,
                                    HIPGRAPH_FALSE,
                                    HIPGRAPH_FALSE,
                                    HIPGRAPH_TRUE,
                                    h_in_degrees,
                                    nullptr);
    }

    int test_out_degrees(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t h_src[]         = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]         = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]         = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t h_out_degrees[] = {1, 2, 3, 1, 1, 0};

        return generic_degrees_test(p_handle,
                                    h_src,
                                    h_dst,
                                    h_wgt,
                                    num_vertices,
                                    num_edges,
                                    nullptr,
                                    0,
                                    HIPGRAPH_FALSE,
                                    HIPGRAPH_TRUE,
                                    HIPGRAPH_FALSE,
                                    HIPGRAPH_TRUE,
                                    nullptr,
                                    h_out_degrees);
    }

    int test_degrees_subset(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges               = 8;
        size_t num_vertices            = 6;
        size_t num_vertices_to_compute = 3;

        vertex_t h_src[]         = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]         = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]         = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t h_vertices[]    = {2, 3, 5};
        vertex_t h_in_degrees[]  = {-1, -1, 0, 2, -1, 2};
        vertex_t h_out_degrees[] = {-1, -1, 3, 1, -1, 0};

        return generic_degrees_test(p_handle,
                                    h_src,
                                    h_dst,
                                    h_wgt,
                                    num_vertices,
                                    num_edges,
                                    h_vertices,
                                    num_vertices_to_compute,
                                    HIPGRAPH_TRUE,
                                    HIPGRAPH_TRUE,
                                    HIPGRAPH_FALSE,
                                    HIPGRAPH_FALSE,
                                    h_in_degrees,
                                    h_out_degrees);
    }

    int test_degrees_symmetric_subset(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges               = 16;
        size_t num_vertices            = 6;
        size_t num_vertices_to_compute = 3;

        vertex_t h_src[]         = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
        vertex_t h_dst[]         = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
        weight_t h_wgt[]         = {0.1f,
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
        vertex_t h_vertices[]    = {2, 3, 5};
        vertex_t h_in_degrees[]  = {-1, -1, 3, 3, -1, 2};
        vertex_t h_out_degrees[] = {-1, -1, 3, 3, -1, 2};

        return generic_degrees_test(p_handle,
                                    h_src,
                                    h_dst,
                                    h_wgt,
                                    num_vertices,
                                    num_edges,
                                    h_vertices,
                                    num_vertices_to_compute,
                                    HIPGRAPH_TRUE,
                                    HIPGRAPH_TRUE,
                                    HIPGRAPH_FALSE,
                                    HIPGRAPH_TRUE,
                                    h_in_degrees,
                                    h_out_degrees);
    }

    int test_in_degrees_subset(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges               = 8;
        size_t num_vertices            = 6;
        size_t num_vertices_to_compute = 3;

        vertex_t h_src[]        = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]        = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]        = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t h_vertices[]   = {2, 3, 5};
        vertex_t h_in_degrees[] = {-1, -1, 0, 2, -1, 2};

        return generic_degrees_test(p_handle,
                                    h_src,
                                    h_dst,
                                    h_wgt,
                                    num_vertices,
                                    num_edges,
                                    h_vertices,
                                    num_vertices_to_compute,
                                    HIPGRAPH_TRUE,
                                    HIPGRAPH_FALSE,
                                    HIPGRAPH_FALSE,
                                    HIPGRAPH_TRUE,
                                    h_in_degrees,
                                    nullptr);
    }

    int test_out_degrees_subset(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges               = 8;
        size_t num_vertices            = 6;
        size_t num_vertices_to_compute = 3;

        vertex_t h_src[]         = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]         = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]         = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        vertex_t h_vertices[]    = {2, 3, 5};
        vertex_t h_out_degrees[] = {-1, -1, 3, 1, -1, 0};

        return generic_degrees_test(p_handle,
                                    h_src,
                                    h_dst,
                                    h_wgt,
                                    num_vertices,
                                    num_edges,
                                    h_vertices,
                                    num_vertices_to_compute,
                                    HIPGRAPH_FALSE,
                                    HIPGRAPH_TRUE,
                                    HIPGRAPH_FALSE,
                                    HIPGRAPH_TRUE,
                                    nullptr,
                                    h_out_degrees);
    }

} // namespace
