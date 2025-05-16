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
    int generic_hits_test(const hipgraph_resource_handle_t* p_handle,
                          vertex_t*                         h_src,
                          vertex_t*                         h_dst,
                          weight_t*                         h_wgt,
                          size_t                            num_vertices,
                          size_t                            num_edges,
                          vertex_t*                         h_initial_vertices,
                          weight_t*                         h_initial_hubs,
                          size_t                            num_initial_vertices,
                          weight_t*                         h_result_hubs,
                          weight_t*                         h_result_authorities,
                          hipgraph_bool_t                   store_transposed,
                          hipgraph_bool_t                   normalize,
                          double                            epsilon,
                          size_t                            max_iterations)
    {
        int test_ret_value = 0;

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_graph_t*       p_graph  = nullptr;
        hipgraph_hits_result_t* p_result = nullptr;

        ret_code = create_mg_test_graph(p_handle,
                                        h_src,
                                        h_dst,
                                        h_wgt,
                                        num_edges,
                                        store_transposed,
                                        HIPGRAPH_FALSE,
                                        &p_graph,
                                        &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "create_mg_test_graph failed. " << hipgraph_error_message(ret_error);

        if(h_initial_vertices == nullptr)
        {
            ret_code = hipgraph_hits(p_handle,
                                     p_graph,
                                     epsilon,
                                     max_iterations,
                                     nullptr,
                                     nullptr,
                                     normalize,
                                     HIPGRAPH_FALSE,
                                     &p_result,
                                     &ret_error);
        }
        else
        {
            int rank = hipgraph_resource_handle_get_rank(p_handle);

            if(rank != 0)
            {
                // Only initialize the vertices on rank 0
                num_initial_vertices = 0;
            }

            hipgraph_type_erased_device_array_t*      initial_vertices;
            hipgraph_type_erased_device_array_t*      initial_hubs;
            hipgraph_type_erased_device_array_view_t* initial_vertices_view;
            hipgraph_type_erased_device_array_view_t* initial_hubs_view;

            ret_code = hipgraph_type_erased_device_array_create(
                p_handle, num_initial_vertices, HIPGRAPH_INT32, &initial_vertices, &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "initial_vertices create failed.";

            ret_code = hipgraph_type_erased_device_array_create(
                p_handle, num_initial_vertices, HIPGRAPH_FLOAT32, &initial_hubs, &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "initial_hubs create failed.";

            initial_vertices_view = hipgraph_type_erased_device_array_view(initial_vertices);
            initial_hubs_view     = hipgraph_type_erased_device_array_view(initial_hubs);

            ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
                p_handle, initial_vertices_view, (hipgraph_byte_t*)h_initial_vertices, &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "src copy_from_host failed.";

            ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
                p_handle, initial_hubs_view, (hipgraph_byte_t*)h_initial_hubs, &ret_error);
            EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "src copy_from_host failed.";

            ret_code = hipgraph_hits(p_handle,
                                     p_graph,
                                     epsilon,
                                     max_iterations,
                                     initial_vertices_view,
                                     initial_hubs_view,
                                     normalize,
                                     HIPGRAPH_FALSE,
                                     &p_result,
                                     &ret_error);
        }

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "hipgraph_hits failed.";

        // NOTE: Because we get back vertex ids, hubs and authorities, we can
        //       simply compare the returned values with the expected results
        //       for the entire graph.  Each GPU will have a subset of the
        //       total vertices, so they will do a subset of the comparisons.
        hipgraph_type_erased_device_array_view_t* vertices;
        hipgraph_type_erased_device_array_view_t* hubs;
        hipgraph_type_erased_device_array_view_t* authorities;

        vertices                 = hipgraph_hits_result_get_vertices(p_result);
        hubs                     = hipgraph_hits_result_get_hubs(p_result);
        authorities              = hipgraph_hits_result_get_authorities(p_result);
        double score_differences = hipgraph_hits_result_get_hub_score_differences(p_result);
        size_t num_iterations    = hipgraph_hits_result_get_number_of_iterations(p_result);

        vertex_t h_vertices[num_vertices];
        weight_t h_hubs[num_vertices];
        weight_t h_authorities[num_vertices];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_vertices, vertices, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_hubs, hubs, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_authorities, authorities, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        size_t num_local_vertices = hipgraph_type_erased_device_array_view_size(vertices);

        for(int i = 0; (i < num_local_vertices) && (test_ret_value == 0); ++i)
        {
            EXPECT_NEAR(h_result_hubs[h_vertices[i]], h_hubs[i], 0.001)
                << "hubs results don't match";
            EXPECT_NEAR(h_result_authorities[h_vertices[i]], h_authorities[i], 0.001)
                << "authorities results don't match";
        }

        hipgraph_hits_result_free(p_result);
        hipgraph_mg_graph_free(p_graph);
        hipgraph_error_free(ret_error);

        return test_ret_value;
    }

    int test_hits(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t h_src[]         = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]         = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]         = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        weight_t h_hubs[]        = {0.347296, 0.532089, 1, 0.00003608, 0.00003608, 0};
        weight_t h_authorities[] = {0.652703, 0.879385, 0, 1, 0.347296, 0.00009136};

        double epsilon        = 0.00002;
        size_t max_iterations = 20;

        // hits wants store_transposed = HIPGRAPH_TRUE
        return generic_hits_test(p_handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 num_vertices,
                                 num_edges,
                                 nullptr,
                                 nullptr,
                                 0,
                                 h_hubs,
                                 h_authorities,
                                 HIPGRAPH_TRUE,
                                 HIPGRAPH_FALSE,
                                 epsilon,
                                 max_iterations);
    }

    int test_hits_with_transpose(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t h_src[]         = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]         = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]         = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        weight_t h_hubs[]        = {0.347296, 0.532089, 1, 0.00003608, 0.00003608, 0};
        weight_t h_authorities[] = {0.652703, 0.879385, 0, 1, 0.347296, 0.00009136};

        double epsilon        = 0.00002;
        size_t max_iterations = 20;

        // Hits wants store_transposed = HIPGRAPH_TRUE
        //    This call will force hipgraph_hits to transpose the graph
        //    But we're passing src/dst backwards so the results will be the same
        return generic_hits_test(p_handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 num_vertices,
                                 num_edges,
                                 nullptr,
                                 nullptr,
                                 0,
                                 h_hubs,
                                 h_authorities,
                                 HIPGRAPH_FALSE,
                                 HIPGRAPH_FALSE,
                                 epsilon,
                                 max_iterations);
    }

    int test_hits_with_initial(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges        = 8;
        size_t num_vertices     = 6;
        size_t num_initial_hubs = 5;

        vertex_t h_src[]              = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]              = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]              = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        weight_t h_hubs[]             = {0.347296, 0.532089, 1, 0.00000959, 0.00000959, 0};
        weight_t h_authorities[]      = {0.652704, 0.879385, 0, 1, 0.347296, 0.00002428};
        vertex_t h_initial_vertices[] = {0, 1, 2, 3, 4};
        weight_t h_initial_hubs[]     = {0.347296, 0.532089, 1, 0.00003608, 0.00003608};

        double epsilon        = 0.0001;
        size_t max_iterations = 20;

        // Hits wants store_transposed = HIPGRAPH_TRUE
        //    This call will force hipgraph_hits to transpose the graph
        //    But we're passing src/dst backwards so the results will be the same
        return generic_hits_test(p_handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 num_vertices,
                                 num_edges,
                                 h_initial_vertices,
                                 h_initial_hubs,
                                 num_initial_hubs,
                                 h_hubs,
                                 h_authorities,
                                 HIPGRAPH_FALSE,
                                 HIPGRAPH_FALSE,
                                 epsilon,
                                 max_iterations);
    }

    int test_hits_bigger(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 48;
        size_t num_vertices = 54;

        vertex_t h_src[] = {29, 45, 6,  8,  16, 45, 8,  16, 6,  38, 45, 45, 48, 45, 45, 45,
                            45, 48, 53, 45, 6,  45, 38, 45, 38, 45, 16, 45, 38, 16, 45, 45,
                            38, 6,  38, 45, 45, 45, 16, 38, 6,  45, 29, 45, 29, 6,  38, 6};
        vertex_t h_dst[] = {45, 45, 16, 45, 6,  45, 45, 16, 45, 38, 45, 6,  45, 38, 16, 45,
                            45, 45, 45, 53, 29, 16, 45, 8,  8,  16, 45, 38, 45, 6,  45, 45,
                            6,  6,  16, 38, 16, 45, 45, 6,  16, 6,  53, 16, 38, 45, 45, 16};
        weight_t h_wgt[]
            = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

        weight_t h_hubs[]
            = {0, 0,        0,        0, 0, 0, 0.323569, 0, 0.156401, 0, 0,        0,        0, 0,
               0, 0,        0.253312, 0, 0, 0, 0,        0, 0,        0, 0,        0,        0, 0,
               0, 0.110617, 0,        0, 0, 0, 0,        0, 0,        0, 0.365733, 0,        0, 0,
               0, 0,        0,        1, 0, 0, 0.156401, 0, 0,        0, 0,        0.0782005};
        weight_t h_authorities[]
            = {0, 0,         0,        0, 0, 0, 0.321874, 0, 0.123424, 0, 0,        0,       0, 0,
               0, 0,         0.595522, 0, 0, 0, 0,        0, 0,        0, 0,        0,       0, 0,
               0, 0.0292397, 0,        0, 0, 0, 0,        0, 0,        0, 0.314164, 0,       0, 0,
               0, 0,         0,        1, 0, 0, 0,        0, 0,        0, 0,        0.100368};

        double epsilon        = 0.0001;
        size_t max_iterations = 20;

        // Hits wants store_transposed = HIPGRAPH_TRUE
        //    This call will force hipgraph_hits to transpose the graph
        //    But we're passing src/dst backwards so the results will be the same
        return generic_hits_test(p_handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 num_vertices,
                                 num_edges,
                                 nullptr,
                                 nullptr,
                                 0,
                                 h_hubs,
                                 h_authorities,
                                 HIPGRAPH_FALSE,
                                 HIPGRAPH_FALSE,
                                 epsilon,
                                 max_iterations);
    }

    int test_hits_bigger_normalized(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 48;
        size_t num_vertices = 54;

        vertex_t h_src[] = {29, 45, 6,  8,  16, 45, 8,  16, 6,  38, 45, 45, 48, 45, 45, 45,
                            45, 48, 53, 45, 6,  45, 38, 45, 38, 45, 16, 45, 38, 16, 45, 45,
                            38, 6,  38, 45, 45, 45, 16, 38, 6,  45, 29, 45, 29, 6,  38, 6};
        vertex_t h_dst[] = {45, 45, 16, 45, 6,  45, 45, 16, 45, 38, 45, 6,  45, 38, 16, 45,
                            45, 45, 45, 53, 29, 16, 45, 8,  8,  16, 45, 38, 45, 6,  45, 45,
                            6,  6,  16, 38, 16, 45, 45, 6,  16, 6,  53, 16, 38, 45, 45, 16};
        weight_t h_wgt[]
            = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

        weight_t h_hubs[]
            = {0,         0, 0, 0, 0, 0,        0.132381, 0, 0.0639876, 0, 0, 0, 0, 0,         0, 0,
               0.103637,  0, 0, 0, 0, 0,        0,        0, 0,         0, 0, 0, 0, 0.0452563, 0, 0,
               0,         0, 0, 0, 0, 0,        0.149631, 0, 0,         0, 0, 0, 0, 0.409126,  0, 0,
               0.0639876, 0, 0, 0, 0, 0.0319938};

        weight_t h_authorities[]
            = {0,        0,        0, 0, 0, 0, 0.129548, 0, 0.0496755, 0, 0, 0, 0, 0, 0,
               0,        0.239688, 0, 0, 0, 0, 0,        0, 0,         0, 0, 0, 0, 0, 0.0117691,
               0,        0,        0, 0, 0, 0, 0,        0, 0.126445,  0, 0, 0, 0, 0, 0,
               0.402479, 0,        0, 0, 0, 0, 0,        0, 0.0403963};

        double epsilon        = 0.000001;
        size_t max_iterations = 100;

        return generic_hits_test(p_handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 num_vertices,
                                 num_edges,
                                 nullptr,
                                 nullptr,
                                 0,
                                 h_hubs,
                                 h_authorities,
                                 HIPGRAPH_FALSE,
                                 HIPGRAPH_TRUE,
                                 epsilon,
                                 max_iterations);
    }

    int test_hits_bigger_unnormalized(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 48;
        size_t num_vertices = 54;

        vertex_t h_src[] = {29, 45, 6,  8,  16, 45, 8,  16, 6,  38, 45, 45, 48, 45, 45, 45,
                            45, 48, 53, 45, 6,  45, 38, 45, 38, 45, 16, 45, 38, 16, 45, 45,
                            38, 6,  38, 45, 45, 45, 16, 38, 6,  45, 29, 45, 29, 6,  38, 6};
        vertex_t h_dst[] = {45, 45, 16, 45, 6,  45, 45, 16, 45, 38, 45, 6,  45, 38, 16, 45,
                            45, 45, 45, 53, 29, 16, 45, 8,  8,  16, 45, 38, 45, 6,  45, 45,
                            6,  6,  16, 38, 16, 45, 45, 6,  16, 6,  53, 16, 38, 45, 45, 16};
        weight_t h_wgt[]
            = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

        weight_t h_hubs[]
            = {0, 0,        0,        0, 0, 0, 0.323569, 0, 0.156401, 0, 0,        0,        0, 0,
               0, 0,        0.253312, 0, 0, 0, 0,        0, 0,        0, 0,        0,        0, 0,
               0, 0.110617, 0,        0, 0, 0, 0,        0, 0,        0, 0.365733, 0,        0, 0,
               0, 0,        0,        1, 0, 0, 0.156401, 0, 0,        0, 0,        0.0782005};
        weight_t h_authorities[]
            = {0, 0,         0,        0, 0, 0, 0.321874, 0, 0.123424, 0, 0,        0,       0, 0,
               0, 0,         0.595522, 0, 0, 0, 0,        0, 0,        0, 0,        0,       0, 0,
               0, 0.0292397, 0,        0, 0, 0, 0,        0, 0,        0, 0.314164, 0,       0, 0,
               0, 0,         0,        1, 0, 0, 0,        0, 0,        0, 0,        0.100368};

        double epsilon        = 0.000001;
        size_t max_iterations = 100;

        return generic_hits_test(p_handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 num_vertices,
                                 num_edges,
                                 nullptr,
                                 nullptr,
                                 0,
                                 h_hubs,
                                 h_authorities,
                                 HIPGRAPH_FALSE,
                                 HIPGRAPH_FALSE,
                                 epsilon,
                                 max_iterations);
    }

} // namespace
