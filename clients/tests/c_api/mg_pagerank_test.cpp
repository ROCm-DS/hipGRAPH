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
    int generic_pagerank_test(const hipgraph_resource_handle_t* p_handle,
                              vertex_t*                         h_src,
                              vertex_t*                         h_dst,
                              weight_t*                         h_wgt,
                              weight_t*                         h_result,
                              size_t                            num_vertices,
                              size_t                            num_edges,
                              hipgraph_bool_t                   store_transposed,
                              double                            alpha,
                              double                            epsilon,
                              size_t                            max_iterations)
    {
        int test_ret_value = 0;

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_graph_t*             p_graph  = nullptr;
        hipgraph_centrality_result_t* p_result = nullptr;

        ret_code = create_mg_test_graph(p_handle,
                                        h_src,
                                        h_dst,
                                        h_wgt,
                                        num_edges,
                                        store_transposed,
                                        HIPGRAPH_FALSE,
                                        &p_graph,
                                        &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "create_mg_test_graph failed.";

        ret_code = hipgraph_pagerank(p_handle,
                                     p_graph,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     alpha,
                                     epsilon,
                                     max_iterations,
                                     HIPGRAPH_FALSE,
                                     &p_result,
                                     &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "hipgraph_pagerank failed.";

        // NOTE: Because we get back vertex ids and pageranks, we can simply compare
        //       the returned values with the expected results for the entire
        //       graph.  Each GPU will have a subset of the total vertices, so
        //       they will do a subset of the comparisons.
        hipgraph_type_erased_device_array_view_t* vertices;
        hipgraph_type_erased_device_array_view_t* pageranks;

        vertices  = hipgraph_centrality_result_get_vertices(p_result);
        pageranks = hipgraph_centrality_result_get_values(p_result);

        size_t num_local_vertices = hipgraph_type_erased_device_array_view_size(vertices);

        vertex_t h_vertices[num_local_vertices];
        weight_t h_pageranks[num_local_vertices];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_vertices, vertices, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_pageranks, pageranks, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        for(int i = 0; (i < num_local_vertices) && (test_ret_value == 0); ++i)
        {
            EXPECT_NEAR(h_result[h_vertices[i]], h_pageranks[i], 0.001)
                << "pagerank results don't match";
        }

        hipgraph_centrality_result_free(p_result);
        hipgraph_mg_graph_free(p_graph);
        hipgraph_error_free(ret_error);

        return test_ret_value;
    }

    int generic_pagerank_nonconverging_test(const hipgraph_resource_handle_t* p_handle,
                                            vertex_t*                         h_src,
                                            vertex_t*                         h_dst,
                                            weight_t*                         h_wgt,
                                            weight_t*                         h_result,
                                            size_t                            num_vertices,
                                            size_t                            num_edges,
                                            hipgraph_bool_t                   store_transposed,
                                            double                            alpha,
                                            double                            epsilon,
                                            size_t                            max_iterations)
    {
        int test_ret_value = 0;

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_graph_t*             p_graph  = nullptr;
        hipgraph_centrality_result_t* p_result = nullptr;

        ret_code = create_mg_test_graph(p_handle,
                                        h_src,
                                        h_dst,
                                        h_wgt,
                                        num_edges,
                                        store_transposed,
                                        HIPGRAPH_FALSE,
                                        &p_graph,
                                        &ret_error);

        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "create_mg_test_graph failed.";

        ret_code = hipgraph_pagerank_allow_nonconvergence(p_handle,
                                                          p_graph,
                                                          nullptr,
                                                          nullptr,
                                                          nullptr,
                                                          nullptr,
                                                          alpha,
                                                          epsilon,
                                                          max_iterations,
                                                          HIPGRAPH_FALSE,
                                                          &p_result,
                                                          &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "hipgraph_pagerank failed.";

        // NOTE: Because we get back vertex ids and pageranks, we can simply compare
        //       the returned values with the expected results for the entire
        //       graph.  Each GPU will have a subset of the total vertices, so
        //       they will do a subset of the comparisons.
        hipgraph_type_erased_device_array_view_t* vertices;
        hipgraph_type_erased_device_array_view_t* pageranks;

        vertices  = hipgraph_centrality_result_get_vertices(p_result);
        pageranks = hipgraph_centrality_result_get_values(p_result);

        size_t num_local_vertices = hipgraph_type_erased_device_array_view_size(vertices);

        vertex_t h_vertices[num_local_vertices];
        weight_t h_pageranks[num_local_vertices];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_vertices, vertices, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_pageranks, pageranks, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        for(int i = 0; (i < num_local_vertices) && (test_ret_value == 0); ++i)
        {
            EXPECT_NEAR(h_result[h_vertices[i]], h_pageranks[i], 0.001)
                << "pagerank results don't match";
        }

        hipgraph_centrality_result_free(p_result);
        hipgraph_mg_graph_free(p_graph);
        hipgraph_error_free(ret_error);

        return test_ret_value;
    }

    int generic_personalized_pagerank_test(const hipgraph_resource_handle_t* p_handle,
                                           vertex_t*                         h_src,
                                           vertex_t*                         h_dst,
                                           weight_t*                         h_wgt,
                                           weight_t*                         h_result,
                                           vertex_t*       h_personalization_vertices,
                                           weight_t*       h_personalization_values,
                                           size_t          num_vertices,
                                           size_t          num_edges,
                                           size_t          num_personalization_vertices,
                                           hipgraph_bool_t store_transposed,
                                           double          alpha,
                                           double          epsilon,
                                           size_t          max_iterations)
    {
        int test_ret_value = 0;

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_graph_t*                         p_graph                       = nullptr;
        hipgraph_centrality_result_t*             p_result                      = nullptr;
        hipgraph_type_erased_device_array_t*      personalization_vertices      = nullptr;
        hipgraph_type_erased_device_array_t*      personalization_values        = nullptr;
        hipgraph_type_erased_device_array_view_t* personalization_vertices_view = nullptr;
        hipgraph_type_erased_device_array_view_t* personalization_values_view   = nullptr;

        hipgraph_data_type_id_t vertex_tid = HIPGRAPH_INT32;
        hipgraph_data_type_id_t weight_tid = HIPGRAPH_FLOAT32;

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
            << "create_test_graph failed. " << hipgraph_error_message(ret_error);

        if(hipgraph_resource_handle_get_rank(p_handle) != 0)
        {
            num_personalization_vertices = 0;
        }

        ret_code = hipgraph_type_erased_device_array_create(p_handle,
                                                            num_personalization_vertices,
                                                            vertex_tid,
                                                            &personalization_vertices,
                                                            &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "personalization_vertices create failed.";

        ret_code = hipgraph_type_erased_device_array_create(p_handle,
                                                            num_personalization_vertices,
                                                            weight_tid,
                                                            &personalization_values,
                                                            &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "personalization_values create failed.";

        personalization_vertices_view
            = hipgraph_type_erased_device_array_view(personalization_vertices);
        personalization_values_view
            = hipgraph_type_erased_device_array_view(personalization_values);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle,
            personalization_vertices_view,
            (hipgraph_byte_t*)h_personalization_vertices,
            &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "personalization_vertices copy_from_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle,
            personalization_values_view,
            (hipgraph_byte_t*)h_personalization_values,
            &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "personalization_values copy_from_host failed.";

        ret_code = hipgraph_personalized_pagerank(p_handle,
                                                  p_graph,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  personalization_vertices_view,
                                                  personalization_values_view,
                                                  alpha,
                                                  epsilon,
                                                  max_iterations,
                                                  HIPGRAPH_FALSE,
                                                  &p_result,
                                                  &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "hipgraph_personalized_pagerank failed. "
                                              << "hipgraph_personalized_pagerank failed.";

        hipgraph_type_erased_device_array_view_t* vertices;
        hipgraph_type_erased_device_array_view_t* pageranks;

        vertices  = hipgraph_centrality_result_get_vertices(p_result);
        pageranks = hipgraph_centrality_result_get_values(p_result);

        size_t num_local_vertices = hipgraph_type_erased_device_array_view_size(vertices);

        vertex_t h_vertices[num_local_vertices];
        weight_t h_pageranks[num_local_vertices];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_vertices, vertices, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_pageranks, pageranks, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        for(int i = 0; (i < num_local_vertices) && (test_ret_value == 0); ++i)
        {
            EXPECT_NEAR(h_result[h_vertices[i]], h_pageranks[i], 0.001)
                << "pagerank results don't match";
        }

        hipgraph_centrality_result_free(p_result);
        hipgraph_mg_graph_free(p_graph);
        hipgraph_error_free(ret_error);

        return test_ret_value;
    }

    int generic_personalized_pagerank_nonconverging_test(const hipgraph_resource_handle_t* p_handle,
                                                         vertex_t*                         h_src,
                                                         vertex_t*                         h_dst,
                                                         weight_t*                         h_wgt,
                                                         weight_t*                         h_result,
                                                         vertex_t* h_personalization_vertices,
                                                         weight_t* h_personalization_values,
                                                         size_t    num_vertices,
                                                         size_t    num_edges,
                                                         size_t    num_personalization_vertices,
                                                         hipgraph_bool_t store_transposed,
                                                         double          alpha,
                                                         double          epsilon,
                                                         size_t          max_iterations)
    {
        int test_ret_value = 0;

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_graph_t*                         p_graph                       = nullptr;
        hipgraph_centrality_result_t*             p_result                      = nullptr;
        hipgraph_type_erased_device_array_t*      personalization_vertices      = nullptr;
        hipgraph_type_erased_device_array_t*      personalization_values        = nullptr;
        hipgraph_type_erased_device_array_view_t* personalization_vertices_view = nullptr;
        hipgraph_type_erased_device_array_view_t* personalization_values_view   = nullptr;

        hipgraph_data_type_id_t vertex_tid = HIPGRAPH_INT32;
        hipgraph_data_type_id_t weight_tid = HIPGRAPH_FLOAT32;

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
            << "create_test_graph failed. " << hipgraph_error_message(ret_error);

        if(hipgraph_resource_handle_get_rank(p_handle) != 0)
        {
            num_personalization_vertices = 0;
        }

        ret_code = hipgraph_type_erased_device_array_create(p_handle,
                                                            num_personalization_vertices,
                                                            vertex_tid,
                                                            &personalization_vertices,
                                                            &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "personalization_vertices create failed.";

        ret_code = hipgraph_type_erased_device_array_create(p_handle,
                                                            num_personalization_vertices,
                                                            weight_tid,
                                                            &personalization_values,
                                                            &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "personalization_values create failed.";

        personalization_vertices_view
            = hipgraph_type_erased_device_array_view(personalization_vertices);
        personalization_values_view
            = hipgraph_type_erased_device_array_view(personalization_values);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle,
            personalization_vertices_view,
            (hipgraph_byte_t*)h_personalization_vertices,
            &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "personalization_vertices copy_from_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle,
            personalization_values_view,
            (hipgraph_byte_t*)h_personalization_values,
            &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "personalization_values copy_from_host failed.";

        ret_code
            = hipgraph_personalized_pagerank_allow_nonconvergence(p_handle,
                                                                  p_graph,
                                                                  nullptr,
                                                                  nullptr,
                                                                  nullptr,
                                                                  nullptr,
                                                                  personalization_vertices_view,
                                                                  personalization_values_view,
                                                                  alpha,
                                                                  epsilon,
                                                                  max_iterations,
                                                                  HIPGRAPH_FALSE,
                                                                  &p_result,
                                                                  &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "hipgraph_personalized_pagerank failed. "
                                              << "hipgraph_personalized_pagerank failed.";

        hipgraph_type_erased_device_array_view_t* vertices;
        hipgraph_type_erased_device_array_view_t* pageranks;

        vertices  = hipgraph_centrality_result_get_vertices(p_result);
        pageranks = hipgraph_centrality_result_get_values(p_result);

        size_t num_local_vertices = hipgraph_type_erased_device_array_view_size(vertices);

        vertex_t h_vertices[num_local_vertices];
        weight_t h_pageranks[num_local_vertices];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_vertices, vertices, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_pageranks, pageranks, &ret_error);
        EXPECT_EQ(ret_code, HIPGRAPH_SUCCESS) << "copy_to_host failed.";

        for(int i = 0; (i < num_local_vertices) && (test_ret_value == 0); ++i)
        {
            EXPECT_NEAR(h_result[h_vertices[i]], h_pageranks[i], 0.001)
                << "pagerank results don't match";
        }

        hipgraph_centrality_result_free(p_result);
        hipgraph_mg_graph_free(p_graph);
        hipgraph_error_free(ret_error);

        return test_ret_value;
    }

    int test_pagerank(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        weight_t h_result[] = {0.0915528, 0.168382, 0.0656831, 0.191468, 0.120677, 0.362237};

        double alpha          = 0.95;
        double epsilon        = 0.0001;
        size_t max_iterations = 20;

        // Pagerank wants store_transposed = HIPGRAPH_TRUE
        return generic_pagerank_test(p_handle,
                                     h_src,
                                     h_dst,
                                     h_wgt,
                                     h_result,
                                     num_vertices,
                                     num_edges,
                                     HIPGRAPH_TRUE,
                                     alpha,
                                     epsilon,
                                     max_iterations);
    }

    int test_pagerank_with_transpose(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        weight_t h_result[] = {0.0915528, 0.168382, 0.0656831, 0.191468, 0.120677, 0.362237};

        double alpha          = 0.95;
        double epsilon        = 0.0001;
        size_t max_iterations = 20;

        // Pagerank wants store_transposed = HIPGRAPH_TRUE
        //    This call will force hipgraph_pagerank to transpose the graph
        //    But we're passing src/dst backwards so the results will be the same
        return generic_pagerank_test(p_handle,
                                     h_src,
                                     h_dst,
                                     h_wgt,
                                     h_result,
                                     num_vertices,
                                     num_edges,
                                     HIPGRAPH_FALSE,
                                     alpha,
                                     epsilon,
                                     max_iterations);
    }

    int test_pagerank_4(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 3;
        size_t num_vertices = 4;

        vertex_t h_src[]    = {0, 1, 2};
        vertex_t h_dst[]    = {1, 2, 3};
        weight_t h_wgt[]    = {1.f, 1.f, 1.f};
        weight_t h_result[] = {
            0.11615584790706635f, 0.21488840878009796f, 0.29881080985069275f, 0.37014490365982056f};

        double alpha          = 0.85;
        double epsilon        = 1.0e-6;
        size_t max_iterations = 500;

        return generic_pagerank_test(p_handle,
                                     h_src,
                                     h_dst,
                                     h_wgt,
                                     h_result,
                                     num_vertices,
                                     num_edges,
                                     HIPGRAPH_FALSE,
                                     alpha,
                                     epsilon,
                                     max_iterations);
    }

    int test_pagerank_4_with_transpose(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 3;
        size_t num_vertices = 4;

        vertex_t h_src[]    = {0, 1, 2};
        vertex_t h_dst[]    = {1, 2, 3};
        weight_t h_wgt[]    = {1.f, 1.f, 1.f};
        weight_t h_result[] = {
            0.11615584790706635f, 0.21488840878009796f, 0.29881080985069275f, 0.37014490365982056f};

        double alpha          = 0.85;
        double epsilon        = 1.0e-6;
        size_t max_iterations = 500;

        return generic_pagerank_test(p_handle,
                                     h_src,
                                     h_dst,
                                     h_wgt,
                                     h_result,
                                     num_vertices,
                                     num_edges,
                                     HIPGRAPH_TRUE,
                                     alpha,
                                     epsilon,
                                     max_iterations);
    }

    int test_pagerank_non_convergence(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 8;
        size_t num_vertices = 6;

        vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4};
        vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5};
        weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
        weight_t h_result[] = {0.0776471, 0.167637, 0.0639699, 0.220202, 0.140046, 0.330498};

        double alpha          = 0.95;
        double epsilon        = 0.0001;
        size_t max_iterations = 2;

        // Pagerank wants store_transposed = HIPGRAPH_TRUE
        return generic_pagerank_nonconverging_test(p_handle,
                                                   h_src,
                                                   h_dst,
                                                   h_wgt,
                                                   h_result,
                                                   num_vertices,
                                                   num_edges,
                                                   HIPGRAPH_TRUE,
                                                   alpha,
                                                   epsilon,
                                                   max_iterations);
    }

    int test_personalized_pagerank(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 3;
        size_t num_vertices = 4;

        vertex_t h_src[]    = {0, 1, 2};
        vertex_t h_dst[]    = {1, 2, 3};
        weight_t h_wgt[]    = {1.f, 1.f, 1.f};
        weight_t h_result[] = {0.0559233f, 0.159381f, 0.303244f, 0.481451f};

        vertex_t h_personalized_vertices[] = {0, 1, 2, 3};
        weight_t h_personalized_values[]   = {0.1, 0.2, 0.3, 0.4};

        double alpha          = 0.85;
        double epsilon        = 1.0e-6;
        size_t max_iterations = 500;

        return generic_personalized_pagerank_test(p_handle,
                                                  h_src,
                                                  h_dst,
                                                  h_wgt,
                                                  h_result,
                                                  h_personalized_vertices,
                                                  h_personalized_values,
                                                  num_vertices,
                                                  num_edges,
                                                  num_vertices,
                                                  HIPGRAPH_FALSE,
                                                  alpha,
                                                  epsilon,
                                                  max_iterations);
    }

    int test_personalized_pagerank_non_convergence(const hipgraph_resource_handle_t* p_handle)
    {
        size_t num_edges    = 3;
        size_t num_vertices = 4;

        vertex_t h_src[]    = {0, 1, 2};
        vertex_t h_dst[]    = {1, 2, 3};
        weight_t h_wgt[]    = {1.f, 1.f, 1.f};
        weight_t h_result[] = {0.03625, 0.285, 0.32125, 0.3575};

        vertex_t h_personalized_vertices[] = {0, 1, 2, 3};
        weight_t h_personalized_values[]   = {0.1, 0.2, 0.3, 0.4};

        double alpha          = 0.85;
        double epsilon        = 1.0e-6;
        size_t max_iterations = 1;

        return generic_personalized_pagerank_nonconverging_test(p_handle,
                                                                h_src,
                                                                h_dst,
                                                                h_wgt,
                                                                h_result,
                                                                h_personalized_vertices,
                                                                h_personalized_values,
                                                                num_vertices,
                                                                num_edges,
                                                                num_vertices,
                                                                HIPGRAPH_FALSE,
                                                                alpha,
                                                                epsilon,
                                                                max_iterations);
    }

} // namespace
