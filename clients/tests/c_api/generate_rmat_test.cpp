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

#include <array>

#include <cinttypes>
#include <cstdio>

#include <gtest/gtest.h>

#include "hipgraph_c/error.h"
#include "hipgraph_c/graph_generators.h"

namespace
{
    using vertex_t = int32_t;
    using edge_t   = int32_t;
    using weight_t = float;

    // ugh.
    template <typename V, typename... T>
    constexpr auto array_of(T&&... t) -> std::array<V, sizeof...(T)>
    {
        return {{std::forward<T>(t)...}};
    }

    /*
   * Simple rmat generator test
   */
    TEST(GeneratorTest, RMAT)
    //test_rmat_generation()
    {
        auto expected_src = array_of<vertex_t>(17,
                                               18,
                                               0,
                                               16,
                                               1,
                                               24,
                                               16,
                                               1,
                                               6,
                                               4,
                                               2,
                                               1,
                                               14,
                                               2,
                                               16,
                                               2,
                                               5,
                                               23,
                                               4,
                                               10,
                                               4,
                                               3,
                                               0,
                                               4,
                                               11,
                                               0,
                                               0,
                                               2,
                                               24,
                                               0);
        auto expected_dst = array_of<vertex_t>(0,
                                               10,
                                               23,
                                               0,
                                               26,
                                               0,
                                               2,
                                               1,
                                               27,
                                               8,
                                               1,
                                               0,
                                               21,
                                               21,
                                               0,
                                               4,
                                               8,
                                               14,
                                               10,
                                               17,
                                               0,
                                               16,
                                               0,
                                               16,
                                               25,
                                               5,
                                               8,
                                               8,
                                               4,
                                               19);

        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_resource_handle_t* p_handle  = nullptr;
        hipgraph_rng_state_t*       rng_state = nullptr;
        hipgraph_coo_t*             coo       = nullptr;

        p_handle = hipgraph_create_resource_handle(nullptr);
        ASSERT_NE(p_handle, nullptr) << "resource handle creation failed";

        ret_code = hipgraph_rng_state_create(p_handle, 0, &rng_state, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "rng_state create failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_generate_rmat_edgelist(p_handle,
                                                   rng_state,
                                                   5,
                                                   30,
                                                   0.57,
                                                   0.19,
                                                   0.19,
                                                   HIPGRAPH_FALSE,
                                                   HIPGRAPH_FALSE,
                                                   &coo,
                                                   &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "generate_rmat_edgelist failed: " << hipgraph_error_message(ret_error);

        hipgraph_type_erased_device_array_view_t* src_view;
        hipgraph_type_erased_device_array_view_t* dst_view;

        src_view = hipgraph_coo_get_sources(coo);
        dst_view = hipgraph_coo_get_destinations(coo);

        size_t src_size = hipgraph_type_erased_device_array_view_size(src_view);

        vertex_t h_src[src_size];
        vertex_t h_dst[src_size];

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_src, src_view, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "src copy_to_host failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
            p_handle, (hipgraph_byte_t*)h_dst, dst_view, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "dst copy_to_host failed: " << hipgraph_error_message(ret_error);

        for(size_t i = 0; i < src_size; ++i)
        {
            EXPECT_EQ(expected_src[i], h_src[i]) << "generated edges don't match at position " << i;
            EXPECT_EQ(expected_dst[i], h_dst[i]) << "generated edges don't match at position " << i;
        }

        hipgraph_type_erased_device_array_view_free(dst_view);
        hipgraph_type_erased_device_array_view_free(src_view);
        hipgraph_coo_free(coo);
        hipgraph_free_resource_handle(p_handle);
        hipgraph_error_free(ret_error);
    }

    TEST(GeneratorTest, RMATList)
    //int test_rmat_list_generation()
    {
        hipgraph_error_code_t ret_code = HIPGRAPH_SUCCESS;
        hipgraph_error_t*     ret_error;

        hipgraph_resource_handle_t* p_handle  = nullptr;
        hipgraph_rng_state_t*       rng_state = nullptr;
        ;
        hipgraph_coo_list_t* coo_list = nullptr;

        p_handle = hipgraph_create_resource_handle(nullptr);
        ASSERT_NE(p_handle, nullptr) << "resource handle creation failed.";

        ret_code = hipgraph_rng_state_create(p_handle, 0, &rng_state, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "rng_state create failed: " << hipgraph_error_message(ret_error);
        //TEST_ALWAYS_ASSERT(ret_code == HIPGRAPH_SUCCESS, hipgraph_error_message(ret_error));

        //
        // NOTE: We can't exactly compare results for functions that make multiple RNG calls
        // within them.  When the RNG state is advanced, it is advanced by a multiple of
        // the number of possible threads involved, not based on how many of the values
        // were actually used.  So different GPU versions will result in subtly different
        // random sequences.
        //
        size_t   num_lists       = 3;
        vertex_t max_vertex_id[] = {32, 16, 32};
        size_t   expected_len[]  = {20, 16, 20};

        ret_code = hipgraph_rng_state_create(p_handle, 0, &rng_state, &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "rng_state create failed: " << hipgraph_error_message(ret_error);

        ret_code = hipgraph_generate_rmat_edgelists(p_handle,
                                                    rng_state,
                                                    num_lists,
                                                    4,
                                                    6,
                                                    4,
                                                    HIPGRAPH_UNIFORM,
                                                    HIPGRAPH_POWER_LAW,
                                                    HIPGRAPH_FALSE,
                                                    HIPGRAPH_FALSE,
                                                    &coo_list,
                                                    &ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "generate_rmat_edgelist failed: " << hipgraph_error_message(ret_error);

        EXPECT_EQ(hipgraph_coo_list_size(coo_list), num_lists)
            << "generated wrong number of results";

        for(size_t i = 0; i < num_lists; i++)
        {
            hipgraph_coo_t* coo = nullptr;

            coo = hipgraph_coo_list_element(coo_list, i);

            hipgraph_type_erased_device_array_view_t* src_view;
            hipgraph_type_erased_device_array_view_t* dst_view;

            src_view = hipgraph_coo_get_sources(coo);
            dst_view = hipgraph_coo_get_destinations(coo);

            size_t src_size = hipgraph_type_erased_device_array_view_size(src_view);

            EXPECT_EQ(src_size, expected_len[i]) << "wrong number of edges";

            vertex_t h_src[src_size];
            vertex_t h_dst[src_size];

            ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (hipgraph_byte_t*)h_src, src_view, &ret_error);
            ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
                << "src copy_to_host failed: " << hipgraph_error_message(ret_error);

            ret_code = hipgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (hipgraph_byte_t*)h_dst, dst_view, &ret_error);
            ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
                << "dst copy_to_host failed: " << hipgraph_error_message(ret_error);

            for(size_t j = 0; j < src_size; ++j)
            {
                EXPECT_LT(h_src[j], max_vertex_id[i])
                    << "generated edges don't match at position " << i;
                EXPECT_LT(h_dst[j], max_vertex_id[i])
                    << "generated edges don't match at position " << i;
            }

            hipgraph_type_erased_device_array_view_free(dst_view);
            hipgraph_type_erased_device_array_view_free(src_view);
        }

        hipgraph_coo_list_free(coo_list);
        hipgraph_free_resource_handle(p_handle);
        hipgraph_error_free(ret_error);
    }
}
