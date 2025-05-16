// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0
/*
 * Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

/*
 * Copyright (C) 2021-2024, NVIDIA CORPORATION.
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

#include "hipgraph_c/resource_handle.h"
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdio>

#include <gtest/gtest.h>

#include "hipgraph_c/error.h"
#include "hipgraph_c/graph.h"

#include "test_utils.h"

//#include <cugraph/utilities/host_scalar_comm.hpp>

namespace hipGRAPH::testing
{

    // Ugh. This is horrible.
    hipgraph_bool_t nearlyEqual(double a, double b, double epsilon)
    {
        using std::abs;
        // FIXME:  There is a better test than this,
        //   perhaps use the gtest comparison for consistency
        //   with C++ and wrap it in a C wrapper.
        return static_cast<hipgraph_bool_t>(abs(a - b)
                                            <= (((abs(a) < abs(b)) ? abs(b) : abs(a)) * epsilon));
    }

    /*
 * Simple check of creating a graph from a COO on device memory.
 */
    void create_test_graph(const hipgraph_resource_handle_t* p_handle,
                           int32_t*                          h_src,
                           int32_t*                          h_dst,
                           float*                            h_wgt,
                           size_t                            num_edges,
                           hipgraph_bool_t                   store_transposed,
                           hipgraph_bool_t                   renumber,
                           hipgraph_bool_t                   is_symmetric,
                           hipgraph_graph_t**                p_graph,
                           hipgraph_error_t**                ret_error)
    {
        hipgraph_error_code_t       ret_code;
        hipgraph_graph_properties_t properties;

        properties.is_symmetric  = is_symmetric;
        properties.is_multigraph = HIPGRAPH_FALSE;

        hipgraph_data_type_id_t vertex_tid = HIPGRAPH_INT32;
        hipgraph_data_type_id_t weight_tid = HIPGRAPH_FLOAT32;

        hipgraph_type_erased_device_array_t*      src;
        hipgraph_type_erased_device_array_t*      dst;
        hipgraph_type_erased_device_array_t*      wgt;
        hipgraph_type_erased_device_array_view_t* src_view;
        hipgraph_type_erased_device_array_view_t* dst_view;
        hipgraph_type_erased_device_array_view_t* wgt_view;

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, vertex_tid, &src, ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "src create failed: " << hipgraph_error_message(*ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, vertex_tid, &dst, ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "dst create failed: " << hipgraph_error_message(*ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, weight_tid, &wgt, ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "wgt create failed: " << hipgraph_error_message(*ret_error);

        src_view = hipgraph_type_erased_device_array_view(src);
        dst_view = hipgraph_type_erased_device_array_view(dst);
        wgt_view = hipgraph_type_erased_device_array_view(wgt);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, src_view, (hipgraph_byte_t*)h_src, ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "src copy_from_host failed: " << hipgraph_error_message(*ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, dst_view, (hipgraph_byte_t*)h_dst, ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "dst copy_from_host failed: " << hipgraph_error_message(*ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, wgt_view, (hipgraph_byte_t*)h_wgt, ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "wgt copy_from_host failed: " << hipgraph_error_message(*ret_error);

        ret_code = hipgraph_sg_graph_create(p_handle,
                                            &properties,
                                            src_view,
                                            dst_view,
                                            wgt_view,
                                            nullptr,
                                            nullptr,
                                            store_transposed,
                                            renumber,
                                            HIPGRAPH_FALSE,
                                            p_graph,
                                            ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "graph creation failed: " << hipgraph_error_message(*ret_error);

        hipgraph_type_erased_device_array_view_free(wgt_view);
        hipgraph_type_erased_device_array_view_free(dst_view);
        hipgraph_type_erased_device_array_view_free(src_view);
        hipgraph_type_erased_device_array_free(wgt);
        hipgraph_type_erased_device_array_free(dst);
        hipgraph_type_erased_device_array_free(src);
    }

    void create_test_graph_double(const hipgraph_resource_handle_t* p_handle,
                                  int32_t*                          h_src,
                                  int32_t*                          h_dst,
                                  double*                           h_wgt,
                                  size_t                            num_edges,
                                  hipgraph_bool_t                   store_transposed,
                                  hipgraph_bool_t                   renumber,
                                  hipgraph_bool_t                   is_symmetric,
                                  hipgraph_graph_t**                p_graph,
                                  hipgraph_error_t**                ret_error)
    {
        hipgraph_error_code_t       ret_code;
        hipgraph_graph_properties_t properties;

        properties.is_symmetric  = is_symmetric;
        properties.is_multigraph = HIPGRAPH_FALSE;

        hipgraph_data_type_id_t vertex_tid = HIPGRAPH_INT32;
        hipgraph_data_type_id_t weight_tid = HIPGRAPH_FLOAT64;

        hipgraph_type_erased_device_array_t*      src;
        hipgraph_type_erased_device_array_t*      dst;
        hipgraph_type_erased_device_array_t*      wgt;
        hipgraph_type_erased_device_array_view_t* src_view;
        hipgraph_type_erased_device_array_view_t* dst_view;
        hipgraph_type_erased_device_array_view_t* wgt_view;

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, vertex_tid, &src, ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "src create failed: " << hipgraph_error_message(*ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, vertex_tid, &dst, ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "dst create failed: " << hipgraph_error_message(*ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, weight_tid, &wgt, ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "wgt create failed: " << hipgraph_error_message(*ret_error);

        src_view = hipgraph_type_erased_device_array_view(src);
        dst_view = hipgraph_type_erased_device_array_view(dst);
        wgt_view = hipgraph_type_erased_device_array_view(wgt);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, src_view, (hipgraph_byte_t*)h_src, ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "src copy_from_host failed: " << hipgraph_error_message(*ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, dst_view, (hipgraph_byte_t*)h_dst, ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "dst copy_from_host failed: " << hipgraph_error_message(*ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, wgt_view, (hipgraph_byte_t*)h_wgt, ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "wgt copy_from_host failed: " << hipgraph_error_message(*ret_error);

        ret_code = hipgraph_sg_graph_create(p_handle,
                                            &properties,
                                            src_view,
                                            dst_view,
                                            wgt_view,
                                            nullptr,
                                            nullptr,
                                            store_transposed,
                                            renumber,
                                            HIPGRAPH_FALSE,
                                            p_graph,
                                            ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "graph creation failed: " << hipgraph_error_message(*ret_error);

        hipgraph_type_erased_device_array_view_free(wgt_view);
        hipgraph_type_erased_device_array_view_free(dst_view);
        hipgraph_type_erased_device_array_view_free(src_view);
        hipgraph_type_erased_device_array_free(wgt);
        hipgraph_type_erased_device_array_free(dst);
        hipgraph_type_erased_device_array_free(src);
    }

    void create_sg_test_graph(const hipgraph_resource_handle_t* p_handle,
                              hipgraph_data_type_id_t           vertex_tid,
                              hipgraph_data_type_id_t           edge_tid,
                              void*                             h_src,
                              void*                             h_dst,
                              hipgraph_data_type_id_t           weight_tid,
                              void*                             h_wgt,
                              hipgraph_data_type_id_t           edge_type_tid,
                              void*                             h_edge_type,
                              hipgraph_data_type_id_t           edge_id_tid,
                              void*                             h_edge_id,
                              size_t                            num_edges,
                              hipgraph_bool_t                   store_transposed,
                              hipgraph_bool_t                   renumber,
                              hipgraph_bool_t                   is_symmetric,
                              hipgraph_bool_t                   is_multigraph,
                              hipgraph_graph_t**                graph,
                              hipgraph_error_t**                ret_error)
    {
        hipgraph_error_code_t       ret_code;
        hipgraph_graph_properties_t properties;

        properties.is_symmetric  = is_symmetric;
        properties.is_multigraph = is_multigraph;

        hipgraph_type_erased_device_array_t*      src            = nullptr;
        hipgraph_type_erased_device_array_t*      dst            = nullptr;
        hipgraph_type_erased_device_array_t*      wgt            = nullptr;
        hipgraph_type_erased_device_array_t*      edge_type      = nullptr;
        hipgraph_type_erased_device_array_t*      edge_id        = nullptr;
        hipgraph_type_erased_device_array_view_t* src_view       = nullptr;
        hipgraph_type_erased_device_array_view_t* dst_view       = nullptr;
        hipgraph_type_erased_device_array_view_t* wgt_view       = nullptr;
        hipgraph_type_erased_device_array_view_t* edge_type_view = nullptr;
        hipgraph_type_erased_device_array_view_t* edge_id_view   = nullptr;

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, vertex_tid, &src, ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "src create failed: " << hipgraph_error_message(*ret_error);

        ret_code = hipgraph_type_erased_device_array_create(
            p_handle, num_edges, vertex_tid, &dst, ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "dst create failed: " << hipgraph_error_message(*ret_error);

        src_view = hipgraph_type_erased_device_array_view(src);
        dst_view = hipgraph_type_erased_device_array_view(dst);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, src_view, (hipgraph_byte_t*)h_src, ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "src copy_from_host failed: " << hipgraph_error_message(*ret_error);

        ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
            p_handle, dst_view, (hipgraph_byte_t*)h_dst, ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "dst copy_from_host failed: " << hipgraph_error_message(*ret_error);

        if(h_wgt != nullptr)
        {
            ret_code = hipgraph_type_erased_device_array_create(
                p_handle, num_edges, weight_tid, &wgt, ret_error);
            ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
                << "wgt create failed: " << hipgraph_error_message(*ret_error);

            wgt_view = hipgraph_type_erased_device_array_view(wgt);

            ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
                p_handle, wgt_view, (hipgraph_byte_t*)h_wgt, ret_error);
            ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
                << "wgt copy_from_host failed: " << hipgraph_error_message(*ret_error);
        }

        if(h_edge_type != nullptr)
        {
            ret_code = hipgraph_type_erased_device_array_create(
                p_handle, num_edges, edge_type_tid, &edge_type, ret_error);
            ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
                << "edge_type create failed: " << hipgraph_error_message(*ret_error);

            edge_type_view = hipgraph_type_erased_device_array_view(edge_type);

            ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
                p_handle, edge_type_view, (hipgraph_byte_t*)h_edge_type, ret_error);
            ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
                << "edge_type copy_from_host failed: " << hipgraph_error_message(*ret_error);
        }

        if(h_edge_id != nullptr)
        {
            ret_code = hipgraph_type_erased_device_array_create(
                p_handle, num_edges, edge_id_tid, &edge_id, ret_error);
            ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
                << "edge_id create failed: " << hipgraph_error_message(*ret_error);

            edge_id_view = hipgraph_type_erased_device_array_view(edge_id);

            ret_code = hipgraph_type_erased_device_array_view_copy_from_host(
                p_handle, edge_id_view, (hipgraph_byte_t*)h_edge_id, ret_error);
            ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
                << "edge_id copy_from_host failed: " << hipgraph_error_message(*ret_error);
        }

        ret_code = hipgraph_sg_graph_create(p_handle,
                                            &properties,
                                            src_view,
                                            dst_view,
                                            wgt_view,
                                            edge_id_view,
                                            edge_type_view,
                                            store_transposed,
                                            renumber,
                                            HIPGRAPH_FALSE,
                                            graph,
                                            ret_error);
        ASSERT_EQ(ret_code, HIPGRAPH_SUCCESS)
            << "graph creation failed: " << hipgraph_error_message(*ret_error);

        if(edge_id != nullptr)
        {
            hipgraph_type_erased_device_array_view_free(edge_id_view);
            hipgraph_type_erased_device_array_free(edge_id);
        }

        if(edge_type != nullptr)
        {
            hipgraph_type_erased_device_array_view_free(edge_type_view);
            hipgraph_type_erased_device_array_free(edge_type);
        }

        if(wgt != nullptr)
        {
            hipgraph_type_erased_device_array_view_free(wgt_view);
            hipgraph_type_erased_device_array_free(wgt);
        }

        hipgraph_type_erased_device_array_view_free(dst_view);
        hipgraph_type_erased_device_array_view_free(src_view);
        hipgraph_type_erased_device_array_free(dst);
        hipgraph_type_erased_device_array_free(src);
    }

#if 0
size_t hipgraph_size_t_allreduce(const hipgraph_resource_handle_t* p_handle, size_t value)
{
  auto internal_handle = reinterpret_cast<hipgraph::c_api::hipgraph_resource_handle_t const *>(p_handle);
  return hipgraph::host_scalar_allreduce(internal_handle->handle_->get_comms(),
                                         value,
                                         raft::comms::op_t::SUM,
                                         internal_handle->handle_->get_stream());
}
#endif

} // namespace
