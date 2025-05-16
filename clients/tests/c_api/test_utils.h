// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "hipgraph_c/error.h"
#include "hipgraph_c/graph.h"
#include "hipgraph_c/resource_handle.h"

#include <gtest/gtest.h>

namespace hipGRAPH
{
    namespace testing
    {
        hipgraph_bool_t nearlyEqual(double a, double b, double epsilon);

        void create_test_graph(const hipgraph_resource_handle_t* handle,
                               int32_t*                          h_src,
                               int32_t*                          h_dst,
                               float*                            h_wgt,
                               size_t                            num_edges,
                               hipgraph_bool_t                   store_transposed,
                               hipgraph_bool_t                   renumber,
                               hipgraph_bool_t                   is_symmetric,
                               hipgraph_graph_t**                p_graph,
                               hipgraph_error_t**                ret_error);

        void create_test_graph_double(const hipgraph_resource_handle_t* handle,
                                      int32_t*                          h_src,
                                      int32_t*                          h_dst,
                                      double*                           h_wgt,
                                      size_t                            num_edges,
                                      hipgraph_bool_t                   store_transposed,
                                      hipgraph_bool_t                   renumber,
                                      hipgraph_bool_t                   is_symmetric,
                                      hipgraph_graph_t**                p_graph,
                                      hipgraph_error_t**                ret_error);

        void create_sg_test_graph(const hipgraph_resource_handle_t* handle,
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
                                  hipgraph_error_t**                ret_error);

#if 0
        size_t hipgraph_size_t_allreduce(const hipgraph_resource_handle_t* handle, size_t value);
#endif
    } // namespace testing
} // namespace hipGRAPH
