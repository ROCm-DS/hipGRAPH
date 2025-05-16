// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*! \file */
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

#include "common.h"
#include <cugraph_c/labeling_algorithms.h>
#include "hipgraph/hipgraph_c/labeling_algorithms.h"

/*
 hipgraph_labeling_result_get_vertices
*/

hipgraph_type_erased_device_array_view_t*
    hipgraph_labeling_result_get_vertices(hipgraph_labeling_result_t* result)
{

    cugraph_type_erased_device_array_view_t* view;

    view = cugraph_labeling_result_get_vertices((cugraph_labeling_result_t*)result);

    return (hipgraph_type_erased_device_array_view_t*)view;
}

/*
 hipgraph_labeling_result_get_labels
*/

hipgraph_type_erased_device_array_view_t*
    hipgraph_labeling_result_get_labels(hipgraph_labeling_result_t* result)
{

    cugraph_type_erased_device_array_view_t* view;

    view = cugraph_labeling_result_get_labels((cugraph_labeling_result_t*)result);

    return (hipgraph_type_erased_device_array_view_t*)view;
}

/*
 hipgraph_labeling_result_free
*/

void hipgraph_labeling_result_free(hipgraph_labeling_result_t* result)
{
    cugraph_labeling_result_free((cugraph_labeling_result_t*)result);
}

/*
 hipgraph_weakly_connected_components
*/

hipgraph_error_code_t hipgraph_weakly_connected_components(const hipgraph_resource_handle_t* handle,
                                                           hipgraph_graph_t*                 graph,
                                                           hipgraph_bool_t do_expensive_check,
                                                           hipgraph_labeling_result_t** result,
                                                           hipgraph_error_t**           error)
{

    cugraph_error_code_t out;

    out = cugraph_weakly_connected_components((const cugraph_resource_handle_t*)handle,
                                              (cugraph_graph_t*)graph,
                                              (bool_t)do_expensive_check,
                                              (cugraph_labeling_result_t**)result,
                                              (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_strongly_connected_components
*/

hipgraph_error_code_t
    hipgraph_strongly_connected_components(const hipgraph_resource_handle_t* handle,
                                           hipgraph_graph_t*                 graph,
                                           hipgraph_bool_t                   do_expensive_check,
                                           hipgraph_labeling_result_t**      result,
                                           hipgraph_error_t**                error)
{

    cugraph_error_code_t out;

    out = cugraph_strongly_connected_components((const cugraph_resource_handle_t*)handle,
                                                (cugraph_graph_t*)graph,
                                                (bool_t)do_expensive_check,
                                                (cugraph_labeling_result_t**)result,
                                                (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}
