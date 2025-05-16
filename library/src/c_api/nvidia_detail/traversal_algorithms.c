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
#include <cugraph_c/traversal_algorithms.h>
#include "hipgraph/hipgraph_c/traversal_algorithms.h"

hipgraph_type_erased_device_array_view_t*
    hipgraph_paths_result_get_vertices(hipgraph_paths_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_paths_result_get_vertices((cugraph_paths_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_paths_result_get_distances(hipgraph_paths_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_paths_result_get_distances((cugraph_paths_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_paths_result_get_predecessors(hipgraph_paths_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_paths_result_get_predecessors((cugraph_paths_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}

void hipgraph_paths_result_free(hipgraph_paths_result_t* result)
{
    cugraph_paths_result_free((cugraph_paths_result_t*)result);
}

hipgraph_error_code_t
    hipgraph_bfs(const hipgraph_resource_handle_t*         handle,
                 hipgraph_graph_t*                         graph,
                 hipgraph_type_erased_device_array_view_t* sources, // Implement the "FIXME"
                 hipgraph_bool_t                           direction_optimizing,
                 size_t                                    depth_limit,
                 hipgraph_bool_t                           compute_predecessors,
                 hipgraph_bool_t                           do_expensive_check,
                 hipgraph_paths_result_t**                 result,
                 hipgraph_error_t**                        error)
{
    // // Make a copy in case the sources need to be modified. See the FIXME in cugraph.
    // cugraph_type_erased_device_array_view_t* sources_copy;
    // cugraph_error_code_t err = hipgraph_type_erased_device_array_view_copy((const cugraph_resource_handle_t*)handle, sources_copy, (cugraph_type_erased_device_array_view_t*)sources, (cugraph_error_t**)error);
    // if (err != CUGRAPH_SUCCESS) return (hipgraph_error_code_t)err;

    cugraph_error_code_t err;
    // IIRC, clang obeys these as well.
    // #pragma GCC diagnostic push
    // #pragma GCC diagnostic ignored "-Wcast_qual"
    err = cugraph_bfs((const cugraph_resource_handle_t*)handle,
                      (cugraph_graph_t*)graph,
                      // Casting away the const is debatable but
                      // provides the const outer interface
                      // desired in the cugraph "FIXME" comment.
                      // An alternative would be to copy the
                      // array as in the commented code, but
                      // that's a performance hit that isn't
                      // needed yet.
                      (cugraph_type_erased_device_array_view_t*)sources,
                      (bool_t)direction_optimizing,
                      depth_limit,
                      (bool_t)compute_predecessors,
                      (bool_t)do_expensive_check,
                      (cugraph_paths_result_t**)result,
                      (cugraph_error_t**)error);
    //#pragma GCC diagnostic pop

    // cugraph_type_erased_device_array_view_free(sources_copy);

    return (hipgraph_error_code_t)err;
}

hipgraph_error_code_t hipgraph_sssp(const hipgraph_resource_handle_t* handle,
                                    hipgraph_graph_t*                 graph,
                                    size_t                            source,
                                    double                            cutoff,
                                    hipgraph_bool_t                   compute_predecessors,
                                    hipgraph_bool_t                   do_expensive_check,
                                    hipgraph_paths_result_t**         result,
                                    hipgraph_error_t**                error)
{
    cugraph_error_code_t err;
    err = cugraph_sssp((const cugraph_resource_handle_t*)handle,
                       (cugraph_graph_t*)graph,
                       source,
                       cutoff,
                       (bool_t)compute_predecessors,
                       (bool_t)do_expensive_check,
                       (cugraph_paths_result_t**)result,
                       (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
}

hipgraph_error_code_t
    hipgraph_extract_paths(const hipgraph_resource_handle_t*               handle,
                           hipgraph_graph_t*                               graph,
                           const hipgraph_type_erased_device_array_view_t* sources,
                           const hipgraph_paths_result_t*                  paths_result,
                           const hipgraph_type_erased_device_array_view_t* destinations,
                           hipgraph_extract_paths_result_t**               result,
                           hipgraph_error_t**                              error)
{
    cugraph_error_code_t err;
    err = cugraph_extract_paths((const cugraph_resource_handle_t*)handle,
                                (cugraph_graph_t*)graph,
                                (const cugraph_type_erased_device_array_view_t*)sources,
                                (const cugraph_paths_result_t*)paths_result,
                                (const cugraph_type_erased_device_array_view_t*)destinations,
                                (cugraph_extract_paths_result_t**)result,
                                (cugraph_error_t**)error);
    return (hipgraph_error_code_t)err;
}

size_t hipgraph_extract_paths_result_get_max_path_length(hipgraph_extract_paths_result_t* result)
{
    return cugraph_extract_paths_result_get_max_path_length(
        (cugraph_extract_paths_result_t*)result);
}

hipgraph_type_erased_device_array_view_t*
    hipgraph_extract_paths_result_get_paths(hipgraph_extract_paths_result_t* result)
{
    return (hipgraph_type_erased_device_array_view_t*)cugraph_extract_paths_result_get_paths(
        (cugraph_extract_paths_result_t*)result);
}
void hipgraph_extract_paths_result_free(hipgraph_extract_paths_result_t* result)
{
    cugraph_extract_paths_result_free((cugraph_extract_paths_result_t*)result);
}
