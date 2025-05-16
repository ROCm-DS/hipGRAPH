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
#include <cugraph_c/graph_functions.h>
#include "hipgraph/hipgraph_c/graph_functions.h"

/*
 hipgraph_create_vertex_pairs
*/
hipgraph_error_code_t
    hipgraph_create_vertex_pairs(const hipgraph_resource_handle_t*               handle,
                                 hipgraph_graph_t*                               graph,
                                 const hipgraph_type_erased_device_array_view_t* first,
                                 const hipgraph_type_erased_device_array_view_t* second,
                                 hipgraph_vertex_pairs_t**                       vertex_pairs,
                                 hipgraph_error_t**                              error)
{

    cugraph_error_code_t out;

    out = cugraph_create_vertex_pairs((const cugraph_resource_handle_t*)handle,
                                      (cugraph_graph_t*)graph,
                                      (const cugraph_type_erased_device_array_view_t*)first,
                                      (const cugraph_type_erased_device_array_view_t*)second,
                                      (cugraph_vertex_pairs_t**)vertex_pairs,
                                      (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_vertex_pairs_get_first
*/
hipgraph_type_erased_device_array_view_t*
    hipgraph_vertex_pairs_get_first(hipgraph_vertex_pairs_t* vertex_pairs)
{

    cugraph_type_erased_device_array_view_t* d_arrview;

    d_arrview = cugraph_vertex_pairs_get_first((cugraph_vertex_pairs_t*)vertex_pairs);

    return (hipgraph_type_erased_device_array_view_t*)d_arrview;
}

/*
 hipgraph_vertex_pairs_get_second
*/
hipgraph_type_erased_device_array_view_t*
    hipgraph_vertex_pairs_get_second(hipgraph_vertex_pairs_t* vertex_pairs)
{

    cugraph_type_erased_device_array_view_t* d_arrview;

    d_arrview = cugraph_vertex_pairs_get_second((cugraph_vertex_pairs_t*)vertex_pairs);

    return (hipgraph_type_erased_device_array_view_t*)d_arrview;
}

/*
 hipgraph_vertex_pairs_free
*/
void hipgraph_vertex_pairs_free(hipgraph_vertex_pairs_t* vertex_pairs)
{
    cugraph_vertex_pairs_free((cugraph_vertex_pairs_t*)vertex_pairs);
}

/*
 hipgraph_two_hop_neighbors
*/
hipgraph_error_code_t
    hipgraph_two_hop_neighbors(const hipgraph_resource_handle_t*               handle,
                               hipgraph_graph_t*                               graph,
                               const hipgraph_type_erased_device_array_view_t* start_vertices,
                               hipgraph_bool_t                                 do_expensive_check,
                               hipgraph_vertex_pairs_t**                       result,
                               hipgraph_error_t**                              error)
{

    cugraph_error_code_t out;

    out = cugraph_two_hop_neighbors((const cugraph_resource_handle_t*)handle,
                                    (cugraph_graph_t*)graph,
                                    (const cugraph_type_erased_device_array_view_t*)start_vertices,
                                    (bool_t)do_expensive_check,
                                    (cugraph_vertex_pairs_t**)result,
                                    (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_induced_subgraph_get_sources
*/
hipgraph_type_erased_device_array_view_t*
    hipgraph_induced_subgraph_get_sources(hipgraph_induced_subgraph_result_t* induced_subgraph)
{

    cugraph_type_erased_device_array_view_t* d_arrview;

    d_arrview = cugraph_induced_subgraph_get_sources(
        (cugraph_induced_subgraph_result_t*)induced_subgraph);

    return (hipgraph_type_erased_device_array_view_t*)d_arrview;
}

/*
 hipgraph_induced_subgraph_get_destinations
*/
hipgraph_type_erased_device_array_view_t*
    hipgraph_induced_subgraph_get_destinations(hipgraph_induced_subgraph_result_t* induced_subgraph)
{

    cugraph_type_erased_device_array_view_t* d_arrview;

    d_arrview = cugraph_induced_subgraph_get_destinations(
        (cugraph_induced_subgraph_result_t*)induced_subgraph);

    return (hipgraph_type_erased_device_array_view_t*)d_arrview;
}

/*
 hipgraph_induced_subgraph_get_edge_weights
*/
hipgraph_type_erased_device_array_view_t*
    hipgraph_induced_subgraph_get_edge_weights(hipgraph_induced_subgraph_result_t* induced_subgraph)
{

    cugraph_type_erased_device_array_view_t* d_arrview;

    d_arrview = cugraph_induced_subgraph_get_edge_weights(
        (cugraph_induced_subgraph_result_t*)induced_subgraph);

    return (hipgraph_type_erased_device_array_view_t*)d_arrview;
}

/*
 hipgraph_induced_subgraph_get_edge_ids
*/
hipgraph_type_erased_device_array_view_t*
    hipgraph_induced_subgraph_get_edge_ids(hipgraph_induced_subgraph_result_t* induced_subgraph)
{

    cugraph_type_erased_device_array_view_t* d_arrview;

    d_arrview = cugraph_induced_subgraph_get_edge_ids(
        (cugraph_induced_subgraph_result_t*)induced_subgraph);

    return (hipgraph_type_erased_device_array_view_t*)d_arrview;
}

/*
 hipgraph_induced_subgraph_get_edge_type_ids
*/
hipgraph_type_erased_device_array_view_t* hipgraph_induced_subgraph_get_edge_type_ids(
    hipgraph_induced_subgraph_result_t* induced_subgraph)
{

    cugraph_type_erased_device_array_view_t* d_arrview;

    d_arrview = cugraph_induced_subgraph_get_edge_type_ids(
        (cugraph_induced_subgraph_result_t*)induced_subgraph);

    return (hipgraph_type_erased_device_array_view_t*)d_arrview;
}

/*
 hipgraph_induced_subgraph_get_subgraph_offsets
*/
hipgraph_type_erased_device_array_view_t* hipgraph_induced_subgraph_get_subgraph_offsets(
    hipgraph_induced_subgraph_result_t* induced_subgraph)
{

    cugraph_type_erased_device_array_view_t* d_arrview;

    d_arrview = cugraph_induced_subgraph_get_subgraph_offsets(
        (cugraph_induced_subgraph_result_t*)induced_subgraph);

    return (hipgraph_type_erased_device_array_view_t*)d_arrview;
}

/*
 hipgraph_induced_subgraph_result_free
*/
void hipgraph_induced_subgraph_result_free(hipgraph_induced_subgraph_result_t* induced_subgraph)
{
    cugraph_induced_subgraph_result_free((cugraph_induced_subgraph_result_t*)induced_subgraph);
}

/*
 hipgraph_extract_induced_subgraph
*/
hipgraph_error_code_t hipgraph_extract_induced_subgraph(
    const hipgraph_resource_handle_t*               handle,
    hipgraph_graph_t*                               graph,
    const hipgraph_type_erased_device_array_view_t* subgraph_offsets,
    const hipgraph_type_erased_device_array_view_t* subgraph_vertices,
    hipgraph_bool_t                                 do_expensive_check,
    hipgraph_induced_subgraph_result_t**            result,
    hipgraph_error_t**                              error)
{

    cugraph_error_code_t out;

    out = cugraph_extract_induced_subgraph(
        (const cugraph_resource_handle_t*)handle,
        (cugraph_graph_t*)graph,
        (const cugraph_type_erased_device_array_view_t*)subgraph_offsets,
        (const cugraph_type_erased_device_array_view_t*)subgraph_vertices,
        (bool_t)do_expensive_check,
        (cugraph_induced_subgraph_result_t**)result,
        (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_allgather
*/
hipgraph_error_code_t
    hipgraph_allgather(const hipgraph_resource_handle_t*               handle,
                       const hipgraph_type_erased_device_array_view_t* src,
                       const hipgraph_type_erased_device_array_view_t* dst,
                       const hipgraph_type_erased_device_array_view_t* weights,
                       const hipgraph_type_erased_device_array_view_t* edge_ids,
                       const hipgraph_type_erased_device_array_view_t* edge_type_ids,
                       hipgraph_induced_subgraph_result_t**            result,
                       hipgraph_error_t**                              error)
{

    cugraph_error_code_t out;

    out = cugraph_allgather((const cugraph_resource_handle_t*)handle,
                            (const cugraph_type_erased_device_array_view_t*)src,
                            (const cugraph_type_erased_device_array_view_t*)dst,
                            (const cugraph_type_erased_device_array_view_t*)weights,
                            (const cugraph_type_erased_device_array_view_t*)edge_ids,
                            (const cugraph_type_erased_device_array_view_t*)edge_type_ids,
                            (cugraph_induced_subgraph_result_t**)result,
                            (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_in_degrees(
*/
hipgraph_error_code_t
    hipgraph_in_degrees(const hipgraph_resource_handle_t*               handle,
                        hipgraph_graph_t*                               graph,
                        const hipgraph_type_erased_device_array_view_t* source_vertices,
                        hipgraph_bool_t                                 do_expensive_check,
                        hipgraph_degrees_result_t**                     result,
                        hipgraph_error_t**                              error)
{

    cugraph_error_code_t out;

    out = cugraph_in_degrees((const cugraph_resource_handle_t*)handle,
                             (cugraph_graph_t*)graph,
                             (const cugraph_type_erased_device_array_view_t*)source_vertices,
                             (bool_t)do_expensive_check,
                             (cugraph_degrees_result_t**)result,
                             (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_out_degrees
*/
hipgraph_error_code_t
    hipgraph_out_degrees(const hipgraph_resource_handle_t*               handle,
                         hipgraph_graph_t*                               graph,
                         const hipgraph_type_erased_device_array_view_t* source_vertices,
                         hipgraph_bool_t                                 do_expensive_check,
                         hipgraph_degrees_result_t**                     result,
                         hipgraph_error_t**                              error)
{

    cugraph_error_code_t out;

    out = cugraph_out_degrees((const cugraph_resource_handle_t*)handle,
                              (cugraph_graph_t*)graph,
                              (const cugraph_type_erased_device_array_view_t*)source_vertices,
                              (bool_t)do_expensive_check,
                              (cugraph_degrees_result_t**)result,
                              (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_degrees
*/
hipgraph_error_code_t
    hipgraph_degrees(const hipgraph_resource_handle_t*               handle,
                     hipgraph_graph_t*                               graph,
                     const hipgraph_type_erased_device_array_view_t* source_vertices,
                     hipgraph_bool_t                                 do_expensive_check,
                     hipgraph_degrees_result_t**                     result,
                     hipgraph_error_t**                              error)
{

    cugraph_error_code_t out;

    out = cugraph_degrees((const cugraph_resource_handle_t*)handle,
                          (cugraph_graph_t*)graph,
                          (const cugraph_type_erased_device_array_view_t*)source_vertices,
                          (bool_t)do_expensive_check,
                          (cugraph_degrees_result_t**)result,
                          (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_degrees_result_get_vertices
*/
hipgraph_type_erased_device_array_view_t*
    hipgraph_degrees_result_get_vertices(hipgraph_degrees_result_t* degrees_result)
{

    cugraph_type_erased_device_array_view_t* d_arrview;

    d_arrview = cugraph_degrees_result_get_vertices((cugraph_degrees_result_t*)degrees_result);

    return (hipgraph_type_erased_device_array_view_t*)d_arrview;
}

/*
 hipgraph_degrees_result_get_in_degrees
*/
hipgraph_type_erased_device_array_view_t*
    hipgraph_degrees_result_get_in_degrees(hipgraph_degrees_result_t* degrees_result)
{

    cugraph_type_erased_device_array_view_t* d_arrview;

    d_arrview = cugraph_degrees_result_get_in_degrees((cugraph_degrees_result_t*)degrees_result);

    return (hipgraph_type_erased_device_array_view_t*)d_arrview;
}

/*
 hipgraph_degrees_result_get_out_degrees
*/
hipgraph_type_erased_device_array_view_t*
    hipgraph_degrees_result_get_out_degrees(hipgraph_degrees_result_t* degrees_result)
{

    cugraph_type_erased_device_array_view_t* d_arrview;

    d_arrview = cugraph_degrees_result_get_out_degrees((cugraph_degrees_result_t*)degrees_result);

    return (hipgraph_type_erased_device_array_view_t*)d_arrview;
}

/*
 hipgraph_degrees_result_free
*/
void hipgraph_degrees_result_free(hipgraph_degrees_result_t* degrees_result)
{
    cugraph_degrees_result_free((cugraph_degrees_result_t*)degrees_result);
}
