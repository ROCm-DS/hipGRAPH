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
#include <cugraph_c/graph.h>
#include "hipgraph/hipgraph_c/graph.h"

/*
 hipgraph_sg_graph_create
*/
hipgraph_error_code_t
    hipgraph_sg_graph_create(const hipgraph_resource_handle_t*               handle,
                             const hipgraph_graph_properties_t*              properties,
                             const hipgraph_type_erased_device_array_view_t* src,
                             const hipgraph_type_erased_device_array_view_t* dst,
                             const hipgraph_type_erased_device_array_view_t* weights,
                             const hipgraph_type_erased_device_array_view_t* edge_ids,
                             const hipgraph_type_erased_device_array_view_t* edge_type_ids,
                             hipgraph_bool_t                                 store_transposed,
                             hipgraph_bool_t                                 renumber,
                             hipgraph_bool_t                                 do_expensive_check,
                             hipgraph_graph_t**                              graph,
                             hipgraph_error_t**                              error)
{

    cugraph_error_code_t out;

    out = cugraph_sg_graph_create((const cugraph_resource_handle_t*)handle,
                                  (const cugraph_graph_properties_t*)properties,
                                  (const cugraph_type_erased_device_array_view_t*)src,
                                  (const cugraph_type_erased_device_array_view_t*)dst,
                                  (const cugraph_type_erased_device_array_view_t*)weights,
                                  (const cugraph_type_erased_device_array_view_t*)edge_ids,
                                  (const cugraph_type_erased_device_array_view_t*)edge_type_ids,
                                  (bool_t)store_transposed,
                                  (bool_t)renumber,
                                  (bool_t)do_expensive_check,
                                  (cugraph_graph_t**)graph,
                                  (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_graph_create_sg
*/
hipgraph_error_code_t
    hipgraph_graph_create_sg(const hipgraph_resource_handle_t*               handle,
                             const hipgraph_graph_properties_t*              properties,
                             const hipgraph_type_erased_device_array_view_t* vertices,
                             const hipgraph_type_erased_device_array_view_t* src,
                             const hipgraph_type_erased_device_array_view_t* dst,
                             const hipgraph_type_erased_device_array_view_t* weights,
                             const hipgraph_type_erased_device_array_view_t* edge_ids,
                             const hipgraph_type_erased_device_array_view_t* edge_type_ids,
                             hipgraph_bool_t                                 store_transposed,
                             hipgraph_bool_t                                 renumber,
                             hipgraph_bool_t                                 drop_self_loops,
                             hipgraph_bool_t                                 drop_multi_edges,
                             hipgraph_bool_t                                 do_expensive_check,
                             hipgraph_graph_t**                              graph,
                             hipgraph_error_t**                              error)
{

    cugraph_error_code_t out;

    out = cugraph_graph_create_sg((const cugraph_resource_handle_t*)handle,
                                  (const cugraph_graph_properties_t*)properties,
                                  (const cugraph_type_erased_device_array_view_t*)vertices,
                                  (const cugraph_type_erased_device_array_view_t*)src,
                                  (const cugraph_type_erased_device_array_view_t*)dst,
                                  (const cugraph_type_erased_device_array_view_t*)weights,
                                  (const cugraph_type_erased_device_array_view_t*)edge_ids,
                                  (const cugraph_type_erased_device_array_view_t*)edge_type_ids,
                                  (bool_t)store_transposed,
                                  (bool_t)renumber,
                                  (bool_t)drop_self_loops,
                                  (bool_t)drop_multi_edges,
                                  (bool_t)do_expensive_check,
                                  (cugraph_graph_t**)graph,
                                  (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_sg_graph_create_from_csr
*/
hipgraph_error_code_t
    hipgraph_sg_graph_create_from_csr(const hipgraph_resource_handle_t*               handle,
                                      const hipgraph_graph_properties_t*              properties,
                                      const hipgraph_type_erased_device_array_view_t* offsets,
                                      const hipgraph_type_erased_device_array_view_t* indices,
                                      const hipgraph_type_erased_device_array_view_t* weights,
                                      const hipgraph_type_erased_device_array_view_t* edge_ids,
                                      const hipgraph_type_erased_device_array_view_t* edge_type_ids,
                                      hipgraph_bool_t    store_transposed,
                                      hipgraph_bool_t    renumber,
                                      hipgraph_bool_t    do_expensive_check,
                                      hipgraph_graph_t** graph,
                                      hipgraph_error_t** error)
{

    cugraph_error_code_t out;

    out = cugraph_sg_graph_create_from_csr(
        (const cugraph_resource_handle_t*)handle,
        (const cugraph_graph_properties_t*)properties,
        (const cugraph_type_erased_device_array_view_t*)offsets,
        (const cugraph_type_erased_device_array_view_t*)indices,
        (const cugraph_type_erased_device_array_view_t*)weights,
        (const cugraph_type_erased_device_array_view_t*)edge_ids,
        (const cugraph_type_erased_device_array_view_t*)edge_type_ids,
        (bool_t)store_transposed,
        (bool_t)renumber,
        (bool_t)do_expensive_check,
        (cugraph_graph_t**)graph,
        (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_graph_create_sg_from_csr
*/
hipgraph_error_code_t
    hipgraph_graph_create_sg_from_csr(const hipgraph_resource_handle_t*               handle,
                                      const hipgraph_graph_properties_t*              properties,
                                      const hipgraph_type_erased_device_array_view_t* offsets,
                                      const hipgraph_type_erased_device_array_view_t* indices,
                                      const hipgraph_type_erased_device_array_view_t* weights,
                                      const hipgraph_type_erased_device_array_view_t* edge_ids,
                                      const hipgraph_type_erased_device_array_view_t* edge_type_ids,
                                      hipgraph_bool_t    store_transposed,
                                      hipgraph_bool_t    renumber,
                                      hipgraph_bool_t    do_expensive_check,
                                      hipgraph_graph_t** graph,
                                      hipgraph_error_t** error)
{

    cugraph_error_code_t out;

    out = cugraph_graph_create_sg_from_csr(
        (const cugraph_resource_handle_t*)handle,
        (const cugraph_graph_properties_t*)properties,
        (const cugraph_type_erased_device_array_view_t*)offsets,
        (const cugraph_type_erased_device_array_view_t*)indices,
        (const cugraph_type_erased_device_array_view_t*)weights,
        (const cugraph_type_erased_device_array_view_t*)edge_ids,
        (const cugraph_type_erased_device_array_view_t*)edge_type_ids,
        (bool_t)store_transposed,
        (bool_t)renumber,
        (bool_t)do_expensive_check,
        (cugraph_graph_t**)graph,
        (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_mg_graph_create
*/
hipgraph_error_code_t
    hipgraph_mg_graph_create(const hipgraph_resource_handle_t*               handle,
                             const hipgraph_graph_properties_t*              properties,
                             const hipgraph_type_erased_device_array_view_t* src,
                             const hipgraph_type_erased_device_array_view_t* dst,
                             const hipgraph_type_erased_device_array_view_t* weights,
                             const hipgraph_type_erased_device_array_view_t* edge_ids,
                             const hipgraph_type_erased_device_array_view_t* edge_type_ids,
                             hipgraph_bool_t                                 store_transposed,
                             size_t                                          num_edges,
                             hipgraph_bool_t                                 do_expensive_check,
                             hipgraph_graph_t**                              graph,
                             hipgraph_error_t**                              error)
{

    cugraph_error_code_t out;

    out = cugraph_mg_graph_create((const cugraph_resource_handle_t*)handle,
                                  (const cugraph_graph_properties_t*)properties,
                                  (const cugraph_type_erased_device_array_view_t*)src,
                                  (const cugraph_type_erased_device_array_view_t*)dst,
                                  (const cugraph_type_erased_device_array_view_t*)weights,
                                  (const cugraph_type_erased_device_array_view_t*)edge_ids,
                                  (const cugraph_type_erased_device_array_view_t*)edge_type_ids,
                                  (bool_t)store_transposed,
                                  (size_t)num_edges,
                                  (bool_t)do_expensive_check,
                                  (cugraph_graph_t**)graph,
                                  (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_graph_create_mg
*/
hipgraph_error_code_t
    hipgraph_graph_create_mg(hipgraph_resource_handle_t const*                      handle,
                             hipgraph_graph_properties_t const*                     properties,
                             hipgraph_type_erased_device_array_view_t const* const* vertices,
                             hipgraph_type_erased_device_array_view_t const* const* src,
                             hipgraph_type_erased_device_array_view_t const* const* dst,
                             hipgraph_type_erased_device_array_view_t const* const* weights,
                             hipgraph_type_erased_device_array_view_t const* const* edge_ids,
                             hipgraph_type_erased_device_array_view_t const* const* edge_type_ids,
                             hipgraph_bool_t    store_transposed,
                             size_t             num_arrays,
                             hipgraph_bool_t    drop_self_loops,
                             hipgraph_bool_t    drop_multi_edges,
                             hipgraph_bool_t    do_expensive_check,
                             hipgraph_graph_t** graph,
                             hipgraph_error_t** error)
{

    cugraph_error_code_t out;

    out = cugraph_graph_create_mg(
        (cugraph_resource_handle_t const*)handle,
        (cugraph_graph_properties_t const*)properties,
        (cugraph_type_erased_device_array_view_t const* const*)vertices,
        (cugraph_type_erased_device_array_view_t const* const*)src,
        (cugraph_type_erased_device_array_view_t const* const*)dst,
        (cugraph_type_erased_device_array_view_t const* const*)weights,
        (cugraph_type_erased_device_array_view_t const* const*)edge_ids,
        (cugraph_type_erased_device_array_view_t const* const*)edge_type_ids,
        (bool_t)store_transposed,
        (size_t)num_arrays,
        (bool_t)drop_self_loops,
        (bool_t)drop_multi_edges,
        (bool_t)do_expensive_check,
        (cugraph_graph_t**)graph,
        (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_graph_free
*/
void hipgraph_graph_free(hipgraph_graph_t* graph)
{
    cugraph_graph_free((cugraph_graph_t*)graph);
}

/*
 hipgraph_sg_graph_free
*/
void hipgraph_sg_graph_free(hipgraph_graph_t* graph)
{
    cugraph_sg_graph_free((cugraph_graph_t*)graph);
}

/*
 hipgraph_mg_graph_free
*/
void hipgraph_mg_graph_free(hipgraph_graph_t* graph)
{
    cugraph_mg_graph_free((cugraph_graph_t*)graph);
}

#if 0
/* These are not implemented in cugraph_c. */
/*
 hipgraph_data_mask_create
*/
 hipgraph_error_code_t
    hipgraph_data_mask_create(const hipgraph_resource_handle_t*               handle,
                              const hipgraph_type_erased_device_array_view_t* vertex_bit_mask,
                              const hipgraph_type_erased_device_array_view_t* edge_bit_mask,
                              hipgraph_bool_t                                 complement,
                              hipgraph_data_mask_t**                          mask,
                              hipgraph_error_t**                              error)
{

    cugraph_error_code_t out;

    out = cugraph_data_mask_create((const cugraph_resource_handle_t*)handle,
                                   (const cugraph_type_erased_device_array_view_t*)vertex_bit_mask,
                                   (const cugraph_type_erased_device_array_view_t*)edge_bit_mask,
                                   (bool_t)complement,
                                   (cugraph_data_mask_t**)mask,
                                   (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_graph_get_data_mask
*/
 hipgraph_error_code_t hipgraph_graph_get_data_mask(hipgraph_graph_t*      graph,
                                                   hipgraph_data_mask_t** mask,
                                                   hipgraph_error_t**     error)
{

    cugraph_error_code_t out;

    out = cugraph_graph_get_data_mask(
        (cugraph_graph_t*)graph, (cugraph_data_mask_t**)mask, (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_graph_add_data_mask
*/
 hipgraph_error_code_t hipgraph_graph_add_data_mask(hipgraph_graph_t*     graph,
                                                   hipgraph_data_mask_t* mask,
                                                   hipgraph_error_t**    error)
{

    cugraph_error_code_t out;

    out = cugraph_graph_add_data_mask(
        (cugraph_graph_t*)graph, (cugraph_data_mask_t*)mask, (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_graph_release_data_mask
*/
 hipgraph_error_code_t hipgraph_graph_release_data_mask(hipgraph_graph_t*      graph,
                                                       hipgraph_data_mask_t** mask,
                                                       hipgraph_error_t**     error)
{

    cugraph_error_code_t out;

    out = cugraph_graph_release_data_mask(
        (cugraph_graph_t*)graph, (cugraph_data_mask_t**)mask, (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_data_mask_destroy
*/
void hipgraph_data_mask_destroy(hipgraph_data_mask_t* mask)
{
    cugraph_data_mask_destroy((cugraph_data_mask_t*)mask);
}
#endif
