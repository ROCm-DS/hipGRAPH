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
#include <cugraph_c/graph_generators.h>
#include "hipgraph/hipgraph_c/graph_generators.h"

/*
 hipgraph_coo_get_sources
*/
hipgraph_type_erased_device_array_view_t* hipgraph_coo_get_sources(hipgraph_coo_t* coo)
{

    cugraph_type_erased_device_array_view_t* view;

    view = cugraph_coo_get_sources((cugraph_coo_t*)coo);

    return (hipgraph_type_erased_device_array_view_t*)view;
}

/*
 hipgraph_coo_get_destinations
*/
hipgraph_type_erased_device_array_view_t* hipgraph_coo_get_destinations(hipgraph_coo_t* coo)
{

    cugraph_type_erased_device_array_view_t* view;

    view = cugraph_coo_get_destinations((cugraph_coo_t*)coo);

    return (hipgraph_type_erased_device_array_view_t*)view;
}

/*
 hipgraph_coo_get_edge_weights
*/
hipgraph_type_erased_device_array_view_t* hipgraph_coo_get_edge_weights(hipgraph_coo_t* coo)
{

    cugraph_type_erased_device_array_view_t* view;

    view = cugraph_coo_get_edge_weights((cugraph_coo_t*)coo);

    return (hipgraph_type_erased_device_array_view_t*)view;
}

/*
 hipgraph_coo_get_edge_id
*/
hipgraph_type_erased_device_array_view_t* hipgraph_coo_get_edge_id(hipgraph_coo_t* coo)
{

    cugraph_type_erased_device_array_view_t* view;

    view = cugraph_coo_get_edge_id((cugraph_coo_t*)coo);

    return (hipgraph_type_erased_device_array_view_t*)view;
}

/*
 hipgraph_coo_get_edge_type
*/
hipgraph_type_erased_device_array_view_t* hipgraph_coo_get_edge_type(hipgraph_coo_t* coo)
{

    cugraph_type_erased_device_array_view_t* view;

    view = cugraph_coo_get_edge_type((cugraph_coo_t*)coo);

    return (hipgraph_type_erased_device_array_view_t*)view;
}

/*
 hipgraph_coo_list_size
*/
size_t hipgraph_coo_list_size(const hipgraph_coo_list_t* coo_list)
{

    return cugraph_coo_list_size((const cugraph_coo_list_t*)coo_list);
}

/*
 hipgraph_coo_list_element
*/
hipgraph_coo_t* hipgraph_coo_list_element(hipgraph_coo_list_t* coo_list, size_t index)
{

    cugraph_coo_t* coo;

    coo = cugraph_coo_list_element((cugraph_coo_list_t*)coo_list, (size_t)index);

    return (hipgraph_coo_t*)coo;
}

/*
 hipgraph_coo_free
*/
void hipgraph_coo_free(hipgraph_coo_t* coo)
{
    cugraph_coo_free((cugraph_coo_t*)coo);
}

/*
 hipgraph_coo_list_free
*/
void hipgraph_coo_list_free(hipgraph_coo_list_t* coo_list)
{
    cugraph_coo_list_free((cugraph_coo_list_t*)coo_list);
}

/*
 hipgraph_generate_rmat_edgelist
*/
hipgraph_error_code_t hipgraph_generate_rmat_edgelist(const hipgraph_resource_handle_t* handle,
                                                      hipgraph_rng_state_t*             rng_state,
                                                      size_t                            scale,
                                                      size_t                            num_edges,
                                                      double                            a,
                                                      double                            b,
                                                      double                            c,
                                                      hipgraph_bool_t    clip_and_flip,
                                                      hipgraph_bool_t    scramble_vertex_ids,
                                                      hipgraph_coo_t**   result,
                                                      hipgraph_error_t** error)
{

    cugraph_error_code_t out;

    out = cugraph_generate_rmat_edgelist((const cugraph_resource_handle_t*)handle,
                                         (cugraph_rng_state_t*)rng_state,
                                         (size_t)scale,
                                         (size_t)num_edges,
                                         (double)a,
                                         (double)b,
                                         (double)c,
                                         (bool_t)clip_and_flip,
                                         (bool_t)scramble_vertex_ids,
                                         (cugraph_coo_t**)result,
                                         (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_generate_rmat_edgelists
*/
hipgraph_error_code_t
    hipgraph_generate_rmat_edgelists(const hipgraph_resource_handle_t* handle,
                                     hipgraph_rng_state_t*             rng_state,
                                     size_t                            n_edgelists,
                                     size_t                            min_scale,
                                     size_t                            max_scale,
                                     size_t                            edge_factor,
                                     hipgraph_generator_distribution_t size_distribution,
                                     hipgraph_generator_distribution_t edge_distribution,
                                     hipgraph_bool_t                   clip_and_flip,
                                     hipgraph_bool_t                   scramble_vertex_ids,
                                     hipgraph_coo_list_t**             result,
                                     hipgraph_error_t**                error)
{

    cugraph_error_code_t out;

    out = cugraph_generate_rmat_edgelists((const cugraph_resource_handle_t*)handle,
                                          (cugraph_rng_state_t*)rng_state,
                                          (size_t)n_edgelists,
                                          (size_t)min_scale,
                                          (size_t)max_scale,
                                          (size_t)edge_factor,
                                          (cugraph_generator_distribution_t)size_distribution,
                                          (cugraph_generator_distribution_t)edge_distribution,
                                          (bool_t)clip_and_flip,
                                          (bool_t)scramble_vertex_ids,
                                          (cugraph_coo_list_t**)result,
                                          (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_generate_edge_weights
*/
hipgraph_error_code_t hipgraph_generate_edge_weights(const hipgraph_resource_handle_t* handle,
                                                     hipgraph_rng_state_t*             rng_state,
                                                     hipgraph_coo_t*                   coo,
                                                     hipgraph_data_type_id_t           dtype,
                                                     double             minimum_weight,
                                                     double             maximum_weight,
                                                     hipgraph_error_t** error)
{

    cugraph_error_code_t out;

    out = cugraph_generate_edge_weights((const cugraph_resource_handle_t*)handle,
                                        (cugraph_rng_state_t*)rng_state,
                                        (cugraph_coo_t*)coo,
                                        (cugraph_data_type_id_t)dtype,
                                        (double)minimum_weight,
                                        (double)maximum_weight,
                                        (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_generate_edge_ids
*/
hipgraph_error_code_t hipgraph_generate_edge_ids(const hipgraph_resource_handle_t* handle,
                                                 hipgraph_coo_t*                   coo,
                                                 hipgraph_bool_t                   multi_gpu,
                                                 hipgraph_error_t**                error)
{

    cugraph_error_code_t out;

    out = cugraph_generate_edge_ids((const cugraph_resource_handle_t*)handle,
                                    (cugraph_coo_t*)coo,
                                    (bool_t)multi_gpu,
                                    (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}

/*
 hipgraph_generate_edge_types
*/
hipgraph_error_code_t hipgraph_generate_edge_types(const hipgraph_resource_handle_t* handle,
                                                   hipgraph_rng_state_t*             rng_state,
                                                   hipgraph_coo_t*                   coo,
                                                   int32_t                           min_edge_type,
                                                   int32_t                           max_edge_type,
                                                   hipgraph_error_t**                error)
{

    cugraph_error_code_t out;
    out = cugraph_generate_edge_types((const cugraph_resource_handle_t*)handle,
                                      (cugraph_rng_state_t*)rng_state,
                                      (cugraph_coo_t*)coo,
                                      (int32_t)min_edge_type,
                                      (int32_t)max_edge_type,
                                      (cugraph_error_t**)error);

    return (hipgraph_error_code_t)out;
}
