// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*
 * Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 */

/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "common.h"
#include <rocgraph/rocgraph.h>
#include "hipgraph/hipgraph_c/graph_generators.h"

hipgraph_type_erased_device_array_view_t* hipgraph_coo_get_sources(hipgraph_coo_t* coo)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_coo_get_sources(
        (rocgraph_coo_t*)coo);
}

hipgraph_type_erased_device_array_view_t* hipgraph_coo_get_destinations(hipgraph_coo_t* coo)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_coo_get_destinations(
        (rocgraph_coo_t*)coo);
}

hipgraph_type_erased_device_array_view_t* hipgraph_coo_get_edge_weights(hipgraph_coo_t* coo)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_coo_get_edge_weights(
        (rocgraph_coo_t*)coo);
}

hipgraph_type_erased_device_array_view_t* hipgraph_coo_get_edge_id(hipgraph_coo_t* coo)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_coo_get_edge_id(
        (rocgraph_coo_t*)coo);
}

hipgraph_type_erased_device_array_view_t* hipgraph_coo_get_edge_type(hipgraph_coo_t* coo)
{
    return (hipgraph_type_erased_device_array_view_t*)rocgraph_coo_get_edge_type(
        (rocgraph_coo_t*)coo);
}

size_t hipgraph_coo_list_size(const hipgraph_coo_list_t* coo_list)
{
    return (size_t)rocgraph_coo_list_size((const rocgraph_coo_list_t*)coo_list);
}

hipgraph_coo_t* hipgraph_coo_list_element(hipgraph_coo_list_t* coo_list, size_t index)
{
    return (hipgraph_coo_t*)rocgraph_coo_list_element((rocgraph_coo_list_t*)coo_list, index);
}

void hipgraph_coo_free(hipgraph_coo_t* coo)
{
    rocgraph_coo_free((rocgraph_coo_t*)coo);
}

void hipgraph_coo_list_free(hipgraph_coo_list_t* coo_list)
{
    rocgraph_coo_list_free((rocgraph_coo_list_t*)coo_list);
}

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
    rocgraph_bool rg_clip_and_flip = hipgraph_bool_t2rocgraph_bool(clip_and_flip);
    if(hghelper_rocgraph_bool_is_invalid(rg_clip_and_flip))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_scramble_vertex_ids = hipgraph_bool_t2rocgraph_bool(scramble_vertex_ids);
    if(hghelper_rocgraph_bool_is_invalid(rg_scramble_vertex_ids))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_generate_rmat_edgelist((const rocgraph_handle_t*)handle,
                                                                (rocgraph_rng_state_t*)rng_state,
                                                                scale,
                                                                num_edges,
                                                                a,
                                                                b,
                                                                c,
                                                                rg_clip_and_flip,
                                                                rg_scramble_vertex_ids,
                                                                (rocgraph_coo_t**)result,
                                                                (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

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
    rocgraph_bool rg_clip_and_flip = hipgraph_bool_t2rocgraph_bool(clip_and_flip);
    if(hghelper_rocgraph_bool_is_invalid(rg_clip_and_flip))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_scramble_vertex_ids = hipgraph_bool_t2rocgraph_bool(scramble_vertex_ids);
    if(hghelper_rocgraph_bool_is_invalid(rg_scramble_vertex_ids))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_generate_rmat_edgelists(
        (const rocgraph_handle_t*)handle,
        (rocgraph_rng_state_t*)rng_state,
        n_edgelists,
        min_scale,
        max_scale,
        edge_factor,
        hipgraph_generator_distribution_t2rocgraph_generator_distribution(size_distribution),
        hipgraph_generator_distribution_t2rocgraph_generator_distribution(edge_distribution),
        rg_clip_and_flip,
        rg_scramble_vertex_ids,
        (rocgraph_coo_list_t**)result,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t hipgraph_generate_edge_weights(const hipgraph_resource_handle_t* handle,
                                                     hipgraph_rng_state_t*             rng_state,
                                                     hipgraph_coo_t*                   coo,
                                                     hipgraph_data_type_id_t           dtype,
                                                     double             minimum_weight,
                                                     double             maximum_weight,
                                                     hipgraph_error_t** error)
{
    rocgraph_data_type_id rg_dtype = hipgraph_data_type_id_t2rocgraph_data_type_id(dtype);
    if(hghelper_rocgraph_data_type_id_is_invalid(rg_dtype))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_generate_edge_weights((const rocgraph_handle_t*)handle,
                                                               (rocgraph_rng_state_t*)rng_state,
                                                               (rocgraph_coo_t*)coo,
                                                               rg_dtype,
                                                               minimum_weight,
                                                               maximum_weight,
                                                               (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t hipgraph_generate_edge_ids(const hipgraph_resource_handle_t* handle,
                                                 hipgraph_coo_t*                   coo,
                                                 hipgraph_bool_t                   multi_gpu,
                                                 hipgraph_error_t**                error)
{
    rocgraph_bool rg_multi_gpu = hipgraph_bool_t2rocgraph_bool(multi_gpu);
    if(hghelper_rocgraph_bool_is_invalid(rg_multi_gpu))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_generate_edge_ids((const rocgraph_handle_t*)handle,
                                                           (rocgraph_coo_t*)coo,
                                                           rg_multi_gpu,
                                                           (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

hipgraph_error_code_t hipgraph_generate_edge_types(const hipgraph_resource_handle_t* handle,
                                                   hipgraph_rng_state_t*             rng_state,
                                                   hipgraph_coo_t*                   coo,
                                                   int32_t                           min_edge_type,
                                                   int32_t                           max_edge_type,
                                                   hipgraph_error_t**                error)
{
    rocgraph_status rg_status = rocgraph_generate_edge_types((const rocgraph_handle_t*)handle,
                                                             (rocgraph_rng_state_t*)rng_state,
                                                             (rocgraph_coo_t*)coo,
                                                             min_edge_type,
                                                             max_edge_type,
                                                             (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}
