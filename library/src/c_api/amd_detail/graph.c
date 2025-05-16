// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*
 * Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 */

/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
#include "hipgraph/hipgraph_c/graph.h"

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
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_renumber = hipgraph_bool_t2rocgraph_bool(renumber);
    if(hghelper_rocgraph_bool_is_invalid(rg_renumber))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_store_transposed = hipgraph_bool_t2rocgraph_bool(store_transposed);
    if(hghelper_rocgraph_bool_is_invalid(rg_store_transposed))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status
        = rocgraph_sg_graph_create((const rocgraph_handle_t*)handle,
                                   (const rocgraph_graph_properties_t*)properties,
                                   (const rocgraph_type_erased_device_array_view_t*)src,
                                   (const rocgraph_type_erased_device_array_view_t*)dst,
                                   (const rocgraph_type_erased_device_array_view_t*)weights,
                                   (const rocgraph_type_erased_device_array_view_t*)edge_ids,
                                   (const rocgraph_type_erased_device_array_view_t*)edge_type_ids,
                                   rg_store_transposed,
                                   rg_renumber,
                                   rg_do_expensive_check,
                                   (rocgraph_graph_t**)graph,
                                   (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

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
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_drop_multi_edges = hipgraph_bool_t2rocgraph_bool(drop_multi_edges);
    if(hghelper_rocgraph_bool_is_invalid(rg_drop_multi_edges))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_drop_self_loops = hipgraph_bool_t2rocgraph_bool(drop_self_loops);
    if(hghelper_rocgraph_bool_is_invalid(rg_drop_self_loops))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_renumber = hipgraph_bool_t2rocgraph_bool(renumber);
    if(hghelper_rocgraph_bool_is_invalid(rg_renumber))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_store_transposed = hipgraph_bool_t2rocgraph_bool(store_transposed);
    if(hghelper_rocgraph_bool_is_invalid(rg_store_transposed))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status
        = rocgraph_graph_create_sg((const rocgraph_handle_t*)handle,
                                   (const rocgraph_graph_properties_t*)properties,
                                   (const rocgraph_type_erased_device_array_view_t*)vertices,
                                   (const rocgraph_type_erased_device_array_view_t*)src,
                                   (const rocgraph_type_erased_device_array_view_t*)dst,
                                   (const rocgraph_type_erased_device_array_view_t*)weights,
                                   (const rocgraph_type_erased_device_array_view_t*)edge_ids,
                                   (const rocgraph_type_erased_device_array_view_t*)edge_type_ids,
                                   rg_store_transposed,
                                   rg_renumber,
                                   rg_drop_self_loops,
                                   rg_drop_multi_edges,
                                   rg_do_expensive_check,
                                   (rocgraph_graph_t**)graph,
                                   (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

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
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_renumber = hipgraph_bool_t2rocgraph_bool(renumber);
    if(hghelper_rocgraph_bool_is_invalid(rg_renumber))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_store_transposed = hipgraph_bool_t2rocgraph_bool(store_transposed);
    if(hghelper_rocgraph_bool_is_invalid(rg_store_transposed))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_sg_graph_create_from_csr(
        (const rocgraph_handle_t*)handle,
        (const rocgraph_graph_properties_t*)properties,
        (const rocgraph_type_erased_device_array_view_t*)offsets,
        (const rocgraph_type_erased_device_array_view_t*)indices,
        (const rocgraph_type_erased_device_array_view_t*)weights,
        (const rocgraph_type_erased_device_array_view_t*)edge_ids,
        (const rocgraph_type_erased_device_array_view_t*)edge_type_ids,
        rg_store_transposed,
        rg_renumber,
        rg_do_expensive_check,
        (rocgraph_graph_t**)graph,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

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
    rocgraph_bool rg_do_expensive_check = hipgraph_bool_t2rocgraph_bool(do_expensive_check);
    if(hghelper_rocgraph_bool_is_invalid(rg_do_expensive_check))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_renumber = hipgraph_bool_t2rocgraph_bool(renumber);
    if(hghelper_rocgraph_bool_is_invalid(rg_renumber))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_bool rg_store_transposed = hipgraph_bool_t2rocgraph_bool(store_transposed);
    if(hghelper_rocgraph_bool_is_invalid(rg_store_transposed))
        return HIPGRAPH_UNKNOWN_ERROR;
    rocgraph_status rg_status = rocgraph_graph_create_sg_from_csr(
        (const rocgraph_handle_t*)handle,
        (const rocgraph_graph_properties_t*)properties,
        (const rocgraph_type_erased_device_array_view_t*)offsets,
        (const rocgraph_type_erased_device_array_view_t*)indices,
        (const rocgraph_type_erased_device_array_view_t*)weights,
        (const rocgraph_type_erased_device_array_view_t*)edge_ids,
        (const rocgraph_type_erased_device_array_view_t*)edge_type_ids,
        rg_store_transposed,
        rg_renumber,
        rg_do_expensive_check,
        (rocgraph_graph_t**)graph,
        (rocgraph_error_t**)error);
    return rocgraph_status2hipgraph_error_code_t(rg_status);
}

// TODO: Disabling multi-GPU support.
#if defined(HIPGRAPH_MULTIGPU_SUPPORT)
// Not yet.
#else
#if !defined(NDEBUG)
#warning ejr: Disabling multi-GPU support. See the following #if 0 blocks.
#endif

#if 0
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
    return (hipgraph_error_code_t)rocgraph_mg_graph_create(
        (const rocgraph_handle_t*)handle,
        (const rocgraph_graph_properties_t*)properties,
        (const rocgraph_type_erased_device_array_view_t*)src,
        (const rocgraph_type_erased_device_array_view_t*)dst,
        (const rocgraph_type_erased_device_array_view_t*)weights,
        (const rocgraph_type_erased_device_array_view_t*)edge_ids,
        (const rocgraph_type_erased_device_array_view_t*)edge_type_ids,
        hipgraph_bool_t2rocgraph_bool(store_transposed),
        num_edges,
        hipgraph_bool_t2rocgraph_bool(do_expensive_check),
        (rocgraph_graph_t**)graph,
        (rocgraph_error_t**)error);
}
#endif

#if 0 // Currently unsupported
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
    return (hipgraph_error_code_t)rocgraph_graph_create_mg(
        (rocgraph_handle_t const*)handle,
        (rocgraph_graph_properties_t const*)properties,
        (rocgraph_type_erased_device_array_view_t const* const*)vertices,
        (rocgraph_type_erased_device_array_view_t const* const*)src,
        (rocgraph_type_erased_device_array_view_t const* const*)dst,
        (rocgraph_type_erased_device_array_view_t const* const*)weights,
        (rocgraph_type_erased_device_array_view_t const* const*)edge_ids,
        (rocgraph_type_erased_device_array_view_t const* const*)edge_type_ids,
        hipgraph_bool_t2rocgraph_bool(store_transposed),
        num_arrays,
        hipgraph_bool_t2rocgraph_bool(drop_self_loops),
        hipgraph_bool_t2rocgraph_bool(drop_multi_edges),
        hipgraph_bool_t2rocgraph_bool(do_expensive_check),
        (rocgraph_graph_t**)graph,
        (rocgraph_error_t**)error);
}
#endif
#endif

void hipgraph_graph_free(hipgraph_graph_t* graph)
{
    rocgraph_graph_free((rocgraph_graph_t*)graph);
}

void hipgraph_sg_graph_free(hipgraph_graph_t* graph)
{
    rocgraph_sg_graph_free((rocgraph_graph_t*)graph);
}

// TODO: Disabling multi-GPU support.
#if defined(HIPGRAPH_MULTIGPU_SUPPORT)
// Not yet.
#else
#if !defined(NDEBUG)
#warning ejr: Disabling multi-GPU support. See the following #if 0 blocks.
#endif

#if 0
void hipgraph_mg_graph_free(hipgraph_graph_t* graph)
{
    rocgraph_mg_graph_free((rocgraph_graph_t*)graph);
}
#endif

#if 0
hipgraph_error_code_t
    hipgraph_data_mask_create(const hipgraph_resource_handle_t*               handle,
                              const hipgraph_type_erased_device_array_view_t* vertex_bit_mask,
                              const hipgraph_type_erased_device_array_view_t* edge_bit_mask,
                              hipgraph_bool_t                                 complement,
                              hipgraph_data_mask_t**                          mask,
                              hipgraph_error_t**                              error)
{
    return (hipgraph_error_code_t)rocgraph_data_mask_create(
        (const rocgraph_handle_t*)handle,
        (const rocgraph_type_erased_device_array_view_t*)vertex_bit_mask,
        (const rocgraph_type_erased_device_array_view_t*)edge_bit_mask,
        hipgraph_bool_t2rocgraph_bool(complement),
        (rocgraph_data_mask_t**)mask,
        (rocgraph_error_t**)error);
}

hipgraph_error_code_t hipgraph_graph_get_data_mask(hipgraph_graph_t*      graph,
                                                                   hipgraph_data_mask_t** mask,
                                                                   hipgraph_error_t**     error)
{
    return (hipgraph_error_code_t)rocgraph_graph_get_data_mask(
        (rocgraph_graph_t*)graph, (rocgraph_data_mask_t**)mask, (rocgraph_error_t**)error);
}

hipgraph_error_code_t hipgraph_graph_add_data_mask(hipgraph_graph_t*     graph,
                                                                   hipgraph_data_mask_t* mask,
                                                                   hipgraph_error_t**    error)
{
    return (hipgraph_error_code_t)rocgraph_graph_add_data_mask(
        (rocgraph_graph_t*)graph, (rocgraph_data_mask_t*)mask, (rocgraph_error_t**)error);
}

hipgraph_error_code_t hipgraph_graph_release_data_mask(hipgraph_graph_t*      graph,
                                                                       hipgraph_data_mask_t** mask,
                                                                       hipgraph_error_t**     error)
{
    return (hipgraph_error_code_t)rocgraph_graph_release_data_mask(
        (rocgraph_graph_t*)graph, (rocgraph_data_mask_t**)mask, (rocgraph_error_t**)error);
}

void hipgraph_data_mask_destroy(hipgraph_data_mask_t* mask)
{
    rocgraph_data_mask_destroy((rocgraph_data_mask_t*)mask);
}
#endif
#endif
