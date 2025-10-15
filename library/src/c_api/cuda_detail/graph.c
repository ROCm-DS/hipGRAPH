// -*- C -*-
// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common.h"
#include <cugraph_c/array.h>
#include <hipgraph_c/array.h>
#include <cugraph_c/hipgraph/hipgraph-common.h>
#include <hipgraph_c/hipgraph/hipgraph-common.h>
#include "error_code.inl.h"

HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_graph_create_sg(const hipgraph_resource_handle_t* handle, const hipgraph_graph_properties_t* properties, const hipgraph_type_erased_device_array_view_t* vertices, const hipgraph_type_erased_device_array_view_t* src, const hipgraph_type_erased_device_array_view_t* dst, const hipgraph_type_erased_device_array_view_t* weights, const hipgraph_type_erased_device_array_view_t* edge_ids, const hipgraph_type_erased_device_array_view_t* edge_type_ids, bool store_transposed, bool renumber, bool drop_self_loops, bool drop_multi_edges, bool symmetrize, bool do_expensive_check, hipgraph_graph_t** graph, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_graph_create_sg((const cugraph_resource_handle_t*)handle, (const cugraph_graph_properties_t*)properties, (const cugraph_type_erased_device_array_view_t*)vertices, (const cugraph_type_erased_device_array_view_t*)src, (const cugraph_type_erased_device_array_view_t*)dst, (const cugraph_type_erased_device_array_view_t*)weights, (const cugraph_type_erased_device_array_view_t*)edge_ids, (const cugraph_type_erased_device_array_view_t*)edge_type_ids, (bool_t)store_transposed, (bool_t)renumber, (bool_t)drop_self_loops, (bool_t)drop_multi_edges, (bool_t)symmetrize, (bool_t)do_expensive_check, (cugraph_graph_t**)graph, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_graph_create_sg_from_csr(const hipgraph_resource_handle_t* handle, const hipgraph_graph_properties_t* properties, const hipgraph_type_erased_device_array_view_t* offsets, const hipgraph_type_erased_device_array_view_t* indices, const hipgraph_type_erased_device_array_view_t* weights, const hipgraph_type_erased_device_array_view_t* edge_ids, const hipgraph_type_erased_device_array_view_t* edge_type_ids, bool store_transposed, bool renumber, bool symmetrize, bool do_expensive_check, hipgraph_graph_t** graph, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_graph_create_sg_from_csr((const cugraph_resource_handle_t*)handle, (const cugraph_graph_properties_t*)properties, (const cugraph_type_erased_device_array_view_t*)offsets, (const cugraph_type_erased_device_array_view_t*)indices, (const cugraph_type_erased_device_array_view_t*)weights, (const cugraph_type_erased_device_array_view_t*)edge_ids, (const cugraph_type_erased_device_array_view_t*)edge_type_ids, (bool_t)store_transposed, (bool_t)renumber, (bool_t)symmetrize, (bool_t)do_expensive_check, (cugraph_graph_t**)graph, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_graph_create_mg(hipgraph_resource_handle_t const* handle, hipgraph_graph_properties_t const* properties, hipgraph_type_erased_device_array_view_t const* const* vertices, hipgraph_type_erased_device_array_view_t const* const* src, hipgraph_type_erased_device_array_view_t const* const* dst, hipgraph_type_erased_device_array_view_t const* const* weights, hipgraph_type_erased_device_array_view_t const* const* edge_ids, hipgraph_type_erased_device_array_view_t const* const* edge_type_ids, bool store_transposed, size_t num_arrays, bool drop_self_loops, bool drop_multi_edges, bool symmetrize, bool do_expensive_check, hipgraph_graph_t** graph, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_graph_create_mg((cugraph_resource_handle_t const*)handle, (cugraph_graph_properties_t const*)properties, (cugraph_type_erased_device_array_view_t const* const*)vertices, (cugraph_type_erased_device_array_view_t const* const*)src, (cugraph_type_erased_device_array_view_t const* const*)dst, (cugraph_type_erased_device_array_view_t const* const*)weights, (cugraph_type_erased_device_array_view_t const* const*)edge_ids, (cugraph_type_erased_device_array_view_t const* const*)edge_type_ids, (bool_t)store_transposed, num_arrays, (bool_t)drop_self_loops, (bool_t)drop_multi_edges, (bool_t)symmetrize, (bool_t)do_expensive_check, (cugraph_graph_t**)graph, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT void hipgraph_graph_free(hipgraph_graph_t* graph)
{
    cugraph_graph_free((cugraph_graph_t*)graph);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_data_mask_create(const hipgraph_resource_handle_t* handle, const hipgraph_type_erased_device_array_view_t* vertex_bit_mask, const hipgraph_type_erased_device_array_view_t* edge_bit_mask, bool complement, hipgraph_data_mask_t** mask, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_data_mask_create((const cugraph_resource_handle_t*)handle, (const cugraph_type_erased_device_array_view_t*)vertex_bit_mask, (const cugraph_type_erased_device_array_view_t*)edge_bit_mask, (bool_t)complement, (cugraph_data_mask_t**)mask, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_graph_get_data_mask(hipgraph_graph_t* graph, hipgraph_data_mask_t** mask, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_graph_get_data_mask((cugraph_graph_t*)graph, (cugraph_data_mask_t**)mask, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_graph_add_data_mask(hipgraph_graph_t* graph, hipgraph_data_mask_t* mask, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_graph_add_data_mask((cugraph_graph_t*)graph, (cugraph_data_mask_t*)mask, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_graph_release_data_mask(hipgraph_graph_t* graph, hipgraph_data_mask_t** mask, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_graph_release_data_mask((cugraph_graph_t*)graph, (cugraph_data_mask_t**)mask, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT void hipgraph_data_mask_destroy(hipgraph_data_mask_t* mask)
{
    cugraph_data_mask_destroy((cugraph_data_mask_t*)mask);
}




