// -*- C -*-
// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common.h"
#include <cugraph_c/array.h>
#include <hipgraph_c/array.h>
#include <cugraph_c/graph.h>
#include <hipgraph_c/graph.h>
#include <cugraph_c/resource_handle.h>
#include <hipgraph_c/resource_handle.h>
#include <cugraph_c/hipgraph/hipgraph-common.h>
#include <hipgraph_c/hipgraph/hipgraph-common.h>
#include "error_code.inl.h"

HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_build_edge_id_and_type_to_src_dst_lookup_map(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, hipgraph_lookup_container_t** lookup_container, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_build_edge_id_and_type_to_src_dst_lookup_map((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (cugraph_lookup_container_t**)lookup_container, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_lookup_endpoints_from_edge_ids_and_single_type(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_lookup_container_t* lookup_container, const hipgraph_type_erased_device_array_view_t* edge_ids_to_lookup, int edge_type_to_lookup, hipgraph_lookup_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_lookup_endpoints_from_edge_ids_and_single_type((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_lookup_container_t*)lookup_container, (const cugraph_type_erased_device_array_view_t*)edge_ids_to_lookup, edge_type_to_lookup, (cugraph_lookup_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_lookup_endpoints_from_edge_ids_and_types(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_lookup_container_t* lookup_container, const hipgraph_type_erased_device_array_view_t* edge_ids_to_lookup, const hipgraph_type_erased_device_array_view_t* edge_types_to_lookup, hipgraph_lookup_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_lookup_endpoints_from_edge_ids_and_types((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_lookup_container_t*)lookup_container, (const cugraph_type_erased_device_array_view_t*)edge_ids_to_lookup, (const cugraph_type_erased_device_array_view_t*)edge_types_to_lookup, (cugraph_lookup_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_lookup_result_get_srcs(const hipgraph_lookup_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_lookup_result_get_srcs((const cugraph_lookup_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_lookup_result_get_dsts(const hipgraph_lookup_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_lookup_result_get_dsts((const cugraph_lookup_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT void hipgraph_lookup_result_free(hipgraph_lookup_result_t* result)
{
    cugraph_lookup_result_free((cugraph_lookup_result_t*)result);
}


HIPGRAPH_EXPORT void hipgraph_lookup_container_free(hipgraph_lookup_container_t* container)
{
    cugraph_lookup_container_free((cugraph_lookup_container_t*)container);
}




