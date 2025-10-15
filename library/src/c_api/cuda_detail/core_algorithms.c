// -*- C -*-
// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common.h"
#include <cugraph_c/array.h>
#include <hipgraph_c/array.h>
#include <cugraph_c/error.h>
#include <hipgraph_c/error.h>
#include <cugraph_c/graph.h>
#include <hipgraph_c/graph.h>
#include <cugraph_c/resource_handle.h>
#include <hipgraph_c/resource_handle.h>
#include <cugraph_c/hipgraph/hipgraph-common.h>
#include <hipgraph_c/hipgraph/hipgraph-common.h>
#include "error_code.inl.h"

HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_core_result_create(const hipgraph_resource_handle_t* handle, hipgraph_type_erased_device_array_view_t* vertices, hipgraph_type_erased_device_array_view_t* core_numbers, hipgraph_core_result_t** core_result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_core_result_create((const cugraph_resource_handle_t*)handle, (cugraph_type_erased_device_array_view_t*)vertices, (cugraph_type_erased_device_array_view_t*)core_numbers, (cugraph_core_result_t**)core_result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_core_result_get_vertices(hipgraph_core_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_core_result_get_vertices((cugraph_core_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_core_result_get_core_numbers(hipgraph_core_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_core_result_get_core_numbers((cugraph_core_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT void hipgraph_core_result_free(hipgraph_core_result_t* result)
{
    cugraph_core_result_free((cugraph_core_result_t*)result);
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_k_core_result_get_src_vertices(hipgraph_k_core_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_k_core_result_get_src_vertices((cugraph_k_core_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_k_core_result_get_dst_vertices(hipgraph_k_core_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_k_core_result_get_dst_vertices((cugraph_k_core_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_k_core_result_get_weights(hipgraph_k_core_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_k_core_result_get_weights((cugraph_k_core_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT void hipgraph_k_core_result_free(hipgraph_k_core_result_t* result)
{
    cugraph_k_core_result_free((cugraph_k_core_result_t*)result);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_core_number(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, hipgraph_k_core_degree_type_t degree_type, bool do_expensive_check, hipgraph_core_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_core_number((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, _hipgraph_to_cugraph_k_core_degree_type_t(degree_type), (bool_t)do_expensive_check, (cugraph_core_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_k_core(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, size_t k, hipgraph_k_core_degree_type_t degree_type, const hipgraph_core_result_t* core_result, bool do_expensive_check, hipgraph_k_core_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_k_core((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, k, _hipgraph_to_cugraph_k_core_degree_type_t(degree_type), (const cugraph_core_result_t*)core_result, (bool_t)do_expensive_check, (cugraph_k_core_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}




