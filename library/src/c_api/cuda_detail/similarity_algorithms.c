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
#include <cugraph_c/graph_functions.h>
#include <hipgraph_c/graph_functions.h>
#include <cugraph_c/resource_handle.h>
#include <hipgraph_c/resource_handle.h>
#include <cugraph_c/hipgraph/hipgraph-common.h>
#include <hipgraph_c/hipgraph/hipgraph-common.h>
#include "error_code.inl.h"

HIPGRAPH_EXPORT hipgraph_vertex_pairs_t* hipgraph_similarity_result_get_vertex_pairs(hipgraph_similarity_result_t* result)
{
    cugraph_vertex_pairs_t* out;
    out = cugraph_similarity_result_get_vertex_pairs((cugraph_similarity_result_t*)result);
    return (hipgraph_vertex_pairs_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_similarity_result_get_similarity(hipgraph_similarity_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_similarity_result_get_similarity((cugraph_similarity_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT void hipgraph_similarity_result_free(hipgraph_similarity_result_t* result)
{
    cugraph_similarity_result_free((cugraph_similarity_result_t*)result);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_jaccard_coefficients(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_vertex_pairs_t* vertex_pairs, bool use_weight, bool do_expensive_check, hipgraph_similarity_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_jaccard_coefficients((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_vertex_pairs_t*)vertex_pairs, (bool_t)use_weight, (bool_t)do_expensive_check, (cugraph_similarity_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_sorensen_coefficients(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_vertex_pairs_t* vertex_pairs, bool use_weight, bool do_expensive_check, hipgraph_similarity_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_sorensen_coefficients((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_vertex_pairs_t*)vertex_pairs, (bool_t)use_weight, (bool_t)do_expensive_check, (cugraph_similarity_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_overlap_coefficients(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_vertex_pairs_t* vertex_pairs, bool use_weight, bool do_expensive_check, hipgraph_similarity_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_overlap_coefficients((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_vertex_pairs_t*)vertex_pairs, (bool_t)use_weight, (bool_t)do_expensive_check, (cugraph_similarity_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_cosine_similarity_coefficients(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_vertex_pairs_t* vertex_pairs, bool use_weight, bool do_expensive_check, hipgraph_similarity_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_cosine_similarity_coefficients((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_vertex_pairs_t*)vertex_pairs, (bool_t)use_weight, (bool_t)do_expensive_check, (cugraph_similarity_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_all_pairs_jaccard_coefficients(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* vertices, bool use_weight, size_t topk, bool do_expensive_check, hipgraph_similarity_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_all_pairs_jaccard_coefficients((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)vertices, (bool_t)use_weight, topk, (bool_t)do_expensive_check, (cugraph_similarity_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_all_pairs_sorensen_coefficients(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* vertices, bool use_weight, size_t topk, bool do_expensive_check, hipgraph_similarity_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_all_pairs_sorensen_coefficients((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)vertices, (bool_t)use_weight, topk, (bool_t)do_expensive_check, (cugraph_similarity_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_all_pairs_overlap_coefficients(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* vertices, bool use_weight, size_t topk, bool do_expensive_check, hipgraph_similarity_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_all_pairs_overlap_coefficients((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)vertices, (bool_t)use_weight, topk, (bool_t)do_expensive_check, (cugraph_similarity_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_all_pairs_cosine_similarity_coefficients(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* vertices, bool use_weight, size_t topk, bool do_expensive_check, hipgraph_similarity_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_all_pairs_cosine_similarity_coefficients((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)vertices, (bool_t)use_weight, topk, (bool_t)do_expensive_check, (cugraph_similarity_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}




