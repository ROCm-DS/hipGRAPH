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
#include <cugraph_c/random.h>
#include <hipgraph_c/random.h>
#include <cugraph_c/resource_handle.h>
#include <hipgraph_c/resource_handle.h>
#include <cugraph_c/hipgraph/hipgraph-common.h>
#include <hipgraph_c/hipgraph/hipgraph-common.h>
#include "error_code.inl.h"

HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_centrality_result_get_vertices(hipgraph_centrality_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_centrality_result_get_vertices((cugraph_centrality_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_centrality_result_get_values(hipgraph_centrality_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_centrality_result_get_values((cugraph_centrality_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT size_t hipgraph_centrality_result_get_num_iterations(hipgraph_centrality_result_t* result)
{
    size_t out;
    out = cugraph_centrality_result_get_num_iterations((cugraph_centrality_result_t*)result);
    return (size_t)out;
}


HIPGRAPH_EXPORT bool_t hipgraph_centrality_result_converged(hipgraph_centrality_result_t* result)
{
    bool_t out;
    out = cugraph_centrality_result_converged((cugraph_centrality_result_t*)result);
    return (bool_t)out;
}


HIPGRAPH_EXPORT void hipgraph_centrality_result_free(hipgraph_centrality_result_t* result)
{
    cugraph_centrality_result_free((cugraph_centrality_result_t*)result);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_pagerank(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices, const hipgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums, const hipgraph_type_erased_device_array_view_t* initial_guess_vertices, const hipgraph_type_erased_device_array_view_t* initial_guess_values, double alpha, double epsilon, size_t max_iterations, bool do_expensive_check, hipgraph_centrality_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_pagerank((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)precomputed_vertex_out_weight_vertices, (const cugraph_type_erased_device_array_view_t*)precomputed_vertex_out_weight_sums, (const cugraph_type_erased_device_array_view_t*)initial_guess_vertices, (const cugraph_type_erased_device_array_view_t*)initial_guess_values, alpha, epsilon, max_iterations, (bool_t)do_expensive_check, (cugraph_centrality_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_pagerank_allow_nonconvergence(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices, const hipgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums, const hipgraph_type_erased_device_array_view_t* initial_guess_vertices, const hipgraph_type_erased_device_array_view_t* initial_guess_values, double alpha, double epsilon, size_t max_iterations, bool do_expensive_check, hipgraph_centrality_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_pagerank_allow_nonconvergence((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)precomputed_vertex_out_weight_vertices, (const cugraph_type_erased_device_array_view_t*)precomputed_vertex_out_weight_sums, (const cugraph_type_erased_device_array_view_t*)initial_guess_vertices, (const cugraph_type_erased_device_array_view_t*)initial_guess_values, alpha, epsilon, max_iterations, (bool_t)do_expensive_check, (cugraph_centrality_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_personalized_pagerank(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices, const hipgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums, const hipgraph_type_erased_device_array_view_t* initial_guess_vertices, const hipgraph_type_erased_device_array_view_t* initial_guess_values, const hipgraph_type_erased_device_array_view_t* personalization_vertices, const hipgraph_type_erased_device_array_view_t* personalization_values, double alpha, double epsilon, size_t max_iterations, bool do_expensive_check, hipgraph_centrality_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_personalized_pagerank((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)precomputed_vertex_out_weight_vertices, (const cugraph_type_erased_device_array_view_t*)precomputed_vertex_out_weight_sums, (const cugraph_type_erased_device_array_view_t*)initial_guess_vertices, (const cugraph_type_erased_device_array_view_t*)initial_guess_values, (const cugraph_type_erased_device_array_view_t*)personalization_vertices, (const cugraph_type_erased_device_array_view_t*)personalization_values, alpha, epsilon, max_iterations, (bool_t)do_expensive_check, (cugraph_centrality_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_personalized_pagerank_allow_nonconvergence(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices, const hipgraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums, const hipgraph_type_erased_device_array_view_t* initial_guess_vertices, const hipgraph_type_erased_device_array_view_t* initial_guess_values, const hipgraph_type_erased_device_array_view_t* personalization_vertices, const hipgraph_type_erased_device_array_view_t* personalization_values, double alpha, double epsilon, size_t max_iterations, bool do_expensive_check, hipgraph_centrality_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_personalized_pagerank_allow_nonconvergence((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)precomputed_vertex_out_weight_vertices, (const cugraph_type_erased_device_array_view_t*)precomputed_vertex_out_weight_sums, (const cugraph_type_erased_device_array_view_t*)initial_guess_vertices, (const cugraph_type_erased_device_array_view_t*)initial_guess_values, (const cugraph_type_erased_device_array_view_t*)personalization_vertices, (const cugraph_type_erased_device_array_view_t*)personalization_values, alpha, epsilon, max_iterations, (bool_t)do_expensive_check, (cugraph_centrality_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_eigenvector_centrality(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, double epsilon, size_t max_iterations, bool do_expensive_check, hipgraph_centrality_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_eigenvector_centrality((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, epsilon, max_iterations, (bool_t)do_expensive_check, (cugraph_centrality_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_katz_centrality(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* betas, double alpha, double beta, double epsilon, size_t max_iterations, bool do_expensive_check, hipgraph_centrality_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_katz_centrality((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)betas, alpha, beta, epsilon, max_iterations, (bool_t)do_expensive_check, (cugraph_centrality_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_betweenness_centrality(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* vertex_list, bool normalized, bool include_endpoints, bool do_expensive_check, hipgraph_centrality_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_betweenness_centrality((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)vertex_list, (bool_t)normalized, (bool_t)include_endpoints, (bool_t)do_expensive_check, (cugraph_centrality_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_edge_centrality_result_get_src_vertices(hipgraph_edge_centrality_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_edge_centrality_result_get_src_vertices((cugraph_edge_centrality_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_edge_centrality_result_get_dst_vertices(hipgraph_edge_centrality_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_edge_centrality_result_get_dst_vertices((cugraph_edge_centrality_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_edge_centrality_result_get_edge_ids(hipgraph_edge_centrality_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_edge_centrality_result_get_edge_ids((cugraph_edge_centrality_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_edge_centrality_result_get_values(hipgraph_edge_centrality_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_edge_centrality_result_get_values((cugraph_edge_centrality_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT void hipgraph_edge_centrality_result_free(hipgraph_edge_centrality_result_t* result)
{
    cugraph_edge_centrality_result_free((cugraph_edge_centrality_result_t*)result);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_edge_betweenness_centrality(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* vertex_list, bool normalized, bool do_expensive_check, hipgraph_edge_centrality_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_edge_betweenness_centrality((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)vertex_list, (bool_t)normalized, (bool_t)do_expensive_check, (cugraph_edge_centrality_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_hits_result_get_vertices(hipgraph_hits_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_hits_result_get_vertices((cugraph_hits_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_hits_result_get_hubs(hipgraph_hits_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_hits_result_get_hubs((cugraph_hits_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_hits_result_get_authorities(hipgraph_hits_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_hits_result_get_authorities((cugraph_hits_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT double hipgraph_hits_result_get_hub_score_differences(hipgraph_hits_result_t* result)
{
    double out;
    out = cugraph_hits_result_get_hub_score_differences((cugraph_hits_result_t*)result);
    return (double)out;
}


HIPGRAPH_EXPORT size_t hipgraph_hits_result_get_number_of_iterations(hipgraph_hits_result_t* result)
{
    size_t out;
    out = cugraph_hits_result_get_number_of_iterations((cugraph_hits_result_t*)result);
    return (size_t)out;
}


HIPGRAPH_EXPORT void hipgraph_hits_result_free(hipgraph_hits_result_t* result)
{
    cugraph_hits_result_free((cugraph_hits_result_t*)result);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_hits(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, double epsilon, size_t max_iterations, const hipgraph_type_erased_device_array_view_t* initial_hubs_guess_vertices, const hipgraph_type_erased_device_array_view_t* initial_hubs_guess_values, bool normalize, bool do_expensive_check, hipgraph_hits_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_hits((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, epsilon, max_iterations, (const cugraph_type_erased_device_array_view_t*)initial_hubs_guess_vertices, (const cugraph_type_erased_device_array_view_t*)initial_hubs_guess_values, (bool_t)normalize, (bool_t)do_expensive_check, (cugraph_hits_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}




