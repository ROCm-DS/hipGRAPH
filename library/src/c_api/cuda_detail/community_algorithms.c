// -*- C -*-
// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common.h"
#include <cugraph_c/error.h>
#include <hipgraph_c/error.h>
#include <cugraph_c/graph.h>
#include <hipgraph_c/graph.h>
#include <cugraph_c/graph_functions.h>
#include <hipgraph_c/graph_functions.h>
#include <cugraph_c/random.h>
#include <hipgraph_c/random.h>
#include <cugraph_c/resource_handle.h>
#include <hipgraph_c/resource_handle.h>
#include <cugraph_c/hipgraph/hipgraph-common.h>
#include <hipgraph_c/hipgraph/hipgraph-common.h>
#include "error_code.inl.h"

HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_triangle_count(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* start, bool do_expensive_check, hipgraph_triangle_count_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_triangle_count((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)start, (bool_t)do_expensive_check, (cugraph_triangle_count_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_triangle_count_result_get_vertices(hipgraph_triangle_count_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_triangle_count_result_get_vertices((cugraph_triangle_count_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_triangle_count_result_get_counts(hipgraph_triangle_count_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_triangle_count_result_get_counts((cugraph_triangle_count_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT void hipgraph_triangle_count_result_free(hipgraph_triangle_count_result_t* result)
{
    cugraph_triangle_count_result_free((cugraph_triangle_count_result_t*)result);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_louvain(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, size_t max_level, double threshold, double resolution, bool do_expensive_check, hipgraph_hierarchical_clustering_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_louvain((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, max_level, threshold, resolution, (bool_t)do_expensive_check, (cugraph_hierarchical_clustering_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_leiden(const hipgraph_resource_handle_t* handle, hipgraph_rng_state_t* rng_state, hipgraph_graph_t* graph, size_t max_level, double resolution, double theta, bool do_expensive_check, hipgraph_hierarchical_clustering_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_leiden((const cugraph_resource_handle_t*)handle, (cugraph_rng_state_t*)rng_state, (cugraph_graph_t*)graph, max_level, resolution, theta, (bool_t)do_expensive_check, (cugraph_hierarchical_clustering_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_hierarchical_clustering_result_get_vertices(hipgraph_hierarchical_clustering_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_hierarchical_clustering_result_get_vertices((cugraph_hierarchical_clustering_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_hierarchical_clustering_result_get_clusters(hipgraph_hierarchical_clustering_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_hierarchical_clustering_result_get_clusters((cugraph_hierarchical_clustering_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT double hipgraph_hierarchical_clustering_result_get_modularity(hipgraph_hierarchical_clustering_result_t* result)
{
    double out;
    out = cugraph_hierarchical_clustering_result_get_modularity((cugraph_hierarchical_clustering_result_t*)result);
    return (double)out;
}


HIPGRAPH_EXPORT void hipgraph_hierarchical_clustering_result_free(hipgraph_hierarchical_clustering_result_t* result)
{
    cugraph_hierarchical_clustering_result_free((cugraph_hierarchical_clustering_result_t*)result);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_ecg(const hipgraph_resource_handle_t* handle, hipgraph_rng_state_t* rng_state, hipgraph_graph_t* graph, double min_weight, size_t ensemble_size, size_t max_level, double threshold, double resolution, bool do_expensive_check, hipgraph_hierarchical_clustering_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_ecg((const cugraph_resource_handle_t*)handle, (cugraph_rng_state_t*)rng_state, (cugraph_graph_t*)graph, min_weight, ensemble_size, max_level, threshold, resolution, (bool_t)do_expensive_check, (cugraph_hierarchical_clustering_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_extract_ego(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* source_vertices, size_t radius, bool do_expensive_check, hipgraph_induced_subgraph_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_extract_ego((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)source_vertices, radius, (bool_t)do_expensive_check, (cugraph_induced_subgraph_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_k_truss_subgraph(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, size_t k, bool do_expensive_check, hipgraph_induced_subgraph_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_k_truss_subgraph((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, k, (bool_t)do_expensive_check, (cugraph_induced_subgraph_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_balanced_cut_clustering(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, size_t n_clusters, size_t n_eigenvectors, double evs_tolerance, int evs_max_iterations, double k_means_tolerance, int k_means_max_iterations, bool do_expensive_check, hipgraph_clustering_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_balanced_cut_clustering((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, n_clusters, n_eigenvectors, evs_tolerance, evs_max_iterations, k_means_tolerance, k_means_max_iterations, (bool_t)do_expensive_check, (cugraph_clustering_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_spectral_modularity_maximization(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, size_t n_clusters, size_t n_eigenvectors, double evs_tolerance, int evs_max_iterations, double k_means_tolerance, int k_means_max_iterations, bool do_expensive_check, hipgraph_clustering_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_spectral_modularity_maximization((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, n_clusters, n_eigenvectors, evs_tolerance, evs_max_iterations, k_means_tolerance, k_means_max_iterations, (bool_t)do_expensive_check, (cugraph_clustering_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_analyze_clustering_modularity(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, size_t n_clusters, const hipgraph_type_erased_device_array_view_t* vertices, const hipgraph_type_erased_device_array_view_t* clusters, double* score, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_analyze_clustering_modularity((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, n_clusters, (const cugraph_type_erased_device_array_view_t*)vertices, (const cugraph_type_erased_device_array_view_t*)clusters, score, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_analyze_clustering_edge_cut(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, size_t n_clusters, const hipgraph_type_erased_device_array_view_t* vertices, const hipgraph_type_erased_device_array_view_t* clusters, double* score, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_analyze_clustering_edge_cut((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, n_clusters, (const cugraph_type_erased_device_array_view_t*)vertices, (const cugraph_type_erased_device_array_view_t*)clusters, score, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_analyze_clustering_ratio_cut(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, size_t n_clusters, const hipgraph_type_erased_device_array_view_t* vertices, const hipgraph_type_erased_device_array_view_t* clusters, double* score, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_analyze_clustering_ratio_cut((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, n_clusters, (const cugraph_type_erased_device_array_view_t*)vertices, (const cugraph_type_erased_device_array_view_t*)clusters, score, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_clustering_result_get_vertices(hipgraph_clustering_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_clustering_result_get_vertices((cugraph_clustering_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_clustering_result_get_clusters(hipgraph_clustering_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_clustering_result_get_clusters((cugraph_clustering_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT void hipgraph_clustering_result_free(hipgraph_clustering_result_t* result)
{
    cugraph_clustering_result_free((cugraph_clustering_result_t*)result);
}




