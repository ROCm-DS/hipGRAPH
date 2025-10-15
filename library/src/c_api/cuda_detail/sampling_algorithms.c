// -*- C -*-
// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common.h"
#include <cugraph_c/coo.h>
#include <hipgraph_c/coo.h>
#include <cugraph_c/error.h>
#include <hipgraph_c/error.h>
#include <cugraph_c/graph.h>
#include <hipgraph_c/graph.h>
#include <cugraph_c/properties.h>
#include <hipgraph_c/properties.h>
#include <cugraph_c/random.h>
#include <hipgraph_c/random.h>
#include <cugraph_c/resource_handle.h>
#include <hipgraph_c/resource_handle.h>
#include <cugraph_c/hipgraph/hipgraph-common.h>
#include <hipgraph_c/hipgraph/hipgraph-common.h>
#include "error_code.inl.h"

HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_uniform_random_walks(const hipgraph_resource_handle_t* handle, hipgraph_rng_state_t* rng_state, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* start_vertices, size_t max_length, hipgraph_random_walk_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_uniform_random_walks((const cugraph_resource_handle_t*)handle, (cugraph_rng_state_t*)rng_state, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)start_vertices, max_length, (cugraph_random_walk_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_biased_random_walks(const hipgraph_resource_handle_t* handle, hipgraph_rng_state_t* rng_state, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* start_vertices, size_t max_length, hipgraph_random_walk_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_biased_random_walks((const cugraph_resource_handle_t*)handle, (cugraph_rng_state_t*)rng_state, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)start_vertices, max_length, (cugraph_random_walk_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_node2vec_random_walks(const hipgraph_resource_handle_t* handle, hipgraph_rng_state_t* rng_state, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* start_vertices, size_t max_length, double p, double q, hipgraph_random_walk_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_node2vec_random_walks((const cugraph_resource_handle_t*)handle, (cugraph_rng_state_t*)rng_state, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)start_vertices, max_length, p, q, (cugraph_random_walk_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_node2vec(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* sources, size_t max_depth, bool compress_result, double p, double q, hipgraph_random_walk_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_node2vec((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)sources, max_depth, (bool_t)compress_result, p, q, (cugraph_random_walk_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT size_t hipgraph_random_walk_result_get_max_path_length(hipgraph_random_walk_result_t* result)
{
    size_t out;
    out = cugraph_random_walk_result_get_max_path_length((cugraph_random_walk_result_t*)result);
    return (size_t)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_random_walk_result_get_paths(hipgraph_random_walk_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_random_walk_result_get_paths((cugraph_random_walk_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_random_walk_result_get_weights(hipgraph_random_walk_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_random_walk_result_get_weights((cugraph_random_walk_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_random_walk_result_get_path_sizes(hipgraph_random_walk_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_random_walk_result_get_path_sizes((cugraph_random_walk_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT void hipgraph_random_walk_result_free(hipgraph_random_walk_result_t* result)
{
    cugraph_random_walk_result_free((cugraph_random_walk_result_t*)result);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_sampling_options_create(hipgraph_sampling_options_t** options, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_sampling_options_create((cugraph_sampling_options_t**)options, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT void hipgraph_sampling_set_retain_seeds(hipgraph_sampling_options_t* options, bool value)
{
    cugraph_sampling_set_retain_seeds((cugraph_sampling_options_t*)options, (bool_t)value);
}


HIPGRAPH_EXPORT void hipgraph_sampling_set_renumber_results(hipgraph_sampling_options_t* options, bool value)
{
    cugraph_sampling_set_renumber_results((cugraph_sampling_options_t*)options, (bool_t)value);
}


HIPGRAPH_EXPORT void hipgraph_sampling_set_compress_per_hop(hipgraph_sampling_options_t* options, bool value)
{
    cugraph_sampling_set_compress_per_hop((cugraph_sampling_options_t*)options, (bool_t)value);
}


HIPGRAPH_EXPORT void hipgraph_sampling_set_with_replacement(hipgraph_sampling_options_t* options, bool value)
{
    cugraph_sampling_set_with_replacement((cugraph_sampling_options_t*)options, (bool_t)value);
}


HIPGRAPH_EXPORT void hipgraph_sampling_set_return_hops(hipgraph_sampling_options_t* options, bool value)
{
    cugraph_sampling_set_return_hops((cugraph_sampling_options_t*)options, (bool_t)value);
}


HIPGRAPH_EXPORT void hipgraph_sampling_set_compression_type(hipgraph_sampling_options_t* options, hipgraph_compression_type_t value)
{
    cugraph_sampling_set_compression_type((cugraph_sampling_options_t*)options, _hipgraph_to_cugraph_compression_type_t(value));
}


HIPGRAPH_EXPORT void hipgraph_sampling_set_prior_sources_behavior(hipgraph_sampling_options_t* options, hipgraph_prior_sources_behavior_t value)
{
    cugraph_sampling_set_prior_sources_behavior((cugraph_sampling_options_t*)options, _hipgraph_to_cugraph_prior_sources_behavior_t(value));
}


HIPGRAPH_EXPORT void hipgraph_sampling_set_dedupe_sources(hipgraph_sampling_options_t* options, bool value)
{
    cugraph_sampling_set_dedupe_sources((cugraph_sampling_options_t*)options, (bool_t)value);
}


HIPGRAPH_EXPORT void hipgraph_sampling_options_free(hipgraph_sampling_options_t* options)
{
    cugraph_sampling_options_free((cugraph_sampling_options_t*)options);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_uniform_neighbor_sample(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* start_vertices, const hipgraph_type_erased_device_array_view_t* start_vertex_labels, const hipgraph_type_erased_device_array_view_t* label_list, const hipgraph_type_erased_device_array_view_t* label_to_comm_rank, const hipgraph_type_erased_device_array_view_t* label_offsets, const hipgraph_type_erased_host_array_view_t* fan_out, hipgraph_rng_state_t* rng_state, const hipgraph_sampling_options_t* options, bool do_expensive_check, hipgraph_sample_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_uniform_neighbor_sample((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)start_vertices, (const cugraph_type_erased_device_array_view_t*)start_vertex_labels, (const cugraph_type_erased_device_array_view_t*)label_list, (const cugraph_type_erased_device_array_view_t*)label_to_comm_rank, (const cugraph_type_erased_device_array_view_t*)label_offsets, (const cugraph_type_erased_host_array_view_t*)fan_out, (cugraph_rng_state_t*)rng_state, (const cugraph_sampling_options_t*)options, (bool_t)do_expensive_check, (cugraph_sample_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_biased_neighbor_sample(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_edge_property_view_t* edge_biases, const hipgraph_type_erased_device_array_view_t* start_vertices, const hipgraph_type_erased_device_array_view_t* start_vertex_labels, const hipgraph_type_erased_device_array_view_t* label_list, const hipgraph_type_erased_device_array_view_t* label_to_comm_rank, const hipgraph_type_erased_device_array_view_t* label_offsets, const hipgraph_type_erased_host_array_view_t* fan_out, hipgraph_rng_state_t* rng_state, const hipgraph_sampling_options_t* options, bool do_expensive_check, hipgraph_sample_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_biased_neighbor_sample((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_edge_property_view_t*)edge_biases, (const cugraph_type_erased_device_array_view_t*)start_vertices, (const cugraph_type_erased_device_array_view_t*)start_vertex_labels, (const cugraph_type_erased_device_array_view_t*)label_list, (const cugraph_type_erased_device_array_view_t*)label_to_comm_rank, (const cugraph_type_erased_device_array_view_t*)label_offsets, (const cugraph_type_erased_host_array_view_t*)fan_out, (cugraph_rng_state_t*)rng_state, (const cugraph_sampling_options_t*)options, (bool_t)do_expensive_check, (cugraph_sample_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_homogeneous_uniform_neighbor_sample(const hipgraph_resource_handle_t* handle, hipgraph_rng_state_t* rng_state, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* start_vertices, const hipgraph_type_erased_device_array_view_t* starting_vertex_label_offsets, const hipgraph_type_erased_host_array_view_t* fan_out, const hipgraph_sampling_options_t* options, bool do_expensive_check, hipgraph_sample_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_homogeneous_uniform_neighbor_sample((const cugraph_resource_handle_t*)handle, (cugraph_rng_state_t*)rng_state, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)start_vertices, (const cugraph_type_erased_device_array_view_t*)starting_vertex_label_offsets, (const cugraph_type_erased_host_array_view_t*)fan_out, (const cugraph_sampling_options_t*)options, (bool_t)do_expensive_check, (cugraph_sample_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_homogeneous_biased_neighbor_sample(const hipgraph_resource_handle_t* handle, hipgraph_rng_state_t* rng_state, hipgraph_graph_t* graph, const hipgraph_edge_property_view_t* edge_biases, const hipgraph_type_erased_device_array_view_t* start_vertices, const hipgraph_type_erased_device_array_view_t* starting_vertex_label_offsets, const hipgraph_type_erased_host_array_view_t* fan_out, const hipgraph_sampling_options_t* options, bool do_expensive_check, hipgraph_sample_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_homogeneous_biased_neighbor_sample((const cugraph_resource_handle_t*)handle, (cugraph_rng_state_t*)rng_state, (cugraph_graph_t*)graph, (const cugraph_edge_property_view_t*)edge_biases, (const cugraph_type_erased_device_array_view_t*)start_vertices, (const cugraph_type_erased_device_array_view_t*)starting_vertex_label_offsets, (const cugraph_type_erased_host_array_view_t*)fan_out, (const cugraph_sampling_options_t*)options, (bool_t)do_expensive_check, (cugraph_sample_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_heterogeneous_uniform_neighbor_sample(const hipgraph_resource_handle_t* handle, hipgraph_rng_state_t* rng_state, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* start_vertices, const hipgraph_type_erased_device_array_view_t* starting_vertex_label_offsets, const hipgraph_type_erased_device_array_view_t* vertex_type_offsets, const hipgraph_type_erased_host_array_view_t* fan_out, int num_edge_types, const hipgraph_sampling_options_t* options, bool do_expensive_check, hipgraph_sample_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_heterogeneous_uniform_neighbor_sample((const cugraph_resource_handle_t*)handle, (cugraph_rng_state_t*)rng_state, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)start_vertices, (const cugraph_type_erased_device_array_view_t*)starting_vertex_label_offsets, (const cugraph_type_erased_device_array_view_t*)vertex_type_offsets, (const cugraph_type_erased_host_array_view_t*)fan_out, num_edge_types, (const cugraph_sampling_options_t*)options, (bool_t)do_expensive_check, (cugraph_sample_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_heterogeneous_biased_neighbor_sample(const hipgraph_resource_handle_t* handle, hipgraph_rng_state_t* rng_state, hipgraph_graph_t* graph, const hipgraph_edge_property_view_t* edge_biases, const hipgraph_type_erased_device_array_view_t* start_vertices, const hipgraph_type_erased_device_array_view_t* starting_vertex_label_offsets, const hipgraph_type_erased_device_array_view_t* vertex_type_offsets, const hipgraph_type_erased_host_array_view_t* fan_out, int num_edge_types, const hipgraph_sampling_options_t* options, bool do_expensive_check, hipgraph_sample_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_heterogeneous_biased_neighbor_sample((const cugraph_resource_handle_t*)handle, (cugraph_rng_state_t*)rng_state, (cugraph_graph_t*)graph, (const cugraph_edge_property_view_t*)edge_biases, (const cugraph_type_erased_device_array_view_t*)start_vertices, (const cugraph_type_erased_device_array_view_t*)starting_vertex_label_offsets, (const cugraph_type_erased_device_array_view_t*)vertex_type_offsets, (const cugraph_type_erased_host_array_view_t*)fan_out, num_edge_types, (const cugraph_sampling_options_t*)options, (bool_t)do_expensive_check, (cugraph_sample_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_sources(const hipgraph_sample_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_sample_result_get_sources((const cugraph_sample_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_destinations(const hipgraph_sample_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_sample_result_get_destinations((const cugraph_sample_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_majors(const hipgraph_sample_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_sample_result_get_majors((const cugraph_sample_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_minors(const hipgraph_sample_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_sample_result_get_minors((const cugraph_sample_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_major_offsets(const hipgraph_sample_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_sample_result_get_major_offsets((const cugraph_sample_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_start_labels(const hipgraph_sample_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_sample_result_get_start_labels((const cugraph_sample_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_edge_id(const hipgraph_sample_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_sample_result_get_edge_id((const cugraph_sample_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_edge_type(const hipgraph_sample_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_sample_result_get_edge_type((const cugraph_sample_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_edge_weight(const hipgraph_sample_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_sample_result_get_edge_weight((const cugraph_sample_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_hop(const hipgraph_sample_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_sample_result_get_hop((const cugraph_sample_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_label_hop_offsets(const hipgraph_sample_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_sample_result_get_label_hop_offsets((const cugraph_sample_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_label_type_hop_offsets(const hipgraph_sample_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_sample_result_get_label_type_hop_offsets((const cugraph_sample_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_index(const hipgraph_sample_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_sample_result_get_index((const cugraph_sample_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_offsets(const hipgraph_sample_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_sample_result_get_offsets((const cugraph_sample_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_renumber_map(const hipgraph_sample_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_sample_result_get_renumber_map((const cugraph_sample_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_renumber_map_offsets(const hipgraph_sample_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_sample_result_get_renumber_map_offsets((const cugraph_sample_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_edge_renumber_map(const hipgraph_sample_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_sample_result_get_edge_renumber_map((const cugraph_sample_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_sample_result_get_edge_renumber_map_offsets(const hipgraph_sample_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_sample_result_get_edge_renumber_map_offsets((const cugraph_sample_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT void hipgraph_sample_result_free(hipgraph_sample_result_t* result)
{
    cugraph_sample_result_free((cugraph_sample_result_t*)result);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_test_sample_result_create(const hipgraph_resource_handle_t* handle, const hipgraph_type_erased_device_array_view_t* srcs, const hipgraph_type_erased_device_array_view_t* dsts, const hipgraph_type_erased_device_array_view_t* edge_id, const hipgraph_type_erased_device_array_view_t* edge_type, const hipgraph_type_erased_device_array_view_t* wgt, const hipgraph_type_erased_device_array_view_t* hop, const hipgraph_type_erased_device_array_view_t* label, hipgraph_sample_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_test_sample_result_create((const cugraph_resource_handle_t*)handle, (const cugraph_type_erased_device_array_view_t*)srcs, (const cugraph_type_erased_device_array_view_t*)dsts, (const cugraph_type_erased_device_array_view_t*)edge_id, (const cugraph_type_erased_device_array_view_t*)edge_type, (const cugraph_type_erased_device_array_view_t*)wgt, (const cugraph_type_erased_device_array_view_t*)hop, (const cugraph_type_erased_device_array_view_t*)label, (cugraph_sample_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_test_uniform_neighborhood_sample_result_create(const hipgraph_resource_handle_t* handle, const hipgraph_type_erased_device_array_view_t* srcs, const hipgraph_type_erased_device_array_view_t* dsts, const hipgraph_type_erased_device_array_view_t* edge_id, const hipgraph_type_erased_device_array_view_t* edge_type, const hipgraph_type_erased_device_array_view_t* weight, const hipgraph_type_erased_device_array_view_t* hop, const hipgraph_type_erased_device_array_view_t* label, hipgraph_sample_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_test_uniform_neighborhood_sample_result_create((const cugraph_resource_handle_t*)handle, (const cugraph_type_erased_device_array_view_t*)srcs, (const cugraph_type_erased_device_array_view_t*)dsts, (const cugraph_type_erased_device_array_view_t*)edge_id, (const cugraph_type_erased_device_array_view_t*)edge_type, (const cugraph_type_erased_device_array_view_t*)weight, (const cugraph_type_erased_device_array_view_t*)hop, (const cugraph_type_erased_device_array_view_t*)label, (cugraph_sample_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_select_random_vertices(const hipgraph_resource_handle_t* handle, const hipgraph_graph_t* graph, hipgraph_rng_state_t* rng_state, size_t num_vertices, hipgraph_type_erased_device_array_t** vertices, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_select_random_vertices((const cugraph_resource_handle_t*)handle, (const cugraph_graph_t*)graph, (cugraph_rng_state_t*)rng_state, num_vertices, (cugraph_type_erased_device_array_t**)vertices, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_negative_sampling(const hipgraph_resource_handle_t* handle, hipgraph_rng_state_t* rng_state, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* vertices, const hipgraph_type_erased_device_array_view_t* src_biases, const hipgraph_type_erased_device_array_view_t* dst_biases, size_t num_samples, bool remove_duplicates, bool remove_existing_edges, bool exact_number_of_samples, bool do_expensive_check, hipgraph_coo_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_negative_sampling((const cugraph_resource_handle_t*)handle, (cugraph_rng_state_t*)rng_state, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)vertices, (const cugraph_type_erased_device_array_view_t*)src_biases, (const cugraph_type_erased_device_array_view_t*)dst_biases, num_samples, (bool_t)remove_duplicates, (bool_t)remove_existing_edges, (bool_t)exact_number_of_samples, (bool_t)do_expensive_check, (cugraph_coo_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}




