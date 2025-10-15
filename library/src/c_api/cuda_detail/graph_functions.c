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

HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_create_vertex_pairs(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* first, const hipgraph_type_erased_device_array_view_t* second, hipgraph_vertex_pairs_t** vertex_pairs, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_create_vertex_pairs((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)first, (const cugraph_type_erased_device_array_view_t*)second, (cugraph_vertex_pairs_t**)vertex_pairs, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_vertex_pairs_get_first(hipgraph_vertex_pairs_t* vertex_pairs)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_vertex_pairs_get_first((cugraph_vertex_pairs_t*)vertex_pairs);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_vertex_pairs_get_second(hipgraph_vertex_pairs_t* vertex_pairs)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_vertex_pairs_get_second((cugraph_vertex_pairs_t*)vertex_pairs);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT void hipgraph_vertex_pairs_free(hipgraph_vertex_pairs_t* vertex_pairs)
{
    cugraph_vertex_pairs_free((cugraph_vertex_pairs_t*)vertex_pairs);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_two_hop_neighbors(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* start_vertices, bool do_expensive_check, hipgraph_vertex_pairs_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_two_hop_neighbors((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)start_vertices, (bool_t)do_expensive_check, (cugraph_vertex_pairs_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_induced_subgraph_get_sources(hipgraph_induced_subgraph_result_t* induced_subgraph)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_induced_subgraph_get_sources((cugraph_induced_subgraph_result_t*)induced_subgraph);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_induced_subgraph_get_destinations(hipgraph_induced_subgraph_result_t* induced_subgraph)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_induced_subgraph_get_destinations((cugraph_induced_subgraph_result_t*)induced_subgraph);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_induced_subgraph_get_edge_weights(hipgraph_induced_subgraph_result_t* induced_subgraph)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_induced_subgraph_get_edge_weights((cugraph_induced_subgraph_result_t*)induced_subgraph);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_induced_subgraph_get_edge_ids(hipgraph_induced_subgraph_result_t* induced_subgraph)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_induced_subgraph_get_edge_ids((cugraph_induced_subgraph_result_t*)induced_subgraph);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_induced_subgraph_get_edge_type_ids(hipgraph_induced_subgraph_result_t* induced_subgraph)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_induced_subgraph_get_edge_type_ids((cugraph_induced_subgraph_result_t*)induced_subgraph);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_induced_subgraph_get_subgraph_offsets(hipgraph_induced_subgraph_result_t* induced_subgraph)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_induced_subgraph_get_subgraph_offsets((cugraph_induced_subgraph_result_t*)induced_subgraph);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT void hipgraph_induced_subgraph_result_free(hipgraph_induced_subgraph_result_t* induced_subgraph)
{
    cugraph_induced_subgraph_result_free((cugraph_induced_subgraph_result_t*)induced_subgraph);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_extract_induced_subgraph(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* subgraph_offsets, const hipgraph_type_erased_device_array_view_t* subgraph_vertices, bool do_expensive_check, hipgraph_induced_subgraph_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_extract_induced_subgraph((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)subgraph_offsets, (const cugraph_type_erased_device_array_view_t*)subgraph_vertices, (bool_t)do_expensive_check, (cugraph_induced_subgraph_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_allgather(const hipgraph_resource_handle_t* handle, const hipgraph_type_erased_device_array_view_t* src, const hipgraph_type_erased_device_array_view_t* dst, const hipgraph_type_erased_device_array_view_t* weights, const hipgraph_type_erased_device_array_view_t* edge_ids, const hipgraph_type_erased_device_array_view_t* edge_type_ids, hipgraph_induced_subgraph_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_allgather((const cugraph_resource_handle_t*)handle, (const cugraph_type_erased_device_array_view_t*)src, (const cugraph_type_erased_device_array_view_t*)dst, (const cugraph_type_erased_device_array_view_t*)weights, (const cugraph_type_erased_device_array_view_t*)edge_ids, (const cugraph_type_erased_device_array_view_t*)edge_type_ids, (cugraph_induced_subgraph_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_count_multi_edges(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, bool do_expensive_check, size_t* result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_count_multi_edges((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (bool_t)do_expensive_check, result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_in_degrees(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* source_vertices, bool do_expensive_check, hipgraph_degrees_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_in_degrees((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)source_vertices, (bool_t)do_expensive_check, (cugraph_degrees_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_out_degrees(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* source_vertices, bool do_expensive_check, hipgraph_degrees_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_out_degrees((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)source_vertices, (bool_t)do_expensive_check, (cugraph_degrees_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_degrees(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* source_vertices, bool do_expensive_check, hipgraph_degrees_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_degrees((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)source_vertices, (bool_t)do_expensive_check, (cugraph_degrees_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_degrees_result_get_vertices(hipgraph_degrees_result_t* degrees_result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_degrees_result_get_vertices((cugraph_degrees_result_t*)degrees_result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_degrees_result_get_in_degrees(hipgraph_degrees_result_t* degrees_result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_degrees_result_get_in_degrees((cugraph_degrees_result_t*)degrees_result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_degrees_result_get_out_degrees(hipgraph_degrees_result_t* degrees_result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_degrees_result_get_out_degrees((cugraph_degrees_result_t*)degrees_result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT void hipgraph_degrees_result_free(hipgraph_degrees_result_t* degrees_result)
{
    cugraph_degrees_result_free((cugraph_degrees_result_t*)degrees_result);
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_edgelist_get_sources(hipgraph_edgelist_t* edgelist)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_edgelist_get_sources((cugraph_edgelist_t*)edgelist);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_edgelist_get_destinations(hipgraph_edgelist_t* edgelist)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_edgelist_get_destinations((cugraph_edgelist_t*)edgelist);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_edgelist_get_edge_weights(hipgraph_edgelist_t* edgelist)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_edgelist_get_edge_weights((cugraph_edgelist_t*)edgelist);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_edgelist_get_edge_ids(hipgraph_edgelist_t* edgelist)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_edgelist_get_edge_ids((cugraph_edgelist_t*)edgelist);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_edgelist_get_edge_type_ids(hipgraph_edgelist_t* edgelist)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_edgelist_get_edge_type_ids((cugraph_edgelist_t*)edgelist);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_edgelist_get_edge_offsets(hipgraph_edgelist_t* edgelist)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_edgelist_get_edge_offsets((cugraph_edgelist_t*)edgelist);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT void hipgraph_edgelist_free(hipgraph_edgelist_t* edgelist)
{
    cugraph_edgelist_free((cugraph_edgelist_t*)edgelist);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_decompress_to_edgelist(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, bool do_expensive_check, hipgraph_edgelist_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_decompress_to_edgelist((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (bool_t)do_expensive_check, (cugraph_edgelist_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_renumber_arbitrary_edgelist(const hipgraph_resource_handle_t* handle, const hipgraph_type_erased_host_array_view_t* renumber_map, hipgraph_type_erased_device_array_view_t* srcs, hipgraph_type_erased_device_array_view_t* dsts, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_renumber_arbitrary_edgelist((const cugraph_resource_handle_t*)handle, (const cugraph_type_erased_host_array_view_t*)renumber_map, (cugraph_type_erased_device_array_view_t*)srcs, (cugraph_type_erased_device_array_view_t*)dsts, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}




