// -*- C -*-
// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common.h"
#include <cugraph_c/error.h>
#include <hipgraph_c/error.h>
#include <cugraph_c/graph.h>
#include <hipgraph_c/graph.h>
#include <cugraph_c/resource_handle.h>
#include <hipgraph_c/resource_handle.h>
#include <cugraph_c/hipgraph/hipgraph-common.h>
#include <hipgraph_c/hipgraph/hipgraph-common.h>
#include "error_code.inl.h"

HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_paths_result_get_vertices(hipgraph_paths_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_paths_result_get_vertices((cugraph_paths_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_paths_result_get_distances(hipgraph_paths_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_paths_result_get_distances((cugraph_paths_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_paths_result_get_predecessors(hipgraph_paths_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_paths_result_get_predecessors((cugraph_paths_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT void hipgraph_paths_result_free(hipgraph_paths_result_t* result)
{
    cugraph_paths_result_free((cugraph_paths_result_t*)result);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_bfs(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, hipgraph_type_erased_device_array_view_t* sources, bool direction_optimizing, size_t depth_limit, bool compute_predecessors, bool do_expensive_check, hipgraph_paths_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_bfs((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (cugraph_type_erased_device_array_view_t*)sources, (bool_t)direction_optimizing, depth_limit, (bool_t)compute_predecessors, (bool_t)do_expensive_check, (cugraph_paths_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_sssp(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, size_t source, double cutoff, bool compute_predecessors, bool do_expensive_check, hipgraph_paths_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_sssp((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, source, cutoff, (bool_t)compute_predecessors, (bool_t)do_expensive_check, (cugraph_paths_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT hipgraph_error_code_t hipgraph_extract_paths(const hipgraph_resource_handle_t* handle, hipgraph_graph_t* graph, const hipgraph_type_erased_device_array_view_t* sources, const hipgraph_paths_result_t* paths_result, const hipgraph_type_erased_device_array_view_t* destinations, hipgraph_extract_paths_result_t** result, hipgraph_error_t** error)
{
    cugraph_error_code_t out;
    out = cugraph_extract_paths((const cugraph_resource_handle_t*)handle, (cugraph_graph_t*)graph, (const cugraph_type_erased_device_array_view_t*)sources, (const cugraph_paths_result_t*)paths_result, (const cugraph_type_erased_device_array_view_t*)destinations, (cugraph_extract_paths_result_t**)result, (cugraph_error_t**)error);
    return _cugraph_to_hipgraph_error_code_t(out);
}


HIPGRAPH_EXPORT size_t hipgraph_extract_paths_result_get_max_path_length(hipgraph_extract_paths_result_t* result)
{
    size_t out;
    out = cugraph_extract_paths_result_get_max_path_length((cugraph_extract_paths_result_t*)result);
    return (size_t)out;
}


HIPGRAPH_EXPORT hipgraph_type_erased_device_array_view_t* hipgraph_extract_paths_result_get_paths(hipgraph_extract_paths_result_t* result)
{
    cugraph_type_erased_device_array_view_t* out;
    out = cugraph_extract_paths_result_get_paths((cugraph_extract_paths_result_t*)result);
    return (hipgraph_type_erased_device_array_view_t*)out;
}


HIPGRAPH_EXPORT void hipgraph_extract_paths_result_free(hipgraph_extract_paths_result_t* result)
{
    cugraph_extract_paths_result_free((cugraph_extract_paths_result_t*)result);
}




